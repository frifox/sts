"""Wires capture -> VAD -> ASR -> translation -> TTS -> playback into one
streaming pipeline, each stage on its own thread connected by queues so a
slow stage (translation, TTS) doesn't stall audio capture.

Recognition/translation/captions run by default (`self.control.active` set);
an operator can still pause them (e.g. during a long musical number) via
Start/Stop. TTS *voicing* is separate and starts **off**
(`self.control.voice_enabled` unset) - captions are silent and harmless to
leave running, but audio spoken over the speakers during music or an
announcement would not be, so that's opt-in via its own button. Capture and
VAD segmentation always run regardless of either switch, both because
they're cheap and because voice-clone training needs segmented audio
independently of whether recognition/voicing is on.
"""

import queue
import tempfile
import threading
import time
import wave

import numpy as np

from sts.asr.whisper_asr import WhisperASR
from sts.audio.capture import AudioCapture
from sts.audio.playback import AudioPlayback
from sts.audio.vad import SpeechSegmenter
from sts.bible.lookup import BibleLookup
from sts.bible.reference import find_references
from sts.config import PipelineConfig
from sts.control import RecognitionControl
from sts.log import vlog
from sts.translate.translator import Translator
from sts.tts import create_synthesizer
from sts.tts.voice_library import VoiceLibrary
from sts.web.server import CaptionBroadcaster


class Pipeline:
    def __init__(self, cfg: PipelineConfig, broadcaster: CaptionBroadcaster | None = None):
        self.cfg = cfg
        self.broadcaster = broadcaster
        self.capture = AudioCapture(cfg.audio)
        self.playback = AudioPlayback(cfg.audio)
        self._stage("Loading speech detector…")
        self.segmenter = SpeechSegmenter(cfg.audio, cfg.vad)
        self._stage(f"Loading speech recognition ({cfg.asr.model_size})…")
        self.asr = WhisperASR(cfg.asr)
        self._stage("Loading translation model…")
        self.translator = Translator(cfg.translation)
        self._stage(f"Loading voice synthesis ({cfg.tts.backend})…")
        self.tts = create_synthesizer(cfg.tts)
        self._stage("Loading voice library and Bible data…")
        self.voice_library = VoiceLibrary(cfg.tts.voice_library_dir)
        self.bible_lookup = BibleLookup()
        self.control = RecognitionControl(cfg.audio.sample_rate)
        self._training_name: str | None = None
        self._active_speaker_id: str | None = None

        self._segments: queue.Queue = queue.Queue()
        self._transcripts: queue.Queue = queue.Queue()
        self._translations: queue.Queue = queue.Queue()
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()

    def _stage(self, text: str, ready: bool = False) -> None:
        vlog("init", text)
        if self.broadcaster:
            self.broadcaster.broadcast_stage(text, ready)

    def clone_voice(self, reference_wav_path: str, name: str | None = None) -> None:
        """Zero-shot clones a pre-supplied reference clip (the `--clone` CLI
        flag). Without `name`, this is a one-off clone for this run only -
        same as always. With `name`, it's also saved into the voice library
        (transcribed first, for backends that use a reference transcript),
        so it's reusable from the dropdown in later sessions without
        reprocessing the file - the CLI equivalent of the "Train voice"
        button, for when you already have a clean recording instead of
        capturing one live. Zero-shot cloning is often good enough on its
        own; this just means not having to redo it every run.
        """
        if not name:
            self.tts.clone_voice(reference_wav_path)
            return

        self._stage(f"Preparing voice '{name}'…")
        ref_text = self._transcribe_or_none(reference_wav_path)
        self._save_and_activate(name, reference_wav_path, ref_text)
        self._broadcast_status()

    # -- operator controls, invoked from the caption page's buttons --------

    def handle_control(self, data: dict) -> None:
        action = data.get("action")
        vlog("control", f"operator action: {action}")
        if action == "start":
            self.control.start()
        elif action == "stop":
            self.control.stop()
        elif action == "voice_on":
            self.control.enable_voice()
        elif action == "voice_off":
            self.control.disable_voice()
        elif action == "reset":
            self._reset()
        elif action == "train_start":
            self._training_name = (data.get("name") or "").strip() or f"speaker-{int(time.time())}"
            self.control.start_training()
        elif action == "train_stop":
            self._finish_training()
        elif action == "default_voice":
            voice = data.get("voice")
            if voice not in self.cfg.tts.builtin_voices:
                return
            self.cfg.tts.default_voice = voice
            self.tts.set_default_voice(voice)
            self._select_speaker(None)  # picking a built-in voice drops any cloned/trained one
        elif action == "select_speaker":
            self._select_speaker(data.get("speaker_id"))
        else:
            return
        self._broadcast_status()

    def _reset(self) -> None:
        """New speaker / performance: clear captions, drop stale in-flight
        text from the previous one, and forget any cloned voice."""
        if self.control.training.is_set():
            self.control.stop_training(discard=True)
        self.tts.reset_voice()
        self._active_speaker_id = None
        self._drain(self._segments)
        self._drain(self._transcripts)
        self._drain(self._translations)
        if self.broadcaster:
            self.broadcaster.broadcast_reset()

    def _select_speaker(self, speaker_id: str | None) -> None:
        if not speaker_id:
            self.tts.reset_voice()
            self._active_speaker_id = None
            return
        entry = self.voice_library.get(speaker_id)
        if entry is not None:
            self.tts.clone_from_entry(entry)
            self._active_speaker_id = speaker_id

    def _finish_training(self, discard: bool = False) -> None:
        audio = self.control.stop_training(discard=discard)
        name, self._training_name = self._training_name, None
        if audio is None:
            return

        path = self._write_training_wav(audio)
        ref_text = self._transcribe_or_none(audio)
        self._save_and_activate(name or "speaker", path, ref_text)

    def _transcribe_or_none(self, audio) -> str | None:
        try:
            return self.asr.transcribe(audio) or None
        except Exception as exc:  # a failed transcript just means a lower-quality clone, not a lost one
            print(f"[voice] couldn't transcribe reference clip, falling back to embedding-only clone: {exc}")
            return None

    def _save_and_activate(self, name: str, wav_source: str, ref_text: str | None) -> None:
        entry = self.voice_library.save(name, wav_source, ref_text)
        self.tts.clone_from_entry(entry)
        self._active_speaker_id = entry.speaker_id
        self._broadcast_library()

    def _write_training_wav(self, audio: np.ndarray) -> str:
        path = tempfile.NamedTemporaryFile(prefix="sts_voice_", suffix=".wav", delete=False).name
        pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        with wave.open(path, "wb") as wav_file:
            wav_file.setnchannels(self.cfg.audio.channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.cfg.audio.sample_rate)
            wav_file.writeframes(pcm16.tobytes())
        return path

    @staticmethod
    def _drain(q: queue.Queue) -> None:
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                break

    def _broadcast_status(self) -> None:
        if self.broadcaster:
            self.broadcaster.broadcast_status(
                active=self.control.active.is_set(),
                voice_enabled=self.control.voice_enabled.is_set(),
                training=self.control.training.is_set(),
                training_seconds=self.control.training_seconds,
                training_target_seconds=self.control.target_seconds,
                cloned=self.tts.cloned_speaker_wav is not None,
                default_voice=self.cfg.tts.default_voice,
                active_speaker_id=self._active_speaker_id,
            )

    def _broadcast_library(self) -> None:
        if self.broadcaster:
            speakers = [
                {"id": e.speaker_id, "name": e.name} for e in self.voice_library.list()
            ]
            self.broadcaster.broadcast_library(speakers)

    # -- pipeline stages -----------------------------------------------

    def _segment_loop(self) -> None:
        while not self._stop.is_set():
            frame = self.capture.frames.get()
            segment = self.segmenter.process(frame)
            if segment is None:
                continue
            if self.control.training.is_set():
                reached_target = self.control.accumulate_training_segment(segment)
                self._broadcast_status()
                if reached_target:
                    self._finish_training()
                    self._broadcast_status()
            if self.control.active.is_set():
                self._segments.put(segment)

    def _asr_loop(self) -> None:
        while not self._stop.is_set():
            segment = self._segments.get()
            t0 = time.monotonic()
            text = self.asr.transcribe(segment)
            vlog("asr", f"transcribed {len(segment) / self.cfg.audio.sample_rate:.1f}s of audio in {time.monotonic() - t0:.2f}s")
            if text:
                print(f"[ru] {text}")
                if self.broadcaster:
                    self.broadcaster.broadcast_original(text)
                    self._broadcast_bible_references(text)
                self._transcripts.put(text)

    def _broadcast_bible_references(self, text: str) -> None:
        for ref in find_references(text):
            quote = self.bible_lookup.lookup(ref.book, ref.ru_label, ref.chapter, ref.verse, ref.verse_end)
            print(f"[verse] {quote.ru_label} {ref.chapter}:{ref.verse}")
            self.broadcaster.broadcast_verse(
                quote.ru_label, quote.chapter, quote.verse, quote.verse_end, quote.ru_text, quote.en_text
            )

    def _translate_loop(self) -> None:
        while not self._stop.is_set():
            text = self._transcripts.get()
            t0 = time.monotonic()
            translated = self.translator.translate(text)
            vlog("translate", f"translated in {time.monotonic() - t0:.2f}s")
            if translated:
                print(f"[en] {translated}")
                if self.broadcaster:
                    self.broadcaster.broadcast_translated(translated)
                self._translations.put(translated)

    def _tts_loop(self) -> None:
        while not self._stop.is_set():
            text = self._translations.get()
            if not self.control.voice_enabled.is_set():
                vlog("tts", "voice is off, skipping synthesis")
                continue
            t0 = time.monotonic()
            audio, sample_rate = self.tts.synthesize(text)
            vlog("tts", f"synthesized {audio.size / sample_rate:.1f}s of speech in {time.monotonic() - t0:.2f}s")
            if audio.size:
                self.playback.play(audio, sample_rate=sample_rate, blocking=True)

    def start(self) -> None:
        self.capture.start()
        self._stage("Listening", ready=True)
        for target in (
            self._segment_loop,
            self._asr_loop,
            self._translate_loop,
            self._tts_loop,
        ):
            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            self._threads.append(thread)
        self._broadcast_status()
        self._broadcast_library()

    def stop(self) -> None:
        self._stop.set()
        self.capture.stop()
