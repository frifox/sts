"""Wires capture -> VAD -> ASR -> translation -> TTS -> playback into one
streaming pipeline, each stage on its own thread connected by queues so a
slow stage (translation, TTS) doesn't stall audio capture.

Recognition starts paused (`self.control.active` unset) - a live feed
carries long stretches of music, announcements, and chatter that don't need
translating, so an operator turns it on only while someone's actually
preaching. Capture and VAD segmentation always run regardless, both because
they're cheap and because voice-clone training needs segmented audio
independently of whether recognition is on.
"""

import queue
import tempfile
import threading
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
from sts.translate.translator import Translator
from sts.tts.xtts import SAMPLE_RATE, XTTSSynthesizer
from sts.web.server import CaptionBroadcaster


class Pipeline:
    def __init__(self, cfg: PipelineConfig, broadcaster: CaptionBroadcaster | None = None):
        self.cfg = cfg
        self.capture = AudioCapture(cfg.audio)
        self.playback = AudioPlayback(cfg.audio)
        self.segmenter = SpeechSegmenter(cfg.audio, cfg.vad)
        self.asr = WhisperASR(cfg.asr)
        self.translator = Translator(cfg.translation)
        self.tts = XTTSSynthesizer(cfg.tts)
        self.bible_lookup = BibleLookup()
        self.broadcaster = broadcaster
        self.control = RecognitionControl(cfg.audio.sample_rate)

        self._segments: queue.Queue = queue.Queue()
        self._transcripts: queue.Queue = queue.Queue()
        self._translations: queue.Queue = queue.Queue()
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()

    def clone_voice(self, reference_wav_path: str) -> None:
        self.tts.clone_voice(reference_wav_path)

    # -- operator controls, invoked from the caption page's buttons --------

    def handle_control(self, action: str) -> None:
        if action == "start":
            self.control.start()
        elif action == "stop":
            self.control.stop()
        elif action == "reset":
            self._reset()
        elif action == "train_start":
            self.control.start_training()
        elif action == "train_stop":
            self._finish_training()
        else:
            return
        self._broadcast_status()

    def _reset(self) -> None:
        """New speaker / performance: clear captions, drop stale in-flight
        text from the previous one, and forget any cloned voice."""
        if self.control.training.is_set():
            self.control.stop_training(discard=True)
        self.tts.reset_voice()
        self._drain(self._segments)
        self._drain(self._transcripts)
        self._drain(self._translations)
        if self.broadcaster:
            self.broadcaster.broadcast_reset()

    def _finish_training(self, discard: bool = False) -> None:
        audio = self.control.stop_training(discard=discard)
        if audio is not None:
            path = self._write_training_wav(audio)
            self.tts.clone_voice(path)

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
                training=self.control.training.is_set(),
                training_seconds=self.control.training_seconds,
                training_target_seconds=self.control.target_seconds,
                cloned=self.tts.cloned_speaker_wav is not None,
            )

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
            text = self.asr.transcribe(segment)
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
            translated = self.translator.translate(text)
            if translated:
                print(f"[en] {translated}")
                if self.broadcaster:
                    self.broadcaster.broadcast_translated(translated)
                self._translations.put(translated)

    def _tts_loop(self) -> None:
        while not self._stop.is_set():
            text = self._translations.get()
            audio = self.tts.synthesize(text)
            if audio.size:
                self.playback.play(audio, sample_rate=SAMPLE_RATE, blocking=True)

    def start(self) -> None:
        self.capture.start()
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

    def stop(self) -> None:
        self._stop.set()
        self.capture.stop()
