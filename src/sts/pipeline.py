"""Wires capture -> VAD -> ASR -> translation -> TTS -> playback into one
streaming pipeline, each stage on its own thread connected by queues so a
slow stage (translation, TTS) doesn't stall audio capture.

Recognition/captions run by default (`self.control.active` set); an operator
can still pause them (e.g. during a long musical number) via Start/Stop.
Translation is a separate switch (`self.control.translation_enabled`, on by
default): turning it off leaves the original-language captions running but
skips translation (and, as a consequence, TTS, since there is no translated
text to speak). TTS *voicing* is its own switch and starts **off**
(`self.control.voice_enabled` unset) - captions are silent and harmless to
leave running, but audio spoken over the speakers during music or an
announcement would not be, so that's opt-in via its own button. Capture and
VAD segmentation always run regardless of any of these switches, both because
they're cheap and because voice-clone training needs segmented audio
independently of whether recognition/translation/voicing is on.
"""

import queue
import struct
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
from sts.tts.voice_library import BUNDLED_VOICES_DIR, VoiceLibrary
from sts.web.server import CaptionBroadcaster

# Bounds for the live pause-threshold slider (milliseconds of silence that end a snippet).
MIN_SILENCE_RANGE_MS = (200, 1500)

# One binary mic frame from a browser: a 4-byte LE sample-rate header (mirrors
# the outgoing TTS format in CaptionBroadcaster.broadcast_audio) followed by
# raw float32 LE mono PCM - one block_ms worth of samples, matching what the
# VAD needs exactly.
_MIC_FRAME_HEADER = struct.Struct("<I")

# What the designed voice says when its reference clip is generated. The clip
# becomes the saved voice's reference, so it should be a natural, moderately
# long passage (about 8-10 s), and it must match the text passed to the model.
DESIGN_REFERENCE_TEXT = (
    "Brothers and sisters, we gather today to remember God's grace, "
    "and to rest in the peace that He gives to all who trust in Him."
)


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
        self.voice_library = VoiceLibrary(cfg.tts.voice_library_dir, bundled_dir=BUNDLED_VOICES_DIR)
        self.bible_lookup = BibleLookup()
        self.control = RecognitionControl(cfg.audio.sample_rate)
        # A freshly trained voice is active immediately but unsaved until the
        # operator names it (the "Save Voice" button): (wav path, transcript).
        self._pending_voice: tuple[str, str | None] | None = None
        # "Create Voice" dialog: the designed voice's reference clip + its text.
        self._design: tuple[str, str] | None = None
        self._design_busy = threading.Lock()
        self._device_lock = threading.Lock()
        self._active_speaker_id: str | None = None

        self._segments: queue.Queue = queue.Queue()
        self._transcripts: queue.Queue = queue.Queue()
        self._translations: queue.Queue = queue.Queue()
        self._audio: queue.Queue = queue.Queue()  # synthesized clips waiting to play
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()

        if cfg.tts.startup_voice:
            self._stage("Preparing the default voice…")
        self._use_startup_voice()

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
            # AudioPlayback.play() only blocks until there's room in the output
            # buffer, not until the clip finishes - so by the time this fires,
            # a sentence or two already queued ahead of it may be sitting in
            # that buffer waiting to play. Drop what's still in our own queue
            # (already covered by _playback_loop's own voice_enabled check,
            # but no reason to let it sit there) and cut off whatever's
            # already been handed to the sound card.
            self._drain(self._audio)
            self.playback.stop()
        elif action == "translate_on":
            self.control.enable_translation()
        elif action == "translate_off":
            self.control.disable_translation()
        elif action == "reset":
            self._reset()
        elif action == "train_start":
            self.control.start_training()
        elif action == "train_stop":
            self._finish_training()
        elif action in ("design_generate", "design_play", "design_save", "design_close"):
            self._handle_design_action(action, data)
        elif action == "save_voice":
            self._save_pending_voice(str(data.get("name") or ""))
        elif action == "default_voice":
            voice = self.cfg.tts.resolve_voice(str(data.get("voice") or ""))
            if voice not in self.cfg.tts.all_voices():
                return
            self.cfg.tts.default_voice = voice
            self.tts.set_default_voice(voice)
            self._select_speaker(None)  # picking a built-in voice drops any cloned/trained one
        elif action in ("refresh_devices", "set_audio_devices", "mic_start", "mic_stop"):
            threading.Thread(target=self._handle_device_action, args=(action, data), daemon=True).start()
            return
        elif action == "set_min_silence":
            try:
                ms = int(data.get("ms"))
            except (TypeError, ValueError):
                return
            # Read by the voice detector on every frame, so this applies to the very next pause.
            self.cfg.vad.min_silence_ms = max(MIN_SILENCE_RANGE_MS[0], min(MIN_SILENCE_RANGE_MS[1], ms))
            vlog("vad", f"pause threshold set to {self.cfg.vad.min_silence_ms} ms")
        elif action == "delete_voice":
            self._delete_voice(str(data.get("speaker_id") or ""))
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
        self._use_startup_voice()
        self._pending_voice = None
        self._drain(self._segments)
        self._drain(self._transcripts)
        self._drain(self._translations)
        self._drain(self._audio)
        self.playback.stop()
        if self.broadcaster:
            self.broadcaster.broadcast_reset()

    def _use_startup_voice(self) -> None:
        """Back to the voice the session starts with: the bundled "Default" voice
        (or the plain built-in voice if that is disabled or missing)."""
        wanted = self.cfg.tts.startup_voice
        self._select_speaker(wanted if wanted and self.voice_library.get(wanted) else None)

    def _select_speaker(self, speaker_id: str | None) -> None:
        self._pending_voice = None
        if not speaker_id:
            self.tts.reset_voice()
            self._active_speaker_id = None
            return
        entry = self.voice_library.get(speaker_id)
        if entry is not None:
            self.tts.clone_from_entry(entry)
            self._active_speaker_id = speaker_id

    def _delete_voice(self, speaker_id: str) -> None:
        entry = self.voice_library.get(speaker_id)
        if entry is None:
            return
        if entry.builtin:
            return  # shipped with the app; only a saved copy the user made of it can be deleted
        was_active = self._active_speaker_id == speaker_id
        if was_active:
            self._select_speaker(None)  # step off the voice before its files go away
        forget = getattr(self.tts, "forget_voice", None)
        if forget:
            forget(speaker_id)
        self.voice_library.delete(speaker_id)
        if was_active:
            self._use_startup_voice()  # e.g. deleting a saved override brings the shipped voice back
        self._broadcast_library()
        self._broadcast_status()
        vlog("voice", f"deleted voice {entry.name!r}")

    def _finish_training(self, discard: bool = False) -> None:
        audio = self.control.stop_training(discard=discard)
        if audio is None:
            return

        path = self._write_training_wav(audio)
        ref_text = self._transcribe_or_none(audio)
        # Activate the clone right away so the operator can listen to it, but
        # don't write it to the library until they choose to (Save Voice).
        self.tts.clone_voice(path, ref_text)
        self._active_speaker_id = None
        self._pending_voice = (path, ref_text)
        vlog("voice", "trained voice is active but not saved yet")

    def _save_pending_voice(self, name: str) -> None:
        name = name.strip()
        if self._pending_voice is None or not name:
            return
        path, ref_text = self._pending_voice
        self._save_and_activate(name, path, ref_text)
        self._pending_voice = None
        vlog("voice", f"saved voice {name!r} to the library")

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
        return self._write_wav(audio, self.cfg.audio.sample_rate)

    @staticmethod
    def _write_wav(audio: np.ndarray, sample_rate: int) -> str:
        path = tempfile.NamedTemporaryFile(prefix="sts_voice_", suffix=".wav", delete=False).name
        pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        with wave.open(path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm16.tobytes())
        return path

    # -- audio device selection (the configuration dialog) --------------------

    def _broadcast_devices(self, error: str | None = None, applied: bool = False) -> None:
        if not self.broadcaster:
            return
        from sts.audio.devices import describe_devices

        try:
            message = describe_devices(self.cfg.audio.input_device, self.cfg.audio.output_device)
        except Exception as exc:
            message = {"type": "devices", "inputs": [], "outputs": [], "input": None, "output": None}
            error = error or f"Couldn't list audio devices: {exc}"
        message.update(error=error, applied=applied)
        self.broadcaster.broadcast_devices(message)

    def _handle_device_action(self, action: str, data: dict) -> None:
        # Opening audio streams can take a moment, so this runs off the web server's event loop.
        if action == "refresh_devices":
            self._broadcast_devices()
            return
        if action == "mic_start":
            with self._device_lock:
                self.capture.switch_to_browser()
            vlog("audio", "input switched to this browser's microphone")
            self._broadcast_status()
            return
        if action == "mic_stop":
            with self._device_lock:
                self.capture.switch_to_alsa()
            vlog("audio", "input switched back to the local ALSA device")
            self._broadcast_status()
            return

        def parse(value):
            return None if value in (None, "") else int(value)

        try:
            new_input, new_output = parse(data.get("input")), parse(data.get("output"))
        except (TypeError, ValueError):
            return
        audio = self.cfg.audio
        # Picking an item from the Input dropdown always means "use ALSA with
        # this device" - including when the browser mic was active and the
        # index itself hasn't changed, since capture isn't reading it yet.
        input_changed = new_input != audio.input_device or audio.source != "alsa"
        with self._device_lock:
            try:
                from sts.audio.devices import check_input, check_output

                if input_changed:
                    check_input(new_input, audio.sample_rate, audio.channels)
                if new_output != audio.output_device:
                    check_output(new_output, 24_000)
                if input_changed:
                    self.capture.switch_to_alsa(new_input)
                if new_output != audio.output_device:
                    audio.output_device = new_output
                    self.playback.stop()  # the output stream reopens on the new device with the next clip
            except Exception as exc:
                print(f"[audio] couldn't apply the selected devices: {exc}")
                self._broadcast_devices(error=f"Couldn't use that device: {exc}")
                return
            vlog("audio", f"devices applied: input={audio.input_device}, output={audio.output_device}")
            self._broadcast_devices(applied=True)
        self._broadcast_status()  # audio.source may have just changed too

    def _handle_mic_frame(self, payload: bytes) -> None:
        """One block_ms worth of PCM pushed by the browser tab currently
        acting as the mic source (see the `mic_start` control action and the
        matching client-side AudioWorklet in index.html). Silently dropped
        if it doesn't match what the VAD needs exactly - e.g. a browser that
        didn't honor its AudioContext's forced sample rate."""
        if len(payload) < _MIC_FRAME_HEADER.size:
            return
        (sample_rate,) = _MIC_FRAME_HEADER.unpack_from(payload)
        if sample_rate != self.cfg.audio.sample_rate:
            return
        samples = np.frombuffer(payload, dtype="<f4", offset=_MIC_FRAME_HEADER.size)
        expected_samples = int(self.cfg.audio.sample_rate * self.cfg.audio.block_ms / 1000)
        if len(samples) != expected_samples:
            return
        self.capture.push_browser_frame(samples.copy())  # copy: payload's buffer is reused by the socket

    # -- "Create Voice" dialog (Qwen3 voice design) ---------------------------

    def _design_state(self, state: str, message: str = "") -> None:
        vlog("design", f"{state}: {message}" if message else state)
        if self.broadcaster:
            self.broadcaster.broadcast_design(state, message, has_voice=self._design is not None)

    def _handle_design_action(self, action: str, data: dict) -> None:
        if not hasattr(self.tts, "design_reference"):
            return
        if action == "design_close":
            self._design = None
            self.tts.unload_designer()
            self._design_state("idle")
            return
        if action == "design_save":
            self._design_save(str(data.get("name") or ""))
            return
        # Generating / previewing run for seconds to minutes, and control
        # messages arrive on the web server's event loop, so use a worker thread.
        if not self._design_busy.acquire(blocking=False):
            return  # a previous generate/play is still running
        task = self._design_generate if action == "design_generate" else self._design_play
        argument = str(data.get("prompt" if action == "design_generate" else "text") or "").strip()

        def run() -> None:
            try:
                if argument:
                    task(argument)
            except Exception as exc:
                print(f"[design] {action} failed: {exc!r}")
                self._design_state("error", f"{action.split('_')[1].capitalize()} failed: {exc}")
            finally:
                self._design_busy.release()

        threading.Thread(target=run, daemon=True).start()

    def _design_generate(self, prompt: str) -> None:
        self._design_state(
            "busy",
            "Loading the voice designer (the first time this downloads several GB)…"
            if not self.tts.design_loaded else "Generating voice…",
        )
        audio, sample_rate = self.tts.design_reference(prompt, DESIGN_REFERENCE_TEXT)
        self._design = (self._write_wav(audio, sample_rate), DESIGN_REFERENCE_TEXT)
        self._design_state("ready", "Voice generated. Play a sample to check it, or Generate again for a different one.")

    def _design_play(self, text: str) -> None:
        if self._design is None:
            self._design_state("error", "Generate a voice first.")
            return
        self._design_state("busy", "Synthesizing sample…")
        path, ref_text = self._design
        audio, sample_rate = self.tts.preview(text, path, ref_text)
        self._design_state("busy", "Playing sample…")
        self.playback.play(audio, sample_rate=sample_rate)
        self._design_state("ready", "Play again, tweak the description and Generate, or name it and Save.")

    def _design_save(self, name: str) -> None:
        name = name.strip()
        if self._design is None or not name:
            return
        path, ref_text = self._design
        self._save_and_activate(name, path, ref_text)
        self._pending_voice = None
        self._design = None
        self.tts.unload_designer()
        self._design_state("saved", f"Saved '{name}'.")
        self._broadcast_status()

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
                translation_enabled=self.control.translation_enabled.is_set(),
                voice_enabled=self.control.voice_enabled.is_set(),
                training=self.control.training.is_set(),
                training_seconds=self.control.training_seconds,
                training_target_seconds=self.control.target_seconds,
                cloned=self.tts.cloned_speaker_wav is not None,
                can_save=self._pending_voice is not None,
                min_silence_ms=self.cfg.vad.min_silence_ms,
                audio_source=self.cfg.audio.source,
                can_design=hasattr(self.tts, "design_reference"),
                default_voice=self.cfg.tts.resolve_voice(self.cfg.tts.default_voice),
                voice_groups=self.cfg.tts.voice_groups,
                active_speaker_id=self._active_speaker_id,
            )

    def _broadcast_library(self) -> None:
        if self.broadcaster:
            speakers = [
                {"id": e.speaker_id, "name": e.name, "builtin": e.builtin} for e in self.voice_library.list()
            ]
            self.broadcaster.broadcast_library(speakers)

    def _broadcast_engines(self) -> None:
        """Model/device info for the header dropdowns. Fixed for the life of
        the process (set from CLI flags at startup), so this is only ever
        sent once per connection rather than tracked as live status."""
        if self.broadcaster:
            tts_model = (
                self.cfg.tts.qwen3_model_name if self.cfg.tts.backend == "qwen3" else self.cfg.tts.model_name
            )
            self.broadcaster.broadcast_engines(
                {
                    "asr": {"model": self.cfg.asr.model_size, "device": self.cfg.asr.device},
                    "translation": {"model": self.cfg.translation.model_name, "device": self.cfg.translation.device},
                    "tts": {"model": tts_model, "device": self.cfg.tts.device},
                }
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
            if not self.control.translation_enabled.is_set():
                vlog("translate", "translation is off, skipping")
                continue
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
            try:
                # Each clip is handed to the playback thread as soon as it is
                # ready, so the next sentence synthesizes while this one plays.
                first_piece = True
                for audio, sample_rate in self._synthesize_pieces(text):
                    if not audio.size:
                        continue
                    vlog("tts", f"synthesized {audio.size / sample_rate:.1f}s of speech in "
                                f"{time.monotonic() - t0:.2f}s ({self._translations.qsize()} sentences waiting)")
                    # A longer pause before a new translated text, a shorter one
                    # between the sentences within it.
                    pause = self.cfg.audio.pause_between_s if first_piece else self.cfg.audio.pause_within_s
                    self._audio.put((audio, sample_rate, pause))
                    first_piece = False
                    t0 = time.monotonic()
            except Exception as exc:  # one bad sentence shouldn't kill voicing for the rest of the session
                print(f"[tts] synthesis failed, skipping this sentence: {exc!r}")

    def _synthesize_pieces(self, text: str):
        stream = getattr(self.tts, "synthesize_stream", None)
        if stream is not None:
            yield from stream(text)
        else:
            yield self.tts.synthesize(text)

    def _playback_loop(self) -> None:
        while not self._stop.is_set():
            audio, sample_rate, pause_before = self._audio.get()
            if not self.control.voice_enabled.is_set():
                continue  # voice was switched off while this was queued
            try:
                self.playback.play(audio, sample_rate=sample_rate, pause_before=pause_before)
            except Exception as exc:
                print(f"[audio] playback failed: {exc!r}")
            if self.broadcaster:
                self.broadcaster.broadcast_audio(audio, sample_rate)

    def start(self) -> None:
        self.capture.start()
        self._stage("Listening", ready=True)
        for target in (
            self._segment_loop,
            self._asr_loop,
            self._translate_loop,
            self._tts_loop,
            self._playback_loop,
        ):
            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            self._threads.append(thread)
        self._broadcast_status()
        self._broadcast_library()
        self._broadcast_devices()
        self._broadcast_engines()

    def stop(self) -> None:
        self._stop.set()
        self.capture.stop()
        self.playback.stop()
