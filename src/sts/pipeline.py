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

import gc
import queue
import struct
import tempfile
import threading
import time
import wave
from collections import deque

import numpy as np
import torch

from sts.asr import create_asr, parakeet_asr
from sts.audio.capture import AudioCapture
from sts.audio.playback import AudioPlayback
from sts.audio.vad import SpeechSegmenter
from sts.bible.lookup import BibleLookup
from sts.bible.reference import find_references
from sts.config import PipelineConfig
from sts.control import RecognitionControl
from sts.log import vlog
from sts.persistent_config import PersistentConfig
from sts.textclean import split_complete
from sts.translate.translator import Translator, split_sentences
from sts.tts import create_synthesizer
from sts.tts.voice_library import BUNDLED_VOICES_DIR, VoiceLibrary
from sts.web.server import CaptionBroadcaster

# Bounds for the live pause-threshold slider (milliseconds of silence that end a snippet).
MIN_SILENCE_RANGE_MS = (200, 1500)

# How far back the header's RTF readings look (seconds) - a single short
# phrase's processing time can spike a one-shot RTF wildly (e.g. a 0.3s
# utterance that took 0.4s to transcribe reads as 1.3x even though the stage
# is nowhere near actually falling behind), so what's shown is smoothed over
# a rolling window instead of the latest reading alone.
RTF_WINDOW_S = 30.0

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


class _NullASR:
    """Last-resort stand-in when a live ASR backend switch fails and the
    fallback reload of the previous engine fails too (see
    Pipeline._switch_asr_backend) - keeps _asr_loop's thread alive instead of
    crashing on a missing self.asr, at the cost of transcribing nothing until
    the operator retries or restarts the service."""

    def transcribe(self, audio) -> str:
        return ""


class Pipeline:
    def __init__(
        self,
        cfg: PipelineConfig,
        broadcaster: CaptionBroadcaster | None = None,
        persistent_config: PersistentConfig | None = None,
    ):
        self.cfg = cfg
        self.broadcaster = broadcaster
        # Remembers the operator's ASR engine/device/voice choices across
        # restarts - see the persist-on-change calls below and
        # _restore_persisted_voice(). None (the default) just means nothing
        # is persisted, e.g. in tests that build a Pipeline stub directly.
        self._persistent_config = persistent_config
        self.capture = AudioCapture(cfg.audio)
        self.playback = AudioPlayback(cfg.audio)
        self._stage("Loading speech detector…")
        self.segmenter = SpeechSegmenter(cfg.audio, cfg.vad)
        asr_label = cfg.asr.model_size if cfg.asr.backend == "whisper" else cfg.asr.backend
        self._stage(f"Loading speech recognition ({asr_label})…")
        self.asr = create_asr(cfg.asr)
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
        # Held around every self.asr.transcribe() call and around swapping
        # self.asr out for a different backend, so _asr_loop can never see a
        # None/half-built ASR object mid-swap (see _switch_asr_backend).
        self._asr_lock = threading.Lock()
        self._asr_busy = False  # a switch is already in flight; ignore a repeat click

        self._segments: queue.Queue = queue.Queue()
        self._transcripts: queue.Queue = queue.Queue()
        # Text already committed (pushed to translation) for the segment
        # currently being spoken, via periodic partial re-transcription - see
        # _segment_loop/_asr_loop. Reset to "" whenever a segment closes.
        self._committed_text = ""
        self._frames_since_partial = 0
        # Most recently transcribed segment's audio duration (seconds) - used
        # as translate's own real-time-factor denominator too (see
        # _translate_segment), since translated text has no duration of its
        # own to measure against.
        self._last_asr_audio_s = 0.0
        # Rolling per-stage (processing_s, audio_s) readings for the header's
        # RTF display, each stage's own deque touched only by that stage's
        # own thread - see _broadcast_rtf.
        self._rtf_window: dict[str, deque] = {}
        self._translations: queue.Queue = queue.Queue()
        self._audio: queue.Queue = queue.Queue()  # synthesized clips waiting to play
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()

        if cfg.tts.startup_voice:
            self._stage("Preparing the default voice…")
        self._use_startup_voice()
        # An explicit --default-voice/--clone already cleared startup_voice
        # (see cli.py) - that same signal means "don't override it with
        # whatever was last saved" here too.
        if self._persistent_config is not None and cfg.tts.startup_voice:
            self._restore_persisted_voice()

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
            # Drop anything already synthesized and queued but not yet handed
            # to the sound card (already covered by _playback_loop's own
            # voice_enabled check too, but no reason to let it sit there).
            # The clip playing *right now* is cut short by _playback_loop's
            # own should_continue polling instead of aborting the output
            # stream from this thread - a different thread calling
            # stream.abort()/close() while _playback_loop's thread is
            # mid-write() has been observed to segfault PortAudio's ALSA
            # backend.
            self._drain(self._audio)
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
            if self._persistent_config is not None:
                self._persistent_config.set(f"voice:{self.cfg.tts.backend}", f"default:{voice}")
        elif action in ("refresh_devices", "set_audio_devices", "mic_start", "mic_stop"):
            threading.Thread(target=self._handle_device_action, args=(action, data), daemon=True).start()
            return
        elif action == "set_asr_backend":
            backend = str(data.get("backend") or "")
            if backend not in ("whisper", "parakeet") or self._asr_busy:
                return
            self._asr_busy = True
            threading.Thread(target=self._switch_asr_backend, args=(backend,), daemon=True).start()
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
            speaker_id = data.get("speaker_id")
            self._select_speaker(speaker_id)
            if speaker_id and self._persistent_config is not None:
                self._persistent_config.set(f"voice:{self.cfg.tts.backend}", f"saved:{speaker_id}")
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
        self._committed_text = ""
        self._frames_since_partial = 0
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

    def _restore_persisted_voice(self) -> None:
        """Applies the operator's last-picked voice for this session's TTS
        backend (saved by handle_control's default_voice/select_speaker
        branches below), layered on top of _use_startup_voice() - called
        right before this, at __init__ only - rather than replacing it, so
        Reset still always goes back to the bundled Default voice as before;
        only the very first state of a fresh process differs."""
        saved = self._persistent_config.get(f"voice:{self.cfg.tts.backend}")
        if not saved:
            return
        if saved.startswith("default:"):
            name = self.cfg.tts.resolve_voice(saved[len("default:") :])
            if name in self.cfg.tts.all_voices():
                self.cfg.tts.default_voice = name
                self.tts.set_default_voice(name)
                self._select_speaker(None)
        elif saved.startswith("saved:"):
            speaker_id = saved[len("saved:") :]
            if self.voice_library.get(speaker_id):
                self._select_speaker(speaker_id)

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
        output_changed = new_output != audio.output_device
        with self._device_lock:
            try:
                from sts.audio.devices import check_input, check_output

                if input_changed:
                    check_input(new_input, audio.sample_rate, audio.channels)
                if output_changed:
                    check_output(new_output, 24_000)
                if input_changed:
                    self.capture.switch_to_alsa(new_input)
                if output_changed:
                    audio.output_device = new_output
                    self.playback.stop()  # the output stream reopens on the new device with the next clip
            except Exception as exc:
                print(f"[audio] couldn't apply the selected devices: {exc}")
                self._broadcast_devices(error=f"Couldn't use that device: {exc}")
                return
            vlog("audio", f"devices applied: input={audio.input_device}, output={audio.output_device}")
            if self._persistent_config is not None:
                # Only a real ALSA device is a restorable choice - not the
                # browser-mic mode, which is inherently session-only.
                if input_changed:
                    self._persistent_config.set(
                        "input_device", "" if audio.input_device is None else str(audio.input_device)
                    )
                if output_changed:
                    self._persistent_config.set(
                        "output_device", "" if audio.output_device is None else str(audio.output_device)
                    )
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

    def _load_asr(self, backend: str) -> None:
        """(Re)builds self.asr for `backend`, freeing whatever GPU memory the
        current one holds first - dropping the reference and collecting
        before building the replacement, rather than briefly holding both,
        since two ASR engines resident at once doesn't fit an 8GB card
        alongside translation/TTS. Must be called with _asr_lock held."""
        if hasattr(self, "asr"):
            del self.asr
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.cfg.asr.backend = backend
        self.asr = create_asr(self.cfg.asr)

    def _switch_asr_backend(self, backend: str) -> None:
        """Swaps the ASR engine live (e.g. Whisper <-> Parakeet from the
        Caption dropdown). Runs off the web server's event loop since a cold
        load can take tens of seconds. _asr_loop is blocked on _asr_lock for
        the whole swap, so it can never see self.asr as None or half-built -
        a transcription simply waits for the swap to finish, same as it
        already waits for a slow one from the current engine."""
        previous = self.cfg.asr.backend
        if backend == previous:
            self._asr_busy = False
            return
        try:
            with self._asr_lock:
                self._stage(f"Switching speech recognition to {backend}…")
                self._load_asr(backend)
            vlog("asr", f"switched backend: {previous} -> {backend}")
            if self._persistent_config is not None:
                self._persistent_config.set("asr_backend", backend)
        except (Exception, SystemExit) as exc:
            # create_asr() raises SystemExit (not Exception) for a missing
            # optional backend, appropriate for a hard CLI-startup failure
            # but not for a live webui toggle - caught here too so a missing
            # dependency falls back cleanly instead of leaving self.asr gone.
            print(f"[asr] couldn't switch to {backend!r}, reloading {previous!r}: {exc!r}")
            try:
                with self._asr_lock:
                    self._load_asr(previous)
            except (Exception, SystemExit) as exc2:
                # The fallback failed too (e.g. the failed switch left GPU
                # memory fragmented/exhausted) - leaving self.asr undefined
                # would crash _asr_loop's next transcribe() call outright and
                # kill that thread. A no-op stand-in keeps every thread alive
                # so the operator can retry (or restart the service) once
                # whatever's holding GPU memory clears, instead of losing
                # captions silently until someone notices and restarts it.
                print(f"[asr] fallback reload of {previous!r} also failed: {exc2!r}")
                with self._asr_lock:
                    self.cfg.asr.backend = previous
                    self.asr = _NullASR()
        finally:
            self._asr_busy = False
            self._stage("Listening", ready=True)
            self._broadcast_engines()
            self._broadcast_status()

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

    def _broadcast_rtf(self, stage: str, processing_s: float, audio_s: float) -> None:
        """Real-time factor (processing time / audio duration) for one
        pipeline stage - below 1.0 is faster than the speaker, above 1.0
        means that stage is falling behind live speech. `audio_s` <= 0 means
        there's nothing yet to measure against (e.g. translate before the
        first segment is transcribed), so it's skipped rather than dividing
        by zero.

        What's broadcast is smoothed over the last RTF_WINDOW_S: total
        processing time over total audio time across every reading still in
        the window, not just the one just taken - a single short phrase's
        processing time can otherwise spike the displayed value wildly even
        though the stage is nowhere near actually falling behind."""
        if audio_s <= 0:
            return
        now = time.monotonic()
        window = self._rtf_window.setdefault(stage, deque())
        window.append((now, processing_s, audio_s))
        cutoff = now - RTF_WINDOW_S
        while window and window[0][0] < cutoff:
            window.popleft()
        if not self.broadcaster:
            return
        windowed_processing = sum(p for _, p, _ in window)
        windowed_audio = sum(a for _, _, a in window)
        self.broadcaster.broadcast_rtf(stage, windowed_processing / windowed_audio)

    def _broadcast_library(self) -> None:
        if self.broadcaster:
            speakers = [
                {"id": e.speaker_id, "name": e.name, "builtin": e.builtin} for e in self.voice_library.list()
            ]
            self.broadcaster.broadcast_library(speakers)

    def _broadcast_engines(self) -> None:
        """Model/device info for the header dropdowns. Translation and TTS
        are fixed for the life of the process (set from CLI flags at
        startup); ASR isn't - its backend can change live via the Caption
        dropdown (see _switch_asr_backend) - so this is re-sent after every
        switch, not just once."""
        if self.broadcaster:
            asr_model = self.cfg.asr.model_size if self.cfg.asr.backend == "whisper" else parakeet_asr.MODEL_NAME
            tts_model = (
                self.cfg.tts.qwen3_model_name if self.cfg.tts.backend == "qwen3" else self.cfg.tts.model_name
            )
            self.broadcaster.broadcast_engines(
                {
                    "asr": {"model": asr_model, "device": self.cfg.asr.device, "backend": self.cfg.asr.backend},
                    "translation": {"model": self.cfg.translation.model_name, "device": self.cfg.translation.device},
                    "tts": {"model": tts_model, "device": self.cfg.tts.device},
                }
            )

    # -- pipeline stages -----------------------------------------------

    def _segment_loop(self) -> None:
        self._frames_since_partial = 0
        while not self._stop.is_set():
            self._process_captured_frame(self.capture.frames.get())

    def _process_captured_frame(self, frame: np.ndarray) -> None:
        """One frame's worth of VAD/segmentation bookkeeping: closes and
        queues a finished segment, or - if the segment is still open -
        periodically queues what's been spoken so far. Split out from
        _segment_loop so it's callable (and testable) frame by frame without
        the blocking queue/infinite-loop scaffolding around it."""
        segment = self.segmenter.process(frame)
        if segment is not None:
            self._frames_since_partial = 0
            if self.control.training.is_set():
                reached_target = self.control.accumulate_training_segment(segment)
                self._broadcast_status()
                if reached_target:
                    self._finish_training()
                    self._broadcast_status()
            if self.control.active.is_set():
                self._segments.put(("final", segment))
            return

        # Segment hasn't closed yet - periodically hand the ASR stage what's
        # been spoken so far, so a segment with several sentences (VAD only
        # cuts on a long enough pause) doesn't make the first of them wait
        # for the last. Only when caught up (queue empty): a slow backend
        # then just falls back to today's behavior instead of building a backlog.
        self._frames_since_partial += 1
        interval = self.cfg.vad.partial_interval_ms
        if not (self.control.active.is_set() and interval > 0):
            return
        if self._frames_since_partial * self.cfg.audio.block_ms < interval:
            return
        if self._segments.qsize() > 0:
            return
        partial_audio = self.segmenter.peek()
        if partial_audio is None:
            return
        self._frames_since_partial = 0
        self._segments.put(("partial", partial_audio))

    def _asr_loop(self) -> None:
        while not self._stop.is_set():
            kind, segment = self._segments.get()
            t0 = time.monotonic()
            with self._asr_lock:
                text = self.asr.transcribe(segment)
            processing_s = time.monotonic() - t0
            audio_s = len(segment) / self.cfg.audio.sample_rate
            vlog("asr", f"transcribed {audio_s:.1f}s of audio ({kind}) in {processing_s:.2f}s")
            self._last_asr_audio_s = audio_s
            self._broadcast_rtf("asr", processing_s, audio_s)
            self._commit_new_asr_text(text, final=kind == "final")

    def _commit_new_asr_text(self, full_text: str, final: bool) -> None:
        """Pushes whatever part of `full_text` - the growing (partial) or
        whole (final) segment's transcript - hasn't already been committed
        for this segment. A partial re-run only releases complete sentences,
        and only when there's real text *after* the sentence-ending
        punctuation too: given only a short audio prefix, the model will
        often end its output with a plausible-looking period simply because
        it ran out of audio, not because that's a real sentence boundary -
        seeing it keep going past that point is what distinguishes a genuine
        boundary from that. A trailing fragment (real or not yet confirmed)
        is left for the next re-run to catch.

        Each re-run re-transcribes the growing audio from scratch (no
        cache/streaming decoder), so in rare cases more context changes how an
        earlier part was heard - this isn't purely a short-prefix problem
        (see below), so it can't be assumed away. Mid-segment (not `final`)
        this only trusts what's still a clean extension of what's already
        committed and waits for the next re-run rather than guessing.

        The final pass is different: measured against real audio, the model
        occasionally reworded even the very start of a long segment once the
        *entire* segment's audio was available as context - not just near
        chunk edges - so requiring the final transcript to still start with
        the committed prefix would, in that case, silently drop every
        sentence after the divergence instead of just duplicating a couple of
        already-shown words. Content loss is the worse failure of the two, so
        the final pass always pushes its complete, authoritative text (never
        held back), accepting that a diverged one occasionally repeats a
        little of what a partial already showed."""
        if final:
            to_commit = full_text.strip()
            if full_text.startswith(self._committed_text):
                to_commit = full_text[len(self._committed_text):].strip()
            elif self._committed_text:
                vlog("asr", "final transcript reworded already-committed text; "
                            "sending it in full rather than risk dropping what follows")
            self._committed_text = ""  # this segment is done; ready for the next one
        elif full_text.startswith(self._committed_text):
            new_text = full_text[len(self._committed_text):].strip()
            to_commit, tail = split_complete(new_text)
            if to_commit and not tail:
                # Nothing follows the period *yet* - can't tell a real sentence
                # boundary from the model just running out of audio to decode.
                return
            if to_commit:
                self._committed_text = f"{self._committed_text} {to_commit}".strip()
        else:
            vlog("asr", "partial re-run revised already-committed text; keeping only what was already sent")
            return

        if not to_commit:
            return
        print(f"[ru] {to_commit}")
        if self.broadcaster:
            # One message per sentence, not one for the whole commit: the
            # caption page's portrait layout pairs each "original" message
            # with a "translated" one by arrival order, and translation
            # already broadcasts one message per sentence (see
            # _translate_segment) - splitting the same way here, with the
            # same split_sentences(), keeps the two streams in lockstep.
            for sentence in split_sentences(to_commit):
                self.broadcaster.broadcast_original(sentence)
            self._broadcast_bible_references(to_commit)
        self._transcripts.put(to_commit)

    def _broadcast_bible_references(self, text: str) -> None:
        for ref in find_references(text):
            label = f"{ref.ru_label} {ref.chapter}:{ref.verse}" + (f"-{ref.verse_end}" if ref.verse_end else "")
            print(f"[bible] detected reference: {label}")
            quote = self.bible_lookup.lookup(ref.book, ref.ru_label, ref.chapter, ref.verse, ref.verse_end)
            if not (quote.ru_text or quote.en_text):
                # broadcast_verse() silently no-ops on this (nothing to show) -
                # logged here so a detected-but-unfound reference isn't mistaken
                # for one that was never caught in the first place.
                print(f"[bible] no verse text found in the bundled Bible data for {label}")
            self.broadcaster.broadcast_verse(
                quote.ru_label, quote.chapter, quote.verse, quote.verse_end, quote.ru_text, quote.en_text
            )

    def _translate_loop(self) -> None:
        while not self._stop.is_set():
            text = self._transcripts.get()
            if not self.control.translation_enabled.is_set():
                vlog("translate", "translation is off, skipping")
                continue
            self._translate_segment(text)

    def _translate_segment(self, text: str) -> None:
        """Translates one ASR segment's text and pushes/broadcasts/queues each
        sentence for TTS the moment it's ready. Yielded, not joined-then-
        returned: a multi-sentence segment (VAD closes on silence, so it can
        hold several full sentences) would otherwise have its first sentence
        wait on the whole block's slowest (usually last) sentence to translate."""
        t0 = time.monotonic()
        for translated in self.translator.translate_sentences(text):
            processing_s = time.monotonic() - t0
            vlog("translate", f"translated a sentence in {processing_s:.2f}s")
            if translated:
                print(f"[en] {translated}")
                if self.broadcaster:
                    self.broadcaster.broadcast_translated(translated)
                self._broadcast_rtf("translate", processing_s, self._last_asr_audio_s)
                self._translations.put(translated)
            t0 = time.monotonic()

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
                    processing_s = time.monotonic() - t0
                    audio_s = audio.size / sample_rate
                    vlog("tts", f"synthesized {audio_s:.1f}s of speech in {processing_s:.2f}s "
                                f"({self._translations.qsize()} sentences waiting)")
                    self._broadcast_rtf("voice", processing_s, audio_s)
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
            # Broadcast before playing locally, not after: play() blocks for
            # close to the clip's real duration (it paces with the sound
            # card draining its buffer), so a connected browser would
            # otherwise only receive the clip once local playback had
            # already finished - the 3-5s "streaming delay" this was
            # written to fix. broadcast_audio itself is fire-and-forget
            # (schedules onto the server's event loop and returns
            # immediately), so this doesn't delay local playback at all.
            if self.broadcaster:
                self.broadcaster.broadcast_audio(audio, sample_rate)
            try:
                self.playback.play(
                    audio, sample_rate=sample_rate, pause_before=pause_before,
                    should_continue=self.control.voice_enabled.is_set,
                )
            except Exception as exc:
                print(f"[audio] playback failed: {exc!r}")

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
