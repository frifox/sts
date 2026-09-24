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
import traceback
import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from sts.asr import create_asr, estimate_vram_bytes, parakeet_asr
from sts.audio.capture import AudioCapture
from sts.audio.devices import check_input, check_output, reinitialize_audio_system
from sts.audio.playback import AudioPlayback
from sts.audio.tempo import time_stretch
from sts.audio.vad import SpeechSegmenter
from sts.bible.lookup import BibleLookup
from sts.bible.reference import find_references
from sts.config import PipelineConfig
from sts.control import RecognitionControl
from sts.log import vlog
from sts.persistent_config import PersistentConfig
from sts.textclean import new_text_since, split_complete
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

# How long a local input stream may go without delivering a frame before it's
# treated as dead (frames arrive every ~32ms even in a silent room), and the
# least time between two audio-system recoveries - a device that's genuinely
# gone would otherwise trigger one every check.
INPUT_STALL_S = 3.0
AUDIO_RECOVERY_MIN_INTERVAL_S = 5.0

# Pre-rendered "Testing 1 2 3" clip for the Output dropdown's Test Output
# button - a fixed recording rather than synthesized fresh each click, so it
# plays even when Voice (and its models) are off.
TEST_OUTPUT_WAV = Path(__file__).resolve().parent / "assets" / "test_output.wav"

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


# How long a stage's blocking queue get() waits before re-checking whether
# the pipeline is stopping - so stop() can actually join the stage threads.
QUEUE_POLL_S = 0.5

# Pause after a stage step raises before taking the next item, so an error
# that repeats on every item (a broken model, a device gone) logs at a
# readable rate instead of spinning a CPU core.
STAGE_ERROR_BACKOFF_S = 0.5


def _is_cuda_oom(exc: BaseException) -> bool:
    """torch.OutOfMemoryError from PyTorch models, or CTranslate2's
    RuntimeError("CUDA failed with error out of memory") from Whisper."""
    return isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()


def _free_gpu_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _strip_repeated_prefix(translated: str, previous: str | None) -> str:
    """`translated` was produced with `previous` as leading context (see
    Translator.translate_sentences and _commit_new_asr_text) - given as
    grammatical lead-in, not meant to be shown again. NLLB sometimes restates
    it verbatim before continuing instead of picking up mid-thought (observed
    against real transcripts: the whole previous sentence repeated, then the
    new content tacked on) - a duplicate that reads as a broken repeat next
    to the caption that already showed it. Detected (case-insensitively,
    ignoring `previous`'s own trailing punctuation) and removed, so the
    caption reads as a continuation of it instead. Falls back to `translated`
    unchanged if there's nothing recognizable to strip, or if stripping would
    leave nothing at all (better a duplicate than empty content)."""
    if not previous:
        return translated
    lead = previous.rstrip(".!?… ")
    if not lead or not translated.lower().startswith(lead.lower()):
        return translated
    remainder = translated[len(lead):].lstrip(" ,.;:-–—")
    return remainder or translated


@dataclass
class _Clip:
    """One piece of synthesized speech waiting in the playback queue."""

    audio: np.ndarray
    sample_rate: int
    pause_before: float = 0.0
    # Plays even with Voice off (the Test Output button's clip).
    force: bool = False
    # A streamed chunk continuing the previous clip mid-piece: no pause
    # before it, and it may be merged with that clip before speeding up.
    seamless: bool = False

    @property
    def seconds(self) -> float:
        return self.pause_before + self.audio.size / self.sample_rate


class _NullASR:
    """Last-resort stand-in when a live ASR backend switch fails and the
    fallback reload of the previous engine fails too (see
    Pipeline._switch_asr_backend) - keeps _asr_loop's thread alive instead of
    crashing on a missing self.asr, at the cost of transcribing nothing until
    the operator retries or restarts the service."""

    def transcribe(self, audio) -> str:
        return ""


class Pipeline:
    # Bumped by _reset(). ASR, translation, and TTS each note it before a
    # slow call and drop their result if it changed meanwhile - so text or
    # speech from the previous speaker that was already mid-flight when the
    # operator hit Reset doesn't show up after the queues were drained.
    # (A class attribute so test stubs built with object.__new__ get it too.)
    _reset_generation = 0
    # Slow operator actions (anything that touches a voice or the models) run
    # here, one at a time and in order, on the control thread - see
    # handle_control. None until start(), in which case they run inline
    # (the CLI's --clone at startup, tests).
    _commands: queue.Queue | None = None

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
        self._audio_recovery_lock = threading.Lock()
        self._last_audio_recovery = float("-inf")
        self._last_audio_recovery_ok = True
        self._active_speaker_id: str | None = None
        # Held around every self.asr.transcribe() call and around swapping
        # self.asr out for a different backend, so _asr_loop can never see a
        # None/half-built ASR object mid-swap (see _switch_asr_backend).
        self._asr_lock = threading.Lock()
        self._asr_busy = False  # a switch is already in flight; ignore a repeat click
        self._last_output_level_broadcast = 0.0

        self._segments: queue.Queue = queue.Queue()
        # (text, first_seq, context) per committed chunk - see _commit_new_asr_text.
        self._transcripts: queue.Queue = queue.Queue()
        self._original_seq = 0  # next sequence number for a broadcast original sentence
        # The most recently committed sentence's original-language text and
        # its own translation (any segment, not just the current one) - see
        # _commit_new_asr_text's use of the former as translation context for
        # a sentence that continues it, and _strip_repeated_prefix's use of
        # the latter to catch NLLB restating that context verbatim.
        self._last_sentence_original: str | None = None
        self._last_sentence_translated: str | None = None
        # Text already committed (pushed to translation) for the segment
        # currently being spoken, via periodic partial re-transcription - see
        # _segment_loop/_asr_loop. Reset to "" whenever a segment closes.
        self._committed_text = ""
        # A sentence a partial re-run just completed, held back until the
        # *next* partial re-run reproduces it identically - see
        # _commit_new_asr_text. None when there's nothing awaiting
        # confirmation. Reset to None whenever a segment closes/resets,
        # alongside _committed_text.
        self._pending_confirmation: str | None = None
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
        # Bumped by _reset() to cut whatever _playback_loop is mid-write on
        # right now, without touching the output stream from this (a
        # different) thread - see _play().
        self._playback_generation = 0
        # name -> (target, thread) for every long-running stage thread, so the
        # watchdog can notice one that died and start it again.
        self._threads: dict[str, tuple] = {}
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

    # Operator actions cheap enough to do right away on the web server's
    # event loop: flipping a switch, dropping queued items, or handing off
    # to a thread of their own. Everything else queues for the control
    # thread - done inline, a voice-training transcription or a clone
    # prompt that waits for a sentence mid-synthesis froze every connected
    # page (captions included) for seconds, and let a Voice off or Reset
    # pressed meanwhile wait behind it.
    _INSTANT_ACTIONS = frozenset({
        "start", "stop", "voice_on", "voice_off", "translate_on", "translate_off", "reset",
        "test_output", "train_start", "train_stop", "set_min_silence", "set_output_volume",
        "refresh_devices", "set_audio_devices", "mic_start", "mic_stop", "set_asr_backend",
        "design_generate", "design_play",
    })

    def handle_control(self, data: dict) -> None:
        """Called on the web server's event loop for each operator action."""
        if data.get("action") in self._INSTANT_ACTIONS:
            self._execute_control(data)
        else:
            self._on_control_thread(self._execute_control, data)

    def _on_control_thread(self, fn, *args) -> None:
        """Runs fn(*args) on the control thread, after anything already
        queued there, and broadcasts the resulting status."""
        if self._commands is None:
            fn(*args)
        else:
            self._commands.put((fn, args))

    def _control_step(self) -> None:
        fn, args = self._commands.get(timeout=QUEUE_POLL_S)
        fn(*args)
        self._broadcast_status()

    def _execute_control(self, data: dict) -> None:
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
            # No more on_chunk calls will happen until something plays again,
            # so the output meter would otherwise be stuck showing whatever
            # level happened to be last, forever - explicitly zero it.
            if self.broadcaster:
                self.broadcaster.broadcast_output_level(0.0)
        elif action == "translate_on":
            self.control.enable_translation()
        elif action == "translate_off":
            self.control.disable_translation()
        elif action == "reset":
            self._reset()
        elif action == "test_output":
            self._queue_test_output()
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
        elif action == "set_output_volume":
            try:
                percent = int(data.get("percent"))
            except (TypeError, ValueError):
                return
            percent = max(0, min(100, percent))
            # Read by AudioPlayback.play() at the start of each clip, so this
            # takes effect on whatever plays next (a clip already mid-playback
            # keeps the gain it started with).
            self.cfg.audio.volume = percent / 100
            vlog("audio", f"output volume set to {percent}%")
            if self._persistent_config is not None:
                self._persistent_config.set("output_volume", str(percent))
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
        text from the previous one, forget any cloned voice, and stop
        whatever's currently playing."""
        if self.control.training.is_set():
            self.control.stop_training(discard=True)
        # After anything already queued - a voice being trained or picked
        # just before Reset must not become active again after it.
        self._on_control_thread(self._reset_voice)
        self._committed_text = ""
        self._pending_confirmation = None
        self._last_sentence_original = None
        self._last_sentence_translated = None
        self._frames_since_partial = 0
        self._reset_generation += 1
        self._drain(self._segments)
        self._drain(self._transcripts)
        self._drain(self._translations)
        self._drain(self._audio)
        # Cuts a clip _playback_loop is mid-write on right now, the same safe
        # way voice_off does: bumping this makes _play()'s should_continue
        # return False on its next ~50ms poll, so _playback_loop's own thread
        # stops writing on its own - calling playback.stop() (which
        # abort()s/close()s the stream) from this thread instead, while that
        # one is mid-write(), has been observed to segfault PortAudio's ALSA
        # backend.
        self._playback_generation += 1
        if self.broadcaster:
            self.broadcaster.broadcast_output_level(0.0)
        if self.broadcaster:
            self.broadcaster.broadcast_reset()

    def _reset_voice(self) -> None:
        self._use_startup_voice()
        self._pending_voice = None

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
        """Ends the training capture right away (so no more audio goes into
        it), and transcribes/clones what was captured on the control thread."""
        audio = self.control.stop_training(discard=discard)
        if audio is not None:
            self._on_control_thread(self._activate_trained_voice, audio)

    def _activate_trained_voice(self, audio: np.ndarray) -> None:
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
            # The lock: an ASR engine switch may be swapping self.asr out
            # right now, and _asr_loop may be mid-transcribe on it.
            with self._asr_lock:
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

    def _load_asr(self, backend: str, check_memory: bool = True) -> None:
        """(Re)builds self.asr for `backend`, freeing whatever GPU memory the
        current one holds first - releasing it and collecting before building
        the replacement, rather than briefly holding both, since two ASR
        engines resident at once doesn't fit an 8GB card alongside
        translation/TTS. Refuses up front (raising, so the caller falls back)
        when what's free once the old engine is gone clearly can't hold the
        new one, rather than finding out halfway through the load with the
        card left full of a half-built model (`check_memory`; skipped for a
        fallback to the engine that was running a moment ago, which is
        simply worth trying). Must be called with _asr_lock held."""
        self._unload_asr()
        self.cfg.asr.backend = backend
        device = self.cfg.asr.device
        if check_memory and device.startswith("cuda") and torch.cuda.is_available():
            free, _total = torch.cuda.mem_get_info(torch.device(device))
            needed = estimate_vram_bytes(self.cfg.asr)
            if free < needed:
                raise RuntimeError(
                    f"not enough GPU memory for {backend}: about {needed / 2**30:.1f} GB needed, "
                    f"{free / 2**30:.1f} GB free"
                )
        self.asr = create_asr(self.cfg.asr)

    def _unload_asr(self) -> None:
        asr = getattr(self, "asr", None)
        if asr is None:
            return
        del self.asr
        close = getattr(asr, "close", None)
        if close is not None:
            try:
                close()
            except Exception as exc:
                print(f"[asr] couldn't release the old engine cleanly: {exc!r}")
        del asr
        _free_gpu_memory()

    def _switch_asr_backend(self, backend: str) -> None:
        """Swaps the ASR engine live (e.g. Whisper <-> Parakeet from the
        Caption dropdown). Runs off the web server's event loop since a cold
        load can take tens of seconds. _asr_loop is blocked on _asr_lock for
        the whole swap, so it can never see self.asr as None or half-built -
        a transcription simply waits for the swap to finish, same as it
        already waits for a slow one from the current engine.

        That has to include the *failure* path too: _load_asr deletes
        self.asr before building the replacement, so if the lock were
        released between a failed attempt and the fallback reload,
        _asr_loop could grab it in that gap, find self.asr gone, and die
        with an AttributeError - killing captions until a restart. So the
        attempt and its fallback share one uninterrupted hold of the lock.

        The fallback runs only after the `except` block has ended: while
        it's still open, the exception's traceback keeps the failed load's
        frames - and the partly built model in them - alive, pinning its
        GPU memory. Reloading the previous engine from inside it ran out of
        memory for exactly that reason in real use."""
        previous = self.cfg.asr.backend
        if backend == previous:
            self._asr_busy = False
            return
        switched = False
        try:
            with self._asr_lock:
                self._stage(f"Switching speech recognition to {backend}…")
                failure = None
                try:
                    self._load_asr(backend)
                    switched = True
                except (Exception, SystemExit) as exc:
                    # create_asr() raises SystemExit (not Exception) for a
                    # missing optional backend, appropriate for a hard
                    # CLI-startup failure but not for a live webui toggle -
                    # caught here too so a missing dependency falls back
                    # cleanly instead of leaving self.asr gone.
                    failure = repr(exc)
                if not switched:
                    print(f"[asr] couldn't switch to {backend!r}, reloading {previous!r}: {failure}")
                    _free_gpu_memory()
                    fallback_failure = None
                    try:
                        self._load_asr(previous, check_memory=False)
                    except (Exception, SystemExit) as exc2:
                        fallback_failure = repr(exc2)
                    if fallback_failure is not None:
                        # The fallback failed too - leaving self.asr undefined
                        # would crash _asr_loop's next transcribe() call. A
                        # no-op stand-in keeps every thread alive so the
                        # operator can retry (or restart the service) once
                        # whatever's holding GPU memory clears, instead of
                        # losing captions silently until someone notices.
                        print(f"[asr] fallback reload of {previous!r} also failed: {fallback_failure}")
                        self.cfg.asr.backend = previous
                        self.asr = _NullASR()
            if switched:
                vlog("asr", f"switched backend: {previous} -> {backend}")
                if self._persistent_config is not None:
                    self._persistent_config.set("asr_backend", backend)
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
                output_volume=round(self.cfg.audio.volume * 100),
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

    def _broadcast_input_level(self, frame: np.ndarray) -> None:
        """RMS of one raw capture frame, for the volume-bar meter next to the
        Caption button. Sent regardless of VAD state or the Caption toggle -
        it reflects whatever is arriving from the selected input (ALSA or a
        browser mic), not whether the pipeline is currently doing anything
        with it, throttled by the caller to about every 100ms."""
        if self.broadcaster and frame.size:
            rms = float(np.sqrt(np.mean(np.square(frame))))
            self.broadcaster.broadcast_input_level(rms)

    def _broadcast_output_level(self, chunk: np.ndarray) -> None:
        """RMS of one output chunk as it's written, for the volume-bar meter
        next to the Voice button. Throttled internally (unlike
        _broadcast_input_level, whose caller owns a loop-local timestamp)
        since this runs as AudioPlayback.play()'s on_chunk callback, called
        from inside its own chunk loop rather than a loop this class owns."""
        if not self.broadcaster or not chunk.size:
            return
        now = time.monotonic()
        if now - self._last_output_level_broadcast < self._LEVEL_METER_INTERVAL_S:
            return
        self._last_output_level_broadcast = now
        rms = float(np.sqrt(np.mean(np.square(chunk))))
        self.broadcaster.broadcast_output_level(rms)

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

    # Shared by both the input and output volume-bar meters. Input frames
    # arrive in fixed block_ms (32ms) chunks, so a plain ">= 0.1s" gate would
    # fire on the first boundary *past* 100ms (128ms: 4 frames), not the
    # closest one (96ms: 3 frames, only 4ms off) - this trims the threshold
    # just under 3 frames' worth so it catches that one instead.
    _LEVEL_METER_INTERVAL_S = 0.095

    def _run_stage(self, name: str, step) -> None:
        """Calls `step()` - one queue item's worth of a stage's work - until
        the pipeline stops. An exception from one item is logged (with its
        traceback) and that item is dropped, instead of ending the thread:
        an uncaught one used to kill the stage silently for the rest of the
        session (e.g. one CUDA out-of-memory error in translation meant no
        more translations until a restart). A GPU out-of-memory error also
        releases PyTorch's cached-but-unused memory, which is often enough
        for the next item to fit."""
        while not self._stop.is_set():
            try:
                step()
            except queue.Empty:
                pass  # nothing arrived within QUEUE_POLL_S; check _stop and wait again
            except Exception as exc:
                print(f"[{name}] error, skipping this item: {exc!r}")
                traceback.print_exc()
                if _is_cuda_oom(exc):
                    _free_gpu_memory()
                self._stop.wait(STAGE_ERROR_BACKOFF_S)

    def _segment_loop(self) -> None:
        self._frames_since_partial = 0
        self._last_level_broadcast = 0.0
        self._run_stage("vad", self._segment_step)

    def _segment_step(self) -> None:
        frame = self.capture.frames.get(timeout=QUEUE_POLL_S)
        now = time.monotonic()
        if now - self._last_level_broadcast >= self._LEVEL_METER_INTERVAL_S:
            self._broadcast_input_level(frame)
            self._last_level_broadcast = now
        self._process_captured_frame(frame)

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
        self._run_stage("asr", self._asr_step)

    def _asr_step(self) -> None:
        kind, segment = self._segments.get(timeout=QUEUE_POLL_S)
        generation = self._reset_generation
        t0 = time.monotonic()
        try:
            with self._asr_lock:
                text = self.asr.transcribe(segment)
        except Exception:
            if kind == "final":
                # The segment's already-committed sentences stay committed;
                # the next segment has to start from a clean slate, or it'd be
                # compared against this one's leftover _committed_text.
                self._committed_text = ""
                self._pending_confirmation = None
            raise
        processing_s = time.monotonic() - t0
        audio_s = len(segment) / self.cfg.audio.sample_rate
        vlog("asr", f"transcribed {audio_s:.1f}s of audio ({kind}) in {processing_s:.2f}s")
        self._last_asr_audio_s = audio_s
        self._broadcast_rtf("asr", processing_s, audio_s)
        if generation != self._reset_generation:
            vlog("asr", "dropping a transcript of audio from before Reset")
            return
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

        A newly-completed sentence isn't trusted on first sight either:
        measured against real audio, a partial re-run's first pass at a
        sentence is sometimes revised once a little more audio context is
        available on the *next* re-run (e.g. "Вопрос:" becoming "Возникает
        вопрос," - not just a reworded ending, a different opening), which
        would otherwise get shown, then shown again differently worded once
        the next re-run's version is committed on top of it. So a candidate
        sentence is held in `_pending_confirmation` and only actually
        committed once a later re-run reproduces it identically - trading a
        bit of latency (it needs to survive one more partial_interval_ms
        cycle before it first appears) for not showing a guess that's still
        likely to change.

        The final pass is different: measured against real audio, the model
        occasionally reworded even the very start of a long segment once the
        *entire* segment's audio was available as context - not just near
        chunk edges - so requiring the final transcript to still start with
        the committed prefix (character for character) would, in that case,
        silently drop every sentence after the divergence instead of just
        duplicating a couple of already-shown words. Content loss is the
        worse failure of the two, so a divergence is never allowed to drop
        anything - but it's compared *word by word, ignoring punctuation*
        (see new_text_since()) rather than as one raw string, so a boundary
        the model re-punctuated differently the second time (a period
        becoming a comma, continuing the sentence instead of ending it)
        doesn't make the whole segment look diverged and get resent in full
        - only whatever's actually new is. The final pass is never held back
        awaiting confirmation either - the segment is closing now, so
        whatever it says is authoritative."""
        if final:
            if self._committed_text and not full_text.startswith(self._committed_text):
                vlog("asr", "final transcript's wording diverged from what's already committed; "
                            "resending only what isn't already covered")
            to_commit = new_text_since(self._committed_text, full_text)
            self._committed_text = ""  # this segment is done; ready for the next one
            self._pending_confirmation = None
        elif full_text.startswith(self._committed_text):
            new_text = full_text[len(self._committed_text):].strip()
            candidate, tail = split_complete(new_text)
            if not candidate or not tail:
                # No complete new sentence yet, or nothing follows the period
                # *yet* - can't tell a real sentence boundary from the model
                # just running out of audio to decode.
                return
            if candidate != self._pending_confirmation:
                # First time this exact wording has completed a sentence -
                # wait for the next re-run to reproduce it before trusting it.
                self._pending_confirmation = candidate
                return
            to_commit = candidate
            self._pending_confirmation = None
            self._committed_text = f"{self._committed_text} {to_commit}".strip()
        else:
            vlog("asr", "partial re-run revised already-committed text; keeping only what was already sent")
            self._pending_confirmation = None
            return

        if not to_commit:
            return
        print(f"[ru] {to_commit}")
        # One message per sentence, not one for the whole commit, each with a
        # sequence number: the caption page's portrait layout pairs every
        # "original" with its "translated" by that number, not by arrival
        # order (which breaks as soon as translation is off, or lags, or a
        # page reconnects mid-stream - some originals then never get a
        # translation, shifting every later pair). Translation splits the
        # same string with the same split_sentences() (see
        # _translate_segment), so sentence i of this commit is `first_seq + i`
        # there too. Numbers advance whether or not anything is listening.
        # The commit's first number doubles as its batch id (see
        # CaptionBroadcaster.broadcast_original).
        sentences = split_sentences(to_commit)
        # A lowercase first letter is the ASR model's own signal that this
        # sentence isn't a fresh one - it's the tail of a thought VAD's pause
        # threshold cut off mid-utterance, re-transcribed from scratch with no
        # memory of what preceded it (see the class docstring). Translating it
        # alone, as if it were self-contained, reliably reads as a disconnected
        # non-sequitur - so it's translated together with the last sentence
        # actually committed (any segment, not just this one) as surrounding
        # context, in _translate_segment.
        context = self._last_sentence_original if sentences and sentences[0][:1].islower() else None
        first_seq = self._original_seq
        self._original_seq += len(sentences)
        if self.broadcaster:
            for i, sentence in enumerate(sentences):
                self.broadcaster.broadcast_original(sentence, first_seq + i, first_seq)
            self._broadcast_bible_references(to_commit)
        self._transcripts.put((to_commit, first_seq, context))
        if sentences:
            self._last_sentence_original = sentences[-1]

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
        self._run_stage("translate", self._translate_step)

    def _translate_step(self) -> None:
        text, first_seq, context = self._transcripts.get(timeout=QUEUE_POLL_S)
        if not self.control.translation_enabled.is_set():
            vlog("translate", "translation is off, skipping")
            return
        self._translate_segment(text, first_seq, context)

    def _translate_segment(self, text: str, first_seq: int | None = None, context: str | None = None) -> None:
        """Translates one ASR segment's text and pushes/broadcasts/queues each
        sentence for TTS the moment it's ready. Yielded, not joined-then-
        returned: a multi-sentence segment (VAD closes on silence, so it can
        hold several full sentences) would otherwise have its first sentence
        wait on the whole block's slowest (usually last) sentence to translate.

        `first_seq` is the sequence number _commit_new_asr_text gave the
        first sentence of `text`; each translation is broadcast with the
        number of the original sentence it belongs to, and with `first_seq`
        itself as its batch id (the same one that commit's originals carry).

        `context`, if given, is the previous sentence committed - passed
        through to Translator.translate_sentences so it can be given as
        leading context for `text`'s own first sentence (see
        _commit_new_asr_text for when/why). NLLB sometimes restates that
        context verbatim before continuing rather than picking up mid-thought;
        _strip_repeated_prefix catches and removes it from the first
        sentence's translation, since the context sentence's own translation
        was already shown on its own line."""
        generation = self._reset_generation
        t0 = time.monotonic()
        for i, translated in enumerate(self.translator.translate_sentences(text, context=context)):
            if generation != self._reset_generation:
                vlog("translate", "dropping the rest of a segment from before Reset")
                return
            processing_s = time.monotonic() - t0
            vlog("translate", f"translated a sentence in {processing_s:.2f}s")
            continues = i == 0 and bool(context)
            if continues:
                translated = _strip_repeated_prefix(translated, self._last_sentence_translated)
            if translated:
                print(f"    [en] {translated}")
                if self.broadcaster:
                    self.broadcaster.broadcast_translated(
                        translated,
                        None if first_seq is None else first_seq + i,
                        first_seq,
                    )
                self._broadcast_rtf("translate", processing_s, self._last_asr_audio_s)
                # `continues` rides along so _tts_loop can give this sentence
                # the short within-a-segment pause instead of the long
                # between-segments one - it's spoken as the continuation it
                # actually is, not introduced as a fresh new utterance.
                self._translations.put((translated, continues))
                self._last_sentence_translated = translated
            t0 = time.monotonic()

    def _tts_loop(self) -> None:
        self._run_stage("tts", self._tts_step)

    def _tts_step(self) -> None:
        text, continues = self._translations.get(timeout=QUEUE_POLL_S)
        if not self.control.voice_enabled.is_set():
            vlog("tts", "voice is off, skipping synthesis")
            return
        generation = self._reset_generation
        t0 = time.monotonic()
        # Each piece is handed to the playback thread as soon as it is ready,
        # so the rest of the sentence (and the next one) synthesizes while
        # this plays. A failure raises out to _run_stage, which skips just
        # this sentence.
        first_piece = True
        for audio, sample_rate, seamless in self._synthesize_pieces(text):
            if generation != self._reset_generation or not self.control.voice_enabled.is_set():
                vlog("tts", "Reset or Voice off mid-sentence; not synthesizing the rest")
                return
            if not audio.size:
                continue
            processing_s = time.monotonic() - t0
            audio_s = audio.size / sample_rate
            vlog("tts", f"synthesized {audio_s:.1f}s of speech in {processing_s:.2f}s "
                        f"({self._translations.qsize()} sentences waiting)")
            self._broadcast_rtf("voice", processing_s, audio_s)
            # A longer pause before a new translated text, a shorter one
            # between the sentences within it - and also for the first
            # piece of a sentence ASR itself flagged as continuing the
            # previous one (see _translate_segment): VAD split it into
            # a separate segment, but it's still one spoken thought, so
            # it's paced like a continuation, not introduced fresh. A
            # streamed chunk that continues the same piece mid-word gets
            # no pause at all.
            if seamless:
                pause = 0.0
            elif first_piece:
                pause = self.cfg.audio.pause_within_s if continues else self.cfg.audio.pause_between_s
            else:
                pause = self.cfg.audio.pause_within_s
            self._audio.put(_Clip(audio, sample_rate, pause, seamless=seamless))
            first_piece = False
            t0 = time.monotonic()

    def _synthesize_pieces(self, text: str):
        """(audio, sample_rate, seamless) per piece. Backends that stream
        (XTTS's stream_chunks) mark each chunk after the first of a piece
        seamless; the others yield whole pieces."""
        chunks = getattr(self.tts, "stream_chunks", None)
        if chunks is not None:
            yield from chunks(text)
            return
        stream = getattr(self.tts, "synthesize_stream", None)
        pieces = stream(text) if stream is not None else [self.tts.synthesize(text)]
        for audio, sample_rate in pieces:
            yield audio, sample_rate, False

    def _queue_test_output(self) -> None:
        """Loads the pre-rendered "Testing 1 2 3" clip and queues it on the
        normal playback path, so it goes out through whatever output device
        is currently selected - forced to play even if Voice is off, since
        this is meant to test the hardware, not the translation pipeline."""
        try:
            with wave.open(str(TEST_OUTPUT_WAV), "rb") as wav_file:
                sample_rate = wav_file.getframerate()
                raw = wav_file.readframes(wav_file.getnframes())
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        except Exception as exc:
            print(f"[audio] couldn't load the test output clip: {exc!r}")
            return
        self._audio.put(_Clip(audio, sample_rate, force=True))

    def _playback_loop(self) -> None:
        self._run_stage("playback", self._playback_step)

    def _playback_step(self) -> None:
        if self.playback.holding_tail:
            # The last piece may continue in another streamed chunk: wait for
            # it only until just before the sound card runs out of what's
            # already written, then play out the resampler's held-back tail.
            try:
                clip = self._audio.get(timeout=max(0.01, self.playback.pending_s() - 0.03))
            except queue.Empty:
                self.playback.flush()
                return
        else:
            clip = self._audio.get(timeout=QUEUE_POLL_S)
        if not clip.force and not self.control.voice_enabled.is_set():
            return  # voice was switched off while this was queued
        clip = self._catch_up(clip)
        # Broadcast before playing locally, not after: play() blocks for
        # close to the clip's real duration (it paces with the sound
        # card draining its buffer), so a connected browser would
        # otherwise only receive the clip once local playback had
        # already finished - the 3-5s "streaming delay" this was
        # written to fix. broadcast_audio itself is fire-and-forget
        # (schedules onto the server's event loop and returns
        # immediately), so this doesn't delay local playback at all.
        if self.broadcaster:
            self.broadcaster.broadcast_audio(clip.audio, clip.sample_rate)
        try:
            self._play(clip.audio, clip.sample_rate, clip.pause_before, clip.force, clip.seamless)
        except Exception as exc:
            print(f"[audio] playback failed: {exc!r}")
            # Typically the output device went stale (laptop slept, the
            # default output changed) - re-initialize audio and give this
            # same clip one more try instead of dropping it, and then
            # every clip after it.
            if self._recover_audio("playback failed"):
                try:
                    self._play(clip.audio, clip.sample_rate, clip.pause_before, clip.force)
                except Exception as exc2:
                    print(f"[audio] playback still failing after re-initializing audio: {exc2!r}")

    # Whether playback is currently sped up to work off a backlog - see _catch_up.
    _catching_up = False

    def _playback_backlog_s(self) -> float:
        """Seconds of synthesized speech waiting to be heard: every queued
        clip (with its lead-in pause) plus what's left of the one playing."""
        with self._audio.mutex:
            queued = sum(clip.seconds for clip in self._audio.queue)
        return queued + self.playback.pending_s()

    def _catch_up(self, clip: _Clip) -> _Clip:
        """Speeds `clip` up (pitch preserved) while playback is lagging - see
        AudioConfig.catchup_*. Only *playback* lag counts: speech already
        synthesized and waiting its turn, which is what makes the English
        drift further and further behind the speaker during a long stretch
        of continuous talking. Hysteresis keeps it from flapping: it starts
        at catchup_start_s of backlog and keeps going until the backlog is
        down to catchup_stop_s, then speech is back to normal speed."""
        if clip.force:
            return clip
        cfg = self.cfg.audio
        if cfg.catchup_max_rate <= 1.0:
            return clip
        backlog = self._playback_backlog_s()
        if cfg.max_backlog_s is not None and backlog > cfg.max_backlog_s:
            clip, backlog = self._skip_backlog(clip, backlog)
        if not self._catching_up and backlog >= cfg.catchup_start_s:
            self._catching_up = True
            print(f"[audio] speech playback is {backlog:.0f}s behind; speeding it up to catch up")
        elif self._catching_up and backlog <= cfg.catchup_stop_s:
            self._catching_up = False
            print("[audio] speech playback caught up; back to normal speed")
        if not self._catching_up:
            return clip

        # Streamed chunks of the same piece that are already waiting are
        # stretched together with this one, so the speed-up doesn't put a
        # seam in the middle of a word.
        parts = [clip]
        with self._audio.mutex:
            while self._audio.queue and self._audio.queue[0].seamless:
                parts.append(self._audio.queue.popleft())
        if len(parts) > 1:
            clip = _Clip(
                np.concatenate([part.audio for part in parts]), clip.sample_rate, clip.pause_before, seamless=clip.seamless
            )
        span = max(cfg.catchup_full_s - cfg.catchup_stop_s, 1e-6)
        rate = cfg.catchup_min_rate + (cfg.catchup_max_rate - cfg.catchup_min_rate) * (backlog - cfg.catchup_stop_s) / span
        rate = min(cfg.catchup_max_rate, max(cfg.catchup_min_rate, rate))
        vlog("audio", f"playback {backlog:.1f}s behind; playing the next {clip.seconds:.1f}s at {rate:.2f}x")
        return _Clip(time_stretch(clip.audio, clip.sample_rate, rate), clip.sample_rate, clip.pause_before / rate, seamless=clip.seamless)

    def _skip_backlog(self, clip: _Clip, backlog: float) -> tuple[_Clip, float]:
        """Drops the oldest waiting speech, whole sentences at a time (a
        sentence's streamed chunks go with it), until the backlog is under
        AudioConfig.max_backlog_s. Returns the clip to play next and the new
        backlog."""
        limit = self.cfg.audio.max_backlog_s
        skipped = 0.0
        with self._audio.mutex:
            waiting = self._audio.queue
            while backlog > limit and waiting:
                # Drop `clip` (already out of the queue, so not part of
                # `backlog`) and move on to the next sentence, skipping the
                # rest of this one's streamed chunks along with it.
                skipped += clip.seconds
                clip = waiting.popleft()
                backlog -= clip.seconds
                while clip.seamless and waiting:
                    skipped += clip.seconds
                    clip = waiting.popleft()
                    backlog -= clip.seconds
        if skipped:
            print(f"[audio] speech playback fell {backlog + skipped:.0f}s behind even sped up; "
                  f"skipped {skipped:.0f}s of it to catch up")
        return clip, backlog

    def _play(self, audio, sample_rate: int, pause_before: float, force: bool = False, seamless: bool = False) -> None:
        generation = self._playback_generation

        def should_continue() -> bool:
            if self._playback_generation != generation:
                return False  # the operator hit Reset while this clip was playing
            return True if force else self.control.voice_enabled.is_set()

        self.playback.play(
            audio, sample_rate=sample_rate, pause_before=pause_before,
            should_continue=should_continue,
            on_chunk=self._broadcast_output_level,
            seamless=seamless,
        )

    def _recover_audio(self, reason: str) -> bool:
        """Re-initializes the audio system and reopens the streams, for when
        the devices PortAudio knew about have gone stale under it - the
        laptop slept and woke, a device was plugged in or removed, or the
        default output changed. Without this, output kept failing with
        "Internal PortAudio error -9986" and the input could stop delivering
        until `sts` was restarted.

        Everything is closed first (terminating PortAudio invalidates open
        streams), PortAudio re-enumerates the machine's devices, a
        selected device that no longer exists falls back to the system
        default, and capture is reopened; the output stream reopens itself
        on the next clip. Safe to call from any thread - callers are
        serialized, and one arriving right after another recovery just
        reuses its result instead of doing it again. Returns whether audio
        looks usable afterwards."""
        with self._audio_recovery_lock:
            if time.monotonic() - self._last_audio_recovery < AUDIO_RECOVERY_MIN_INTERVAL_S:
                return self._last_audio_recovery_ok
            print(f"[audio] {reason}; re-initializing the audio system")
            ok = True
            with self._device_lock:
                audio = self.cfg.audio
                was_local = audio.source != "browser"
                try:
                    self.capture.stop()
                    self.playback.forget_devices()
                    reinitialize_audio_system()
                    if audio.input_device is not None:
                        try:
                            check_input(audio.input_device, audio.sample_rate, audio.channels)
                        except Exception as exc:
                            print(f"[audio] input device {audio.input_device} is gone ({exc}); using the system default")
                            audio.input_device = None
                    if audio.output_device is not None:
                        try:
                            check_output(audio.output_device, 24_000)
                        except Exception as exc:
                            print(f"[audio] output device {audio.output_device} is gone ({exc}); using the system default")
                            audio.output_device = None
                    if was_local:
                        self.capture.start()
                except Exception as exc:
                    print(f"[audio] couldn't re-initialize audio: {exc!r}")
                    ok = False
            self._last_audio_recovery = time.monotonic()
            self._last_audio_recovery_ok = ok
        if ok:
            self._broadcast_devices()
            self._broadcast_status()
        return ok

    def _check_audio_health(self) -> None:
        """One pass of the input watchdog: a local input stream that has gone
        quiet is dead (see AudioCapture.is_stalled) - recover it."""
        if self.capture.is_stalled(INPUT_STALL_S):
            self._recover_audio("input stream stopped delivering audio")

    def _check_stage_threads(self) -> None:
        """Starts any stage thread that has died again. _run_stage already
        survives an exception from any one item, so this only catches what
        gets past that (a BaseException, a bug in the loop itself) - but
        without it, that stage would be gone for the rest of the session with
        nothing but a traceback in the log to show for it."""
        for name, (target, thread) in list(self._threads.items()):
            if not thread.is_alive() and not self._stop.is_set():
                print(f"[{name}] stage thread died; restarting it")
                self._start_thread(name, target)

    def _audio_watchdog_loop(self) -> None:
        while not self._stop.wait(1.0):
            for check in (self._check_audio_health, self._check_stage_threads):
                try:
                    check()
                except Exception as exc:  # a health check must never take itself down
                    print(f"[watchdog] health check failed: {exc!r}")

    def _start_thread(self, name: str, target) -> None:
        thread = threading.Thread(target=target, name=f"sts-{name}", daemon=True)
        thread.start()
        self._threads[name] = (target, thread)

    def start(self) -> None:
        self._commands = queue.Queue()
        self.capture.start()
        self._stage("Listening", ready=True)
        for name, target in (
            ("control", lambda: self._run_stage("control", self._control_step)),
            ("vad", self._segment_loop),
            ("asr", self._asr_loop),
            ("translate", self._translate_loop),
            ("tts", self._tts_loop),
            ("playback", self._playback_loop),
        ):
            self._start_thread(name, target)
        # Not in self._threads: it's what watches them.
        threading.Thread(target=self._audio_watchdog_loop, name="sts-watchdog", daemon=True).start()
        self._broadcast_status()
        self._broadcast_library()
        self._broadcast_devices()
        self._broadcast_engines()

    def stop(self) -> None:
        self._stop.set()
        # Every stage wakes up within QUEUE_POLL_S to notice; one in the
        # middle of a long model call is a daemon thread and doesn't hold up
        # the exit.
        for _, thread in self._threads.values():
            thread.join(timeout=QUEUE_POLL_S * 2)
        self.capture.stop()
        self.playback.stop()
