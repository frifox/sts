"""Wires capture -> captions -> translation -> TTS -> playback into one
streaming pipeline, each stage on its own thread connected by queues so a
slow stage (translation, TTS) doesn't stall audio capture.

Captions (sts.transcript): the audio that isn't settled yet is
re-transcribed every ASRConfig.interval_ms; a sentence is locked once it
has settled (finished and unchanged for a while, or open too long - see
sts.transcript), and only then sent on to translation and voicing. The not-yet-locked text is pushed to
the caption page as it changes, as one running line.

Recognition/captions run by default (`self.control.active` set); an operator
can still pause them (e.g. during a long musical number) via Start/Stop.
Translation is a separate switch (`self.control.translation_enabled`, on by
default): turning it off leaves the original-language captions running but
skips translation (and, as a consequence, TTS, since there is no translated
text to speak). TTS *voicing* is its own switch and starts **off**
(`self.control.voice_enabled` unset) - captions are silent and harmless to
leave running, but audio spoken over the speakers during music or an
announcement would not be, so that's opt-in via its own button. Capture
always runs, and so does Silero VAD while "Train voice" is collecting the
speaker's speech.
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
import sounddevice as sd
import torch

from sts.asr import create_asr
from sts.asr.gigaam_asr import MODEL_NAME as ASR_MODEL_NAME
from sts.audio.capture import AudioCapture
from sts.audio.devices import check_input, check_output, reinitialize_audio_system
from sts.audio.playback import AudioPlayback
from sts.audio.tempo import time_stretch
from sts.audio.vad import SpeechSegmenter
from sts.tts.accent import ACCENT_DONORS
from sts.bible.lookup import BibleLookup
from sts.bible.reference import find_references
from sts.config import PipelineConfig
from sts.control import RecognitionControl
from sts.log import vlog
from sts.persistent_config import PersistentConfig
from sts.startup_progress import StartupProgress
from sts.translate.translator import Translator
from sts.tts import create_synthesizer
from sts.transcript import Sentence, Transcript
from sts.tts.voice_library import BUNDLED_VOICES_DIR, VoiceLibrary
from sts.web.server import CaptionBroadcaster
from sts.worker import StageProcess

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

# Silence between two locked sentences (seconds) above which the voice
# treats the second as a fresh thought - the longer pause_between_s before
# it - rather than the next sentence of the same one (pause_within_s).
SPEECH_PAUSE_S = 1.0

# Pause after a stage step raises before taking the next item, so an error
# that repeats on every item (a broken model, a device gone) logs at a
# readable rate instead of spinning a CPU core.
STAGE_ERROR_BACKOFF_S = 0.5
# How long Clone's finished step list stays in the Voice dropdown.
CLONE_PROGRESS_LINGER_S = 5.0


def _is_cuda_oom(exc: BaseException) -> bool:
    """torch.OutOfMemoryError, or a library's own RuntimeError that surfaces
    a CUDA out-of-memory failure in its message."""
    return isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()


def _free_gpu_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _stage_model(factory, cfg, name: str):
    """`factory(cfg)`, in a process of its own when it runs on mps (see
    sts.worker for why); elsewhere, in this process as before."""
    if cfg.device == "mps":
        return StageProcess(factory, cfg, name=f"sts-{name}")
    return factory(cfg)


def _wait_ready(model) -> None:
    if isinstance(model, StageProcess):
        model.wait_ready()


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
    # The first clip of a translated sentence, and when (time.monotonic())
    # that sentence's caption became final - what playback lateness is
    # measured from (see Pipeline._catch_up).
    sentence_start: bool = False
    finalized_at: float | None = None
    # The translated sentence (its caption `seq`) this is speech of, for the
    # page's voice-state indicators; None for the Test clip.
    seq: int | None = None

    @property
    def seconds(self) -> float:
        return self.pause_before + self.audio.size / self.sample_rate


def _device_name(index: int | None, kind: str) -> str | None:
    """The name of audio device `index` (None: the system default - not a
    specific device)."""
    if index is None:
        return None
    try:
        return sd.query_devices(index, kind)["name"]
    except Exception:
        return None


def _device_by_name(name: str | None, kind: str, index: int | None) -> int | None:
    """The index `name` has now (the device list was just re-read), or None
    - the system default - if it's gone. A device that had no name to go by
    keeps `index`'s default meaning."""
    if name is None:
        return None if index is None else index
    for i, device in enumerate(sd.query_devices()):
        if device["name"] == name and device[f"max_{kind}_channels"] > 0:
            return i
    print(f"[audio] {kind} device {name!r} is gone after the rescan; using the system default")
    return None


class Pipeline:
    # Bumped by _reset(). ASR, translation, and TTS each note it before a
    # slow call and drop their result if it changed meanwhile - so text or
    # speech from the previous speaker that was already mid-flight when the
    # operator hit Reset doesn't show up after the queues were drained.
    # (A class attribute so test stubs built with object.__new__ get it too.)
    _reset_generation = 0
    # Voice progress per translated sentence (by its caption seq) - see
    # _set_voice_state. Created on first use, so test stubs get it too.
    _voice_states: dict | None = None
    _voice_generated: set | None = None
    _voice_now_playing: int | None = None
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
        startup: StartupProgress | None = None,
    ):
        self.cfg = cfg
        self.broadcaster = broadcaster
        self._startup = startup  # times start-up's steps for the page's percentage (see _stage)
        # Remembers the operator's device/volume/voice choices across
        # restarts - see the persist-on-change calls below and
        # _restore_persisted_voice(). None (the default) just means nothing
        # is persisted, e.g. in tests that build a Pipeline stub directly.
        self._persistent_config = persistent_config
        # The caption page's "Show unfinished captions" - on unless it was
        # turned off before (remembered across restarts).
        saved = persistent_config.get("show_unfinished") if persistent_config is not None else None
        self._show_unfinished = saved != "0"
        self.capture = AudioCapture(cfg.audio)
        # The local input ("alsa" or "none") the browser mic returns to when it stops.
        self._local_source = cfg.audio.source if cfg.audio.source != "browser" else "alsa"
        self.playback = AudioPlayback(cfg.audio)
        self._stage("Loading speech detector…")
        self.segmenter = SpeechSegmenter(cfg.audio, cfg.vad)  # for "Train voice" only
        # On mps, translation and TTS each run in a process of their own (see
        # sts.worker), started first so they load while ASR does.
        self.translator = _stage_model(Translator, cfg.translation, "translation")
        self.tts = _stage_model(create_synthesizer, cfg.tts, "tts")
        self._stage("Loading speech recognition…")
        self.asr = create_asr(cfg.asr)
        self._stage("Loading translation model…")
        _wait_ready(self.translator)
        self._stage(f"Loading voice synthesis ({cfg.tts.backend})…")
        _wait_ready(self.tts)
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
        self._voice_cloned = False  # a cloned (zero-shot) voice is active - see _broadcast_status
        # Held around every use of self.asr: captions and a voice-training
        # transcription can otherwise run on the one model at the same time.
        self._asr_lock = threading.Lock()
        self._last_output_level_broadcast = 0.0

        # Live captions: capture feeds this while Caption is on, and
        # _caption_step re-transcribes what isn't locked yet.
        self.transcript = Transcript(
            self.asr, cfg.asr.final_after_unchanged_seconds, cfg.asr.final_force_after_seconds,
            cfg.audio.sample_rate, after_silence=cfg.asr.final_after_silence_seconds,
        )
        # (text, seq, spoken seconds, continues, finalized at) per final sentence, for translation.
        self._transcripts: queue.Queue = queue.Queue()
        self._original_seq = 0  # next sequence number for a broadcast original sentence
        self._last_final_end: float | None = None  # when the last locked sentence ended
        self._recent_text = ""  # the not-yet-locked caption text the page last got
        self._finalize_timers: list = []  # the finalize timers the page last got
        self._last_caption_step = time.monotonic()
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
        if self._startup is not None:
            if ready:
                self._startup.finish()
            else:
                self._startup.begin(text)
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
            self._voice_cloned = True
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
        "test_output", "train_start", "train_stop", "set_output_volume", "show_unfinished",
        "refresh_devices", "rescan_devices", "set_audio_devices", "mic_start", "mic_stop",
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
            # Whatever was heard but not locked yet counts as final now - it
            # would otherwise never be translated, and is dropped from the page.
            self._publish_captions(self.transcript.flush(), [])
            self._publish_finalize_timer()
        elif action == "voice_on":
            self.control.enable_voice()
        elif action == "voice_off":
            self.control.disable_voice()
            self._clear_voice_states()
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
            self._begin_clone_progress()
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
        elif action in ("refresh_devices", "rescan_devices", "set_audio_devices", "mic_start", "mic_stop"):
            threading.Thread(target=self._handle_device_action, args=(action, data), daemon=True).start()
            return
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
        elif action == "show_unfinished":
            self._show_unfinished = bool(data.get("on"))
            if self._persistent_config is not None:
                self._persistent_config.set("show_unfinished", "1" if self._show_unfinished else "0")
        elif action == "set_accent_donor":
            self._set_accent_donor(str(data.get("donor") or ""))
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
        self._clone_progress = None
        # After anything already queued - a voice being trained or picked
        # just before Reset must not become active again after it.
        self._on_control_thread(self._reset_voice)
        self._reset_generation += 1
        self.transcript.clear()
        self._recent_text = ""
        self._publish_finalize_timer()
        self._last_final_end = None
        self._drain(self._transcripts)
        self._drain(self._translations)
        self._drain(self._audio)
        self._clear_voice_states()
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

    def _select_speaker(self, speaker_id: str | None, preparing_label: str | None = None) -> None:
        self._pending_voice = None
        if not speaker_id:
            self._next_voice_request()  # drops a voice still being prepared
            self.tts.reset_voice()
            self._voice_cloned = False
            self._active_speaker_id = None
            return
        entry = self.voice_library.get(speaker_id)
        if entry is None:
            return

        def apply() -> None:
            self.tts.clone_from_entry(entry)
            self._voice_cloned = entry.kind != "lora"
            self._active_speaker_id = speaker_id

        if entry.kind == "lora":
            self._next_voice_request()
            apply()
        else:
            self._clone_when_ready(str(entry.wav_path), entry.ref_text, apply, preparing_label or entry.name, speaker_id)

    def _next_voice_request(self) -> int:
        self._voice_request = getattr(self, "_voice_request", 0) + 1
        self._preparing_voice = None
        return self._voice_request

    def _clone_when_ready(self, wav_path: str, ref_text: str | None, apply, name: str, speaker_id: str | None,
                          progress: bool = False) -> None:
        """Runs `apply` (the clone) on the control thread - straight away,
        unless the voice first needs its English reference made (Qwen3, a
        non-English clip, not made before): tens of seconds of synthesis that,
        done by the voice stage itself, stopped the current voice mid-speech.
        That's made in a helper process instead (on mps; a thread elsewhere)
        while the current voice keeps speaking, and the switch happens once
        it's there - unless another voice was chosen meanwhile. During start-up
        (before the pipeline runs) it's simply done here, waiting."""
        request = self._next_voice_request()
        from sts.tts.english_reference import needs_english_reference

        if not getattr(self, "_started", False) or not needs_english_reference(self.cfg.tts, wav_path, ref_text):
            if progress:
                self._clone_step("english", "skipped")  # an English voice, or its reference is already made
            apply()
            return
        if progress:
            self._clone_step("english", "running")
        self._preparing_voice = (name, speaker_id)
        vlog("voice", f"preparing {name!r} alongside the current voice (its English reference)")
        self._broadcast_status()

        def finish() -> None:
            if request != getattr(self, "_voice_request", 0):
                if progress:
                    self._clone_progress = None  # another voice was chosen meanwhile
                    self._broadcast_status()
                return
            self._preparing_voice = None
            apply()
            vlog("voice", f"{name!r} is ready")

        def prepare() -> None:
            try:
                self._make_english_reference(wav_path)
                if progress:
                    self._clone_step("english", "done")
            except Exception as exc:
                print(f"[voice] couldn't prepare {name!r} alongside ({exc}); preparing it in the voice stage")
                if progress:
                    self._clone_step("english", "failed")
            self._on_control_thread(finish)

        threading.Thread(target=prepare, name="sts-voice-prep", daemon=True).start()

    def _make_english_reference(self, wav_path: str) -> None:
        """On mps the voice stage is a process serving one request at a time,
        so a second one (the same model, loaded for the job) makes it; it's
        closed again after, freeing its memory. Elsewhere the voice stage is in
        this process and takes turns per generation with live synthesis."""
        if not isinstance(self.tts, StageProcess):
            self.tts.make_english_reference(wav_path)
            return
        from dataclasses import replace

        helper = StageProcess(create_synthesizer, replace(self.cfg.tts, qwen3_preload_presets=False), name="sts-voice-prep")
        try:
            helper.make_english_reference(wav_path)
        finally:
            helper.close()

    def _set_accent_donor(self, donor: str) -> None:
        """Voice dropdown: which American preset voice English references take
        their accent from ("": none). Runs on the control thread: the active
        voice is cloned again with it, which makes a new reference the first
        time (seconds)."""
        if self.cfg.tts.backend != "qwen3" or (donor and donor not in ACCENT_DONORS):
            return
        self.cfg.tts.qwen3_accent_donor = donor
        self.tts.set_accent_donor(donor)
        if self._persistent_config is not None:
            self._persistent_config.set("qwen3_accent_donor", donor)
        vlog("voice", f"base voice for cloning: {donor or 'none'}")
        # The note while it's prepared says what's happening: the same voice, re-made.
        base = f"base voice {donor}" if donor else "no base voice"
        if self._active_speaker_id:
            entry = self.voice_library.get(self._active_speaker_id)
            label = f"{entry.name if entry else 'the voice'} with {base}"
            self._select_speaker(self._active_speaker_id, preparing_label=label)
        elif self._pending_voice is not None:
            path, ref_text = self._pending_voice
            self._clone_when_ready(path, ref_text, lambda: self.tts.clone_voice(path, ref_text),
                                   f"the cloned voice with {base}", None)

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
            self._clone_step("record", "done")
            self._on_control_thread(self._activate_trained_voice, audio)
        else:
            self._clone_progress = None  # cancelled (Reset), or nothing was heard
            self._broadcast_status()

    # -- Clone's progress, shown step by step in the Voice dropdown ---------

    CLONE_STEPS = (
        ("record", "Recording the speaker"),
        ("transcribe", "Transcribing what they said"),
        ("english", "Preparing the voice's English"),
        ("activate", "Switching to the new voice"),
    )

    def _begin_clone_progress(self) -> None:
        self._clone_progress = [{"key": key, "label": label, "state": "pending"} for key, label in self.CLONE_STEPS]
        self._clone_step("record", "running")

    def _clone_step(self, key: str, state: str) -> None:
        """Sets one step's state: pending, running, done, skipped or failed.
        Once every step is over, the list stays up a few seconds, then goes."""
        progress = getattr(self, "_clone_progress", None)
        if not progress:
            return
        for step in progress:
            if step["key"] == key:
                step["state"] = state
        self._broadcast_status()
        if all(step["state"] in ("done", "skipped", "failed") for step in progress):
            def clear() -> None:
                if self._clone_progress is progress:
                    self._clone_progress = None
                    self._broadcast_status()

            timer = threading.Timer(CLONE_PROGRESS_LINGER_S, clear)
            timer.daemon = True
            timer.start()

    def _activate_trained_voice(self, audio: np.ndarray) -> None:
        path = self._write_training_wav(audio)
        self._clone_step("transcribe", "running")
        ref_text = self._transcribe_or_none(audio)
        self._clone_step("transcribe", "done" if ref_text else "failed")
        # Activate the clone as soon as it's ready so the operator can listen to
        # it, but don't write it to the library until they choose to (Save Voice).
        def apply() -> None:
            self._clone_step("activate", "running")
            self.tts.clone_voice(path, ref_text)
            self._voice_cloned = True
            self._active_speaker_id = None
            self._pending_voice = (path, ref_text)
            vlog("voice", "trained voice is active but not saved yet")
            self._clone_step("activate", "done")

        self._clone_when_ready(path, ref_text, apply, "the cloned voice", None, progress=True)

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
            # The lock: the caption stage may be mid-transcribe on the same model.
            with self._asr_lock:
                return self.asr.transcribe(audio) or None
        except Exception as exc:  # a failed transcript just means a lower-quality clone, not a lost one
            print(f"[voice] couldn't transcribe reference clip, falling back to embedding-only clone: {exc}")
            return None

    def _save_and_activate(self, name: str, wav_source: str, ref_text: str | None) -> None:
        entry = self.voice_library.save(name, wav_source, ref_text)
        self.tts.clone_from_entry(entry)
        self._voice_cloned = entry.kind != "lora"
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
        from sts.audio.permission import is_blocked

        message.update(
            error=error,
            applied=applied,
            # Only matters while a local device is the input: the browser mic has its own permission.
            mic_blocked=self.cfg.audio.source == "alsa" and is_blocked(self.capture.mic_permission),
            # A local device side turned off (see _handle_device_action).
            input_off=self.cfg.audio.source == "none",
            output_off=not self.cfg.audio.output_enabled,
        )
        self.broadcaster.broadcast_devices(message)

    def _handle_device_action(self, action: str, data: dict) -> None:
        # Opening audio streams can take a moment, so this runs off the web server's event loop.
        if action == "refresh_devices":
            self._broadcast_devices()
            return
        if action == "rescan_devices":
            self._rescan_audio_devices()
            return
        if action == "mic_start":
            with self._device_lock:
                if self.cfg.audio.source != "browser":
                    self._local_source = self.cfg.audio.source  # what mic_stop goes back to
                self.capture.switch_to_browser()
            vlog("audio", "input switched to this browser's microphone")
            self._broadcast_status()
            self._broadcast_devices()
            return
        if action == "mic_stop":
            with self._device_lock:
                # Back to the local input as it was: a device, or off.
                if self._local_source == "none":
                    self.capture.switch_to_none()
                else:
                    self.capture.switch_to_alsa()
            vlog("audio", f"input switched back to the local input ({self.cfg.audio.source})")
            self._broadcast_status()
            self._broadcast_devices()
            return

        # `input` / `output`: a device index, "" for the system default, or
        # "off" to turn that side's local device off (nothing captured /
        # nothing played locally; the browser mic and voice streaming still
        # work). A side left out of the request stays as it is.
        def parse(value):
            if value == "off":
                return "off"
            return None if value in (None, "") else int(value)

        try:
            new_input = parse(data["input"]) if "input" in data else ...
            new_output = parse(data["output"]) if "output" in data else ...
        except (TypeError, ValueError):
            return
        audio = self.cfg.audio
        # Picking an input device always means "use ALSA with this device" -
        # including when the browser mic was active (or the input off) and
        # the index itself hasn't changed, since capture isn't reading it yet.
        if new_input is ...:
            input_changed = False
        elif new_input == "off":
            input_changed = audio.source != "none"
        else:
            input_changed = new_input != audio.input_device or audio.source != "alsa"
        if new_output is ...:
            output_changed = False
        elif new_output == "off":
            output_changed = audio.output_enabled
        else:
            output_changed = new_output != audio.output_device or not audio.output_enabled
        with self._device_lock:
            try:
                from sts.audio.devices import check_input, check_output

                if input_changed and new_input != "off":
                    check_input(new_input, audio.sample_rate, audio.channels)
                if output_changed and new_output != "off":
                    check_output(new_output, 24_000)
                if input_changed:
                    if new_input == "off":
                        self.capture.switch_to_none()
                    else:
                        self.capture.switch_to_alsa(new_input)
                if output_changed:
                    if new_output == "off":
                        audio.output_enabled = False
                    else:
                        audio.output_device = new_output
                        audio.output_enabled = True
                    self.playback.stop()  # the output stream reopens on the new device (if any) with the next clip
            except Exception as exc:
                print(f"[audio] couldn't apply the selected devices: {exc}")
                self._broadcast_devices(error=f"Couldn't use that device: {exc}")
                return
            vlog("audio", f"devices applied: input={'off' if audio.source == 'none' else audio.input_device}, "
                          f"output={audio.output_device if audio.output_enabled else 'off'}")
            if self._persistent_config is not None:
                # A real ALSA device, or off, is a restorable choice - not the
                # browser-mic mode, which is inherently session-only.
                if input_changed:
                    self._persistent_config.set(
                        "input_device",
                        "off" if audio.source == "none" else "" if audio.input_device is None else str(audio.input_device),
                    )
                if output_changed:
                    self._persistent_config.set(
                        "output_device",
                        "off" if not audio.output_enabled
                        else "" if audio.output_device is None else str(audio.output_device),
                    )
            self._broadcast_devices(applied=True)
        self._broadcast_status()  # audio.source may have just changed too

    def _rescan_audio_devices(self) -> None:
        """The page's refresh button: finds audio devices plugged in (or
        removed) since start-up. PortAudio only lists the machine's devices
        when it starts, so it's restarted - which closes every audio stream
        (the output waits for the clip playing to finish) - and the devices
        in use are found again by name, since their numbers can change.
        One that's gone falls back to the system default."""
        with self._device_lock:
            audio = self.cfg.audio
            input_name = _device_name(audio.input_device, "input")
            output_name = _device_name(audio.output_device, "output")
            self.capture.stop()
            try:
                self.playback.while_idle(reinitialize_audio_system)
            except Exception as exc:
                print(f"[audio] couldn't rescan the audio devices: {exc!r}")
            audio.input_device = _device_by_name(input_name, "input", audio.input_device)
            audio.output_device = _device_by_name(output_name, "output", audio.output_device)
            if audio.source == "alsa":
                try:
                    self.capture.start()
                except Exception as exc:
                    print(f"[audio] input device didn't reopen after the rescan ({exc}); using the system default")
                    audio.input_device = None
                    self.capture.start()
            if self._persistent_config is not None:
                for key, device, on in (("input_device", audio.input_device, audio.source != "none"),
                                        ("output_device", audio.output_device, audio.output_enabled)):
                    if on:
                        self._persistent_config.set(key, "" if device is None else str(device))
            vlog("audio", f"rescanned audio devices: input={audio.input_device}, output={audio.output_device}")
        self._broadcast_devices(applied=True)
        self._broadcast_status()

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
                # Remembered here, not asked of the voice stage: on mps that runs in a
                # process of its own, and asking waits until the sentence it's
                # synthesizing is done - holding up this status (a Voice off
                # click's confirmation) and the web server's whole event loop.
                cloned=getattr(self, "_voice_cloned", False),
                can_save=self._pending_voice is not None,
                output_volume=round(self.cfg.audio.volume * 100),
                audio_source=self.cfg.audio.source,
                can_design=hasattr(self.tts, "design_reference"),
                default_voice=self.cfg.tts.resolve_voice(self.cfg.tts.default_voice),
                voice_groups=self.cfg.tts.voice_groups,
                active_speaker_id=self._active_speaker_id,
                show_unfinished=getattr(self, "_show_unfinished", True),
                accent_donor=self.cfg.tts.qwen3_accent_donor,
                clone_steps=getattr(self, "_clone_progress", None),
                preparing_voice=(getattr(self, "_preparing_voice", None) or ("", None))[0],
                preparing_speaker_id=(getattr(self, "_preparing_voice", None) or ("", None))[1],
                accent_donors=list(ACCENT_DONORS) if self.cfg.tts.backend == "qwen3" else [],
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
            self._last_library = speakers
            self.broadcaster.broadcast_library(speakers)

    def _check_library(self) -> None:
        """Watchdog pass, at most every 5 s: re-sends the voice list when the
        library changed behind the app's back - voices added by sts-add-voices
        (or copied in) while it runs, which otherwise only showed after a
        restart."""
        now = time.monotonic()
        if not self.broadcaster or now - getattr(self, "_last_library_check", 0.0) < 5.0:
            return
        self._last_library_check = now
        speakers = [{"id": e.speaker_id, "name": e.name, "builtin": e.builtin} for e in self.voice_library.list()]
        if speakers != getattr(self, "_last_library", None):
            self._broadcast_library()

    def _engines_info(self) -> dict:
        """Model, device and memory (bytes of model weights - see
        sts.gpu.model_bytes) per stage, for the header dropdowns."""

        def memory(component) -> int | None:
            try:
                return component.memory_bytes()
            except Exception:
                return None

        tts_model = self.cfg.tts.qwen3_model_name if self.cfg.tts.backend == "qwen3" else self.cfg.tts.model_name
        tts_device = self.cfg.tts.device
        if self.cfg.tts.backend == "qwen3":
            from sts.tts.qwen3_engine import resolve_engine

            if resolve_engine(self.cfg.tts.qwen3_engine) == "mlx":
                from sts.tts.qwen3_mlx import mlx_model_name

                tts_model, tts_device = mlx_model_name(tts_model), f"{tts_device} (MLX)"
        return {
            "asr": {"model": ASR_MODEL_NAME, "device": self.cfg.asr.device, "memory": memory(self.asr)},
            "translation": {
                "model": self.cfg.translation.model_name, "device": self.cfg.translation.device,
                "memory": memory(self.translator),
            },
            "tts": {"model": tts_model, "device": tts_device, "memory": memory(self.tts)},
        }

    def _broadcast_engines(self) -> None:
        if self.broadcaster:
            self._last_engines = self._engines_info()
            self.broadcaster.broadcast_engines(self._last_engines)

    def _check_engines(self) -> None:
        """Watchdog pass: re-sends the engines info when a stage's memory has
        changed - the voice stage loads and drops models as it goes (Qwen3's
        voice designer, preset model, trained voices) - at most every 5 s."""
        now = time.monotonic()
        if now - getattr(self, "_last_engines_check", 0.0) < 5.0:
            return
        self._last_engines_check = now
        if self.broadcaster and self._engines_info() != getattr(self, "_last_engines", None):
            self._broadcast_engines()

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
        self._last_level_broadcast = 0.0
        self._run_stage("capture", self._segment_step)

    def _segment_step(self) -> None:
        frame = self.capture.frames.get(timeout=QUEUE_POLL_S)
        now = time.monotonic()
        if now - self._last_level_broadcast >= self._LEVEL_METER_INTERVAL_S:
            self._broadcast_input_level(frame)
            self._last_level_broadcast = now
        self._process_captured_frame(frame)

    def _process_captured_frame(self, frame: np.ndarray) -> None:
        """Hands one captured frame to the captions (while Caption is on),
        and - while "Train voice" is collecting - to Silero VAD, whose
        segments are the speaker's actual speech, pauses left out."""
        if self.control.active.is_set():
            self.transcript.add(frame)
        if not self.control.training.is_set():
            return
        segment = self.segmenter.process(frame)
        if segment is not None:
            reached_target = self.control.accumulate_training_segment(segment)
            self._broadcast_status()
            if reached_target:
                self._finish_training()

    def _caption_loop(self) -> None:
        self._run_stage("asr", self._caption_step)

    def _caption_step(self) -> None:
        """One caption update: re-transcribes what isn't locked yet, sends
        newly locked sentences on (see _publish_captions), and pushes the
        still-open text to the page."""
        if self._stop.wait(self.cfg.asr.interval_ms / 1000) or not self.transcript.has_audio:
            return
        generation = self._reset_generation
        t0 = time.monotonic()
        with self._asr_lock:
            final, still_open = self.transcript.step()
        processing_s = time.monotonic() - t0
        # Real-time factor: the time a step took over the audio that arrived
        # since the previous one - above 1.0, captions fall behind the speaker.
        self._broadcast_rtf("asr", processing_s, t0 - self._last_caption_step)
        self._last_caption_step = t0
        if generation != self._reset_generation:
            return  # Reset landed mid-step; this is the previous speaker's text
        self._publish_captions(final, still_open)
        self._publish_finalize_timer()

    def _publish_finalize_timer(self) -> None:
        """How far each still-open caption sentence is from being finalized
        by a time rule (Transcript.finalize_timers) - the page draws them as
        circles emptying clockwise - sent whenever they change."""
        timers = list(self.transcript.finalize_timers)
        if timers != self._finalize_timers:
            self._finalize_timers = timers
            if self.broadcaster:
                self.broadcaster.broadcast_finalize_timers(timers)

    def _publish_captions(self, final: list[Sentence], still_open: list[str]) -> None:
        """Locked sentences: shown as final, checked for Bible references,
        and queued for translation (and so voicing). The still-open text
        replaces the page's live caption line."""
        for sentence in final:
            seq = self._original_seq
            self._original_seq += 1
            print(f"[ru] {sentence.text}")
            # Spoken right on from the previous sentence (no real pause) -> the
            # voice runs them together like one thought; after a pause, it
            # leaves the longer between-thoughts gap.
            continues = self._last_final_end is not None and sentence.start - self._last_final_end < SPEECH_PAUSE_S
            self._last_final_end = sentence.end
            if self.broadcaster:
                self.broadcaster.broadcast_original(sentence.text, seq, seq)
                self._broadcast_bible_references(sentence.text)
            self._transcripts.put((sentence.text, seq, sentence.end - sentence.start, continues, time.monotonic()))
        recent = " ".join(still_open)
        if recent != self._recent_text:
            self._recent_text = recent
            if self.broadcaster:
                self.broadcaster.broadcast_original_recent(recent, list(still_open))

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
                quote.ru_label, quote.chapter, quote.verse, quote.verse_end, quote.ru_text, quote.en_text,
                en_reference=quote.en_reference,
            )

    def _translate_loop(self) -> None:
        self._run_stage("translate", self._translate_step)

    def _translate_step(self) -> None:
        text, seq, spoken_s, continues, finalized_at = self._transcripts.get(timeout=QUEUE_POLL_S)
        if not self.control.translation_enabled.is_set():
            vlog("translate", "translation is off, skipping")
            return
        self._translate_sentence(text, seq, spoken_s, continues, finalized_at)

    def _translate_sentence(self, text: str, seq: int | None = None, spoken_s: float = 0.0,
                            continues: bool = False, finalized_at: float | None = None) -> None:
        """Translates one locked caption sentence, broadcasts it with the
        original's sequence number (the page pairs them by it), and queues
        it for TTS. `spoken_s` - how long the original took to say - is the
        translation real-time factor's denominator. `continues` rides along
        so _tts_step gives it the short within-a-thought pause, and
        `finalized_at` so playback can tell how late it's spoken."""
        generation = self._reset_generation
        t0 = time.monotonic()
        translated = " ".join(self.translator.translate_sentences(text)).strip()
        if generation != self._reset_generation:
            vlog("translate", "dropping a translation from before Reset")
            return
        processing_s = time.monotonic() - t0
        vlog("translate", f"translated a sentence in {processing_s:.2f}s")
        if not translated:
            return
        print(f"    [en] {translated}")
        if self.broadcaster:
            self.broadcaster.broadcast_translated(translated, seq, seq)
        self._broadcast_rtf("translate", processing_s, spoken_s)
        control = getattr(self, "control", None)
        if control is not None and control.voice_enabled.is_set():
            self._set_voice_state(seq, "queued")
        self._translations.put((translated, continues, finalized_at, seq))

    def _tts_loop(self) -> None:
        self._run_stage("tts", self._tts_step)

    def _tts_step(self) -> None:
        text, continues, finalized_at, seq = self._translations.get(timeout=QUEUE_POLL_S)
        if not self.control.voice_enabled.is_set():
            vlog("tts", "voice is off, skipping synthesis")
            return
        self._set_voice_state(seq, "generating")
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
            # A longer pause before a sentence the speaker started after a
            # real pause, a shorter one before one they ran straight on to
            # (`continues` - see _publish_captions) and between the pieces
            # of one sentence. A streamed chunk that continues the same
            # piece mid-word gets no pause at all.
            if seamless:
                pause = 0.0
            elif first_piece:
                pause = self.cfg.audio.pause_within_s if continues else self.cfg.audio.pause_between_s
            else:
                pause = self.cfg.audio.pause_within_s
            self._audio.put(_Clip(
                audio, sample_rate, pause, seamless=seamless, sentence_start=first_piece, finalized_at=finalized_at,
                seq=seq,
            ))
            first_piece = False
            t0 = time.monotonic()
        self._voice_generation_done(seq)

    # -- voice-state indicators ------------------------------------------
    # Per translated sentence, while Voice is on: queued -> generating ->
    # generated -> playing -> played (a streamed sentence starts playing
    # while still generating). Sent to the page (keyed by caption seq),
    # which marks each translated sentence with it.

    def _voice_lock(self) -> threading.Lock:
        if self.__dict__.get("_voice_state_lock") is None:
            self._voice_state_lock = threading.Lock()
            self._voice_states, self._voice_generated = {}, set()
        return self._voice_state_lock

    def _set_voice_state(self, seq: int | None, state: str | None) -> None:
        if seq is None:
            return
        with self._voice_lock():
            if state is None or state == "played":
                self._voice_states.pop(seq, None)
                self._voice_generated.discard(seq)
            else:
                self._voice_states[seq] = state
        if self.broadcaster and hasattr(self.broadcaster, "broadcast_voice_state"):
            self.broadcaster.broadcast_voice_state(seq, state)

    def _voice_generation_done(self, seq: int | None) -> None:
        """All of `seq`'s speech is synthesized: "generated" if it isn't
        playing yet; if it is, "played" once nothing of it is left to play."""
        if seq is None:
            return
        with self._voice_lock():
            self._voice_generated.add(seq)
            state = self._voice_states.get(seq)
            finished = state == "playing" and self._voice_now_playing != seq and not self._queued_speech_of(seq)
        if state == "generating":
            self._set_voice_state(seq, "generated")
        elif finished:
            self._set_voice_state(seq, "played")

    def _queued_speech_of(self, seq: int) -> bool:
        with self._audio.mutex:
            return any(clip.seq == seq for clip in self._audio.queue)

    def _voice_playing(self, seq: int | None) -> None:
        """A clip of `seq` starts playing: anything before it has played."""
        if seq is None:
            return
        with self._voice_lock():
            self._voice_now_playing = seq
            earlier = [s for s in self._voice_states if s < seq]
            already = self._voice_states.get(seq) == "playing"
        for s in earlier:
            self._set_voice_state(s, "played")
        if not already:
            self._set_voice_state(seq, "playing")

    def _voice_clip_done(self, seq: int | None) -> None:
        """A clip of `seq` finished playing: "played" if it was the last."""
        if seq is None:
            return
        with self._voice_lock():
            self._voice_now_playing = None
            finished = seq in self._voice_generated and not self._queued_speech_of(seq)
        if finished:
            self._set_voice_state(seq, "played")

    def _clear_voice_states(self) -> None:
        """Voice off / Reset: what was waiting to be spoken won't be - its
        indicators go (played ones stay)."""
        with self._voice_lock():
            pending = list(self._voice_states)
            self._voice_now_playing = None
        for seq in pending:
            self._set_voice_state(seq, None)

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
        self._voice_playing(clip.seq)
        try:
            self._play(clip.audio, clip.sample_rate, clip.pause_before, clip.force, clip.seamless)
            self._voice_clip_done(clip.seq)
        except Exception as exc:
            print(f"[audio] playback failed: {exc!r}")
            # Typically the output device went stale (laptop slept, the
            # default output changed) - re-initialize audio and give this
            # same clip one more try instead of dropping it, and then
            # every clip after it.
            if self._recover_audio("playback failed"):
                try:
                    self._play(clip.audio, clip.sample_rate, clip.pause_before, clip.force)
                    self._voice_clip_done(clip.seq)
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
        """Speeds `clip` up (pitch preserved) while the voice is running late
        - see AudioConfig.catchup_*.

        Late means: a sentence starts being spoken long after its caption
        became final. Only that counts - not the time the text spent
        un-finalized before (a delay captions have by design, not playback
        falling behind), and not how much audio is queued (one long
        sentence waiting its turn isn't lateness if it will still start on
        time). It's judged at the start of each sentence, and every piece
        of that sentence plays at the speed chosen then. Hysteresis keeps it
        from flapping: speed-up starts once a sentence is catchup_start_s
        late and keeps going until one starts within catchup_stop_s."""
        if clip.force:
            return clip
        cfg = self.cfg.audio
        backlog = self._playback_backlog_s()
        if cfg.max_backlog_s is not None and backlog > cfg.max_backlog_s:
            clip, backlog = self._skip_backlog(clip, backlog)
        if clip.sentence_start:
            late_s = time.monotonic() - clip.finalized_at if clip.finalized_at is not None else 0.0
            self._rate = self._catch_up_rate(late_s)
        rate = self._rate
        if rate <= 1.0:
            return clip

        # Streamed chunks of the same piece that are already waiting are
        # stretched together with this one, so the speed-up doesn't put a
        # seam in the middle of a word.
        parts = [clip]
        with self._audio.mutex:
            while self._audio.queue and self._audio.queue[0].seamless:
                parts.append(self._audio.queue.popleft())
        audio = np.concatenate([part.audio for part in parts]) if len(parts) > 1 else clip.audio
        vlog("audio", f"playing the next {audio.size / clip.sample_rate:.1f}s at {rate:.2f}x")
        return _Clip(
            time_stretch(audio, clip.sample_rate, rate), clip.sample_rate, clip.pause_before / rate,
            seamless=clip.seamless, sentence_start=clip.sentence_start, finalized_at=clip.finalized_at,
            seq=clip.seq,
        )

    # Playback speed for the sentence now playing, chosen at its start - see _catch_up.
    _rate = 1.0

    def _catch_up_rate(self, late_s: float) -> float:
        """Playback speed for a sentence starting `late_s` after it became
        final, updating the catching-up state (with hysteresis)."""
        cfg = self.cfg.audio
        if cfg.catchup_max_rate <= 1.0:
            return 1.0
        if not self._catching_up and late_s >= cfg.catchup_start_s:
            self._catching_up = True
            print(f"[audio] voice is starting sentences {late_s:.0f}s after they're final; speeding it up to catch up")
        elif self._catching_up and late_s <= cfg.catchup_stop_s:
            self._catching_up = False
            print("[audio] voice caught up; back to normal speed")
        if not self._catching_up:
            return 1.0
        span = max(cfg.catchup_full_s - cfg.catchup_stop_s, 1e-6)
        rate = cfg.catchup_min_rate + (cfg.catchup_max_rate - cfg.catchup_min_rate) * (late_s - cfg.catchup_stop_s) / span
        return min(cfg.catchup_max_rate, max(cfg.catchup_min_rate, rate))

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
            for check in (self._check_audio_health, self._check_stage_threads, self._check_engines, self._check_library):
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
        self._started = True  # voices are prepared alongside from now on (see _clone_when_ready)
        self.capture.start()
        self._stage("Listening", ready=True)
        for name, target in (
            ("control", lambda: self._run_stage("control", self._control_step)),
            ("capture", self._segment_loop),
            ("asr", self._caption_loop),
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
        for model in (getattr(self, "translator", None), getattr(self, "tts", None)):
            if isinstance(model, StageProcess):
                model.close()
