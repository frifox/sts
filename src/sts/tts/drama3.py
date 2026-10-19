"""Speech with Fish Audio's Drama 3 (fishaudio/drama-3) on Apple's MLX
(mlx-audio's fish_drama3, the 4-bit conversion by default): 35B MoE, 3B active
per token, 44.1 kHz. Measured on 10 sermon sentences: ~0.37x real time (Qwen3
1.7B: ~0.27x), first streamed audio in ~0.2 s, ~27 GB of memory. Fish Audio
Community License - see its terms before commercial use or hosting.

It has no built-in voices: every voice comes from a reference recording and
its transcript. So:

- a voice with an English transcript (Default, Default 1.25, designed ones)
  is cloned from its own recording;
- one with any other transcript (a Russian speaker), or none, is cloned from
  its English reference - the clip Qwen3 makes of that voice speaking English
  (sts.tts.english_reference; the pipeline has it made, in a Qwen3 helper
  process, before switching): cloned from the Russian recording, the English
  came out slower and Russian-paced;
- with no voice chosen (or a built-in Qwen3 preset like Aiden picked), it
  speaks as the bundled Default voice - without a reference it would invent a
  different voice every sentence.

Everything around generation - streaming with the hold and the ramble checks,
the length checks, level and tone matching - is Qwen3TTSSynthesizer's, which
this extends; only loading, generating and choosing the reference differ.
"""

import json
import threading
from pathlib import Path
from collections import deque

import numpy as np

from sts.config import TTSConfig
from sts.gpu import gpu_lock
from sts.log import vlog
from sts.tts.english_reference import ENGLISH_BRIDGE_TEXT, cached_reference_path, transcript_is_english
from sts.tts.qwen3 import MIN_CAP_SECONDS, Qwen3TTSSynthesizer
from sts.tts.qwen3_mlx import _quiet_config_type_warning
from sts.tts.voice_library import BUNDLED_VOICES_DIR, VoiceEntry

DRAMA3_FIRST_CHUNK_FRAMES = 4  # a smaller first streamed piece: first audio sooner
# How many of the sentences just spoken each new one gets as history (their
# text and the audio codes generated for them): Drama 3 is built as a
# conversation, and continues the delivery of the turns before - pace, energy,
# loudness - instead of guessing anew for every sentence (its own long-text
# mode keeps the previous chunk for that reason).
HISTORY_SENTENCES = 2


def needs_english_reference(ref_text: str | None) -> bool:
    """Drama 3 clones a voice from its English reference unless its transcript is English."""
    return not transcript_is_english(ref_text)


class _Drama3Model:
    """mlx-audio's Drama 3 model, as Qwen3TTSSynthesizer's methods use a model."""

    def __init__(self, name: str):
        from mlx_audio.tts.utils import load_model

        _quiet_config_type_warning()
        self._model = load_model(name)

    @property
    def sample_rate(self) -> int:
        return int(self._model.sample_rate)

    @property
    def frames_per_second(self) -> float:
        return self.sample_rate / self._model.codec.frame_length

    def weight_bytes(self) -> int:
        from mlx.utils import tree_flatten

        return sum(array.nbytes for _, array in tree_flatten(self._model.parameters()))

    def stream(self, text: str, ref_audio: str, ref_text: str, interval_s: float, max_tokens: int,
               history=(), temperature=None, top_k=None, top_p=None):
        """The speech for `text` in pieces (np.float32) as it's generated, with
        `history` - earlier turns, (text, codes) - before it in the prompt. When
        it's done, `last_turns` holds this text's own turns, to be history for
        later ones. mlx-audio's generate() does the same for the chunks of one
        long text, but takes no history from outside: this is its loop, using
        the model's own pieces (pinned mlx-audio version, see pyproject.toml)."""
        import mlx.core as mx
        from mlx_audio.tts.models.fish_drama3 import drama3 as d3
        from mlx_audio.tts.models.fish_drama3.chunking import split_text

        model = self._model
        sampling = (
            d3.DEFAULT_TEMPERATURE if temperature is None else temperature,
            d3.DEFAULT_TOP_K if top_k is None else top_k,
            d3.DEFAULT_TOP_P if top_p is None else top_p,
        )
        voice = model._voice(ref_audio, ref_text)
        interval = max(1, round(interval_s * self.frames_per_second))
        model.codec.reset_streaming_state()
        self.last_turns = []
        turns = list(history)
        for index, chunk in enumerate(split_text(text)):
            frames, system_rows = model.build_frames(voice.references, turns[-HISTORY_SENTENCES:], chunk)
            codes, emitted = [], 0
            for frame in model._frames(frames, system_rows, voice, max_tokens, *sampling):
                codes.append(frame)
                due = DRAMA3_FIRST_CHUNK_FRAMES if index == 0 and emitted == 0 else interval
                if len(codes) - emitted >= due:
                    yield self._decode(codes[emitted:])
                    emitted = len(codes)
            if len(codes) > emitted:
                yield self._decode(codes[emitted:])
            if codes:
                turn = (chunk, mx.stack(codes, axis=1))
                turns.append(turn)
                self.last_turns.append(turn)

    def _decode(self, codes) -> np.ndarray:
        import mlx.core as mx

        audio = self._model.codec.streaming_step(mx.stack(codes, axis=1)[None])[0, 0]
        mx.eval(audio)
        return np.asarray(audio, dtype=np.float32).reshape(-1)

    def generate(self, text: str, ref_audio: str, ref_text: str, max_tokens: int, history=()) -> np.ndarray:
        pieces = list(self.stream(text, ref_audio, ref_text, 30.0, max_tokens, history=history))
        return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)


class Drama3Synthesizer(Qwen3TTSSynthesizer):
    # Drama 3 now and then stops talking partway through a take - after a
    # broken-off opening like "I just, uh, for us, ..." it did in ~40% of takes:
    # the first few words (0.3-4 s of a ~10 s sentence), then silence to the end
    # (see STALL_S). So a streamed take is held back until 60% of its expected
    # length has been spoken (generating runs ~2.7x faster than speech: ~2 s for
    # a long sentence); one that stops before that is caught before anything
    # plays, and regenerated - up to 5 takes in all.
    RELEASE_FRACTION = 0.6
    CUT_SHORT_FRACTION = 0.6
    STREAM_ATTEMPTS = 5

    def __init__(self, cfg: TTSConfig):  # not super().__init__: that would load Qwen3's models
        self.cfg = cfg
        self.engine = "mlx"
        vlog("tts", f"Drama 3: loading {cfg.drama3_model_name}")
        self.model = _Drama3Model(cfg.drama3_model_name)
        self._model_class = None
        self._custom_model = None
        self.default_voice = cfg.resolve_voice(cfg.default_voice)
        self._lock = gpu_lock(cfg.device) if cfg.device == "mps" else threading.Lock()
        self._pace_history: deque = deque(maxlen=20)
        self._clarity_history: deque = deque(maxlen=20)
        self._tone_history: deque = deque(maxlen=20)
        self._design_model = None
        self._design_load_lock = threading.Lock()
        self._prompt = None  # (Qwen3's clone prompt: unused here)
        self.cloned_speaker_wav: str | None = None
        self._lora_models: dict = {}
        self._active_lora_speaker_id: str | None = None
        self._reference: tuple[str, str] | None = None  # (recording, its transcript) the voice is cloned from
        self._history: deque = deque(maxlen=HISTORY_SENTENCES)  # (text, codes) of the sentences just spoken
        self.reset_voice()

    # -- voices ------------------------------------------------------------------

    def clone_voice(self, reference_wav_path: str, ref_text: str | None = None, speed: float | None = None) -> None:
        if needs_english_reference(ref_text):
            english = cached_reference_path(self.cfg, reference_wav_path)
            if not english.is_file():
                raise RuntimeError(f"no English reference made yet for {reference_wav_path} (Qwen3 makes it)")
            reference = (str(english), ENGLISH_BRIDGE_TEXT)
            vlog("tts", "voice conditioning: the voice's English reference + its text")
        else:
            reference = (reference_wav_path, ref_text)
            vlog("tts", "voice conditioning: clip + transcript")
        self._use(reference, speed)
        self.cloned_speaker_wav = reference_wav_path

    def clone_from_entry(self, entry: VoiceEntry) -> None:
        if entry.kind == "lora":
            raise RuntimeError(f"voice {entry.name!r} is a trained (LoRA) Qwen3 voice: not for Drama 3")
        self.clone_voice(str(entry.wav_path), entry.ref_text, entry.drama3_reference_speed)

    def reset_voice(self) -> None:
        """No voice chosen: the bundled Default voice (Drama 3 has no built-in ones)."""
        folder = BUNDLED_VOICES_DIR / "default"
        meta = json.loads((folder / "meta.json").read_text())
        self._use((str(folder / "reference.wav"), meta["ref_text"]), meta.get("drama3_reference_speed"))
        self.cloned_speaker_wav = None

    def set_default_voice(self, voice: str) -> None:
        """A Qwen3 preset voice (Aiden, ...) picked: Drama 3 has none; it speaks as Default."""
        self.default_voice = self.cfg.resolve_voice(voice)
        self.reset_voice()

    def _use(self, reference: tuple[str, str], speed: float | None = None) -> None:
        self._pace_history.clear()
        self._clarity_history.clear()
        self._tone_history.clear()
        audio, text = reference
        self._reference = (self._paced(audio, speed or self.cfg.drama3_reference_speed), text)
        self._history.clear()  # a new voice: the last one's sentences are no context for it

    def _paced(self, reference_wav_path: str, speed: float) -> str:
        """The reference sped up by `speed`, pitch kept (made once, kept beside the
        English references): Drama 3 imitates it at Qwen3's pace."""
        if abs(speed - 1.0) < 0.005:
            return reference_wav_path
        import hashlib

        import soundfile

        from sts.audio.tempo import time_stretch

        source = Path(reference_wav_path).read_bytes()
        key = hashlib.sha1(source + f"{speed:.3f}".encode()).hexdigest()[:16]
        path = Path(self.cfg.voice_library_dir) / ".drama3-references" / f"{key}-x{speed:.2f}.wav"
        if not path.is_file():
            audio, sample_rate = soundfile.read(reference_wav_path, dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            path.parent.mkdir(parents=True, exist_ok=True)
            soundfile.write(str(path), time_stretch(audio, sample_rate, speed), sample_rate)
        vlog("tts", f"Drama 3: reference at {speed:.2f}x its pace")
        return str(path)

    def make_english_reference(self, reference_wav_path: str) -> str:
        raise NotImplementedError("English references are made by Qwen3 (a helper process)")

    def memory_bytes(self) -> int:
        return self.model.weight_bytes()

    # -- generating --------------------------------------------------------------

    def _can_stream(self) -> bool:
        return bool(self.cfg.qwen3_stream)

    def _sampling(self, text: str) -> dict:
        """Drama 3's own sampling recipe (temperature 0.7, top-p 0.7, top-k 30), and
        a length cap in its frames, as Qwen3's (see Qwen3TTSSynthesizer._sampling)."""
        seconds = max(MIN_CAP_SECONDS, len(text) * self.cfg.qwen3_max_seconds_per_char)
        return {"max_tokens": int(seconds * self.model.frames_per_second)}

    def _stream_generate(self, text: str, attempt: int = 0):
        audio, ref_text = self._reference
        from sts.tts.qwen3 import STREAM_PIECE_S

        sampling = self._sampling(text)
        if attempt:
            # A retry (the last try stopped talking partway): broader sampling. On a
            # sentence that stalled in 4 of 10 takes with the recipe, top-p 0.9 /
            # top-k 60 stalled in 1 of 10 (temperature 1.0 too, but it changes the
            # delivery more).
            sampling.update(top_p=0.9, top_k=60)
        return self.model.stream(text, audio, ref_text, STREAM_PIECE_S, history=list(self._history), **sampling)

    def _spoken_whole(self, text: str) -> None:
        self._history.extend(self.model.last_turns)

    def _generate_once(self, text: str) -> tuple[np.ndarray, int]:
        audio, ref_text = self._reference
        with self._lock:
            return (self.model.generate(text, audio, ref_text, history=list(self._history), **self._sampling(text)),
                    self.model.sample_rate)

    # -- not in Drama 3 ------------------------------------------------------------

    def design_reference(self, instruct: str, ref_text: str):
        raise NotImplementedError("voice design is Qwen3's")

    def unload_designer(self) -> None:
        pass

    def preview(self, text: str, reference_wav_path: str, ref_text: str):
        raise NotImplementedError("voice design is Qwen3's")


def drama3_installed() -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec("mlx_audio.tts.models.fish_drama3") is not None
    except (ImportError, ValueError):
        return False
