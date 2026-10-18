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

    def stream(self, text: str, ref_audio: str, ref_text: str, interval_s: float, max_tokens: int):
        results = self._model.generate(
            text=text, ref_audio=ref_audio, ref_text=ref_text, max_tokens=max_tokens,
            stream=True, streaming_interval=interval_s, first_chunk_frames=DRAMA3_FIRST_CHUNK_FRAMES,
        )
        try:
            for result in results:
                audio = np.asarray(result.audio, dtype=np.float32).reshape(-1)
                if audio.size:
                    yield audio
        finally:
            close = getattr(results, "close", None)
            if close:
                close()

    def generate(self, text: str, ref_audio: str, ref_text: str, max_tokens: int) -> np.ndarray:
        pieces = [
            np.asarray(result.audio, dtype=np.float32).reshape(-1)
            for result in self._model.generate(text=text, ref_audio=ref_audio, ref_text=ref_text, max_tokens=max_tokens)
        ]
        return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)


class Drama3Synthesizer(Qwen3TTSSynthesizer):
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

    def _stream_generate(self, text: str):
        audio, ref_text = self._reference
        from sts.tts.qwen3 import STREAM_PIECE_S

        return self.model.stream(text, audio, ref_text, STREAM_PIECE_S, **self._sampling(text))

    def _generate_once(self, text: str) -> tuple[np.ndarray, int]:
        audio, ref_text = self._reference
        with self._lock:
            return self.model.generate(text, audio, ref_text, **self._sampling(text)), self.model.sample_rate

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
