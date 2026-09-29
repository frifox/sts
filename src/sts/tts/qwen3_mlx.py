"""Qwen3-TTS on Apple's MLX (the `mlx-audio` package, `.venv-mlx`), behind the
same interface as `qwen_tts.Qwen3TTSModel` - the part of it Qwen3TTSSynthesizer
uses - so everything around generation (the English reference for a non-English
voice, the length checks, tone and level matching) works the same either way.

Same models, converted to MLX (mlx-community/...-bf16): measured on 17 sermon
sentences on Apple Silicon, the 1.7B model generated at 0.25-0.29x real time,
against 0.71-0.72x with PyTorch on mps - the per-step overhead of PyTorch's
Metal backend, not the model's size, is what made it slow (the 0.6B model was
hardly faster there).

Not supported here: trained LoRA voices (a PyTorch/peft adapter); zero-shot
cloned voices, the preset voices and voice design are.
"""

from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 24_000


def mlx_model_name(name: str) -> str:
    """The MLX conversion of a Qwen3-TTS model: "Qwen/Qwen3-TTS-12Hz-1.7B-Base" ->
    "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16". Anything else (an MLX model
    named directly) is used as is."""
    if name.startswith("Qwen/Qwen3-TTS-"):
        return "mlx-community/" + name.split("/", 1)[1] + "-bf16"
    return name


@dataclass
class _ClonePrompt:
    ref_audio: object  # mx.array at the model's sample rate
    ref_text: str | None  # None: speaker embedding only (x-vector)


class MLXQwen3TTSModel:
    def __init__(self, model):
        self._model = model
        self.model = None  # (no torch module: see weight_bytes)

    @classmethod
    def from_pretrained(cls, name: str, **_torch_options) -> "MLXQwen3TTSModel":
        from mlx_audio.tts.utils import load_model

        return cls(load_model(mlx_model_name(name)))

    @property
    def sample_rate(self) -> int:
        return int(getattr(self._model, "sample_rate", SAMPLE_RATE))

    def weight_bytes(self) -> int:
        from mlx.utils import tree_flatten

        return sum(array.nbytes for _, array in tree_flatten(self._model.parameters()))

    def create_voice_clone_prompt(self, ref_audio, ref_text=None, x_vector_only_mode=False) -> _ClonePrompt:
        from mlx_audio.utils import load_audio

        audio = load_audio(ref_audio, sample_rate=self.sample_rate) if isinstance(ref_audio, str) else ref_audio
        return _ClonePrompt(audio, None if x_vector_only_mode else ref_text)

    def generate_voice_clone(self, text, language="English", voice_clone_prompt=None, non_streaming_mode=True, **sampling):
        # ICL (clip + transcript) always runs non-streaming text input in mlx-audio.
        prompt = voice_clone_prompt
        return self._collect(self._model.generate(
            text=text, ref_audio=prompt.ref_audio, ref_text=prompt.ref_text,
            lang_code=language.lower(), split_pattern="", **_mlx_sampling(sampling),
        ))

    def generate_custom_voice(self, text, speaker, language="English", instruct=None, **sampling):
        return self._collect(self._model.generate_custom_voice(
            text=text, speaker=speaker, language=language, instruct=instruct, **_mlx_sampling(sampling),
        ))

    def generate_voice_design(self, text, instruct, language="English", **sampling):
        return self._collect(self._model.generate_voice_design(
            text=text, instruct=instruct, language=language, **_mlx_sampling(sampling),
        ))

    def _collect(self, results) -> tuple[list[np.ndarray], int]:
        audio = [np.asarray(result.audio, dtype=np.float32) for result in results]
        return [np.concatenate(audio) if audio else np.zeros(0, dtype=np.float32)], self.sample_rate


def _mlx_sampling(sampling: dict) -> dict:
    """qwen_tts's sampling options, as mlx-audio names them (it has no separate
    settings for the sub-talker)."""
    out = {}
    if "temperature" in sampling:
        out["temperature"] = sampling["temperature"]
    if "top_k" in sampling:
        out["top_k"] = sampling["top_k"]
    if "max_new_tokens" in sampling:
        out["max_tokens"] = sampling["max_new_tokens"]
    return out


def clear_cache() -> None:
    import mlx.core as mx

    mx.clear_cache()
