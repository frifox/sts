from dataclasses import dataclass

from sts.config import ASRConfig


@dataclass
class TimedWord:
    """One recognized word (with any punctuation attached, e.g. "плода,")
    and when it was spoken, in seconds from the start of the audio given."""

    text: str
    start: float
    end: float


def _missing_backend(backend: str, exc: ImportError) -> str:
    return (
        f"The '{backend}' ASR backend isn't installed in this environment ({exc}).\n"
        f"  Install it with:  pip install -e '.[{backend}]'\n"
        f"  nemo_toolkit installs alongside either TTS extra's pinned transformers version fine, so "
        f"this can go in the same virtualenv you're already using."
    )


def create_asr(cfg: ASRConfig):
    if cfg.backend == "parakeet":
        from sts.asr.parakeet_asr import ParakeetASR  # module itself is always importable - see its docstring

        try:
            return ParakeetASR(cfg)
        except ImportError as exc:
            raise SystemExit(_missing_backend("parakeet", exc)) from exc
    if cfg.backend == "gigaam":
        try:
            from sts.asr.gigaam_asr import GigaAMASR

            return GigaAMASR(cfg)
        except ImportError as exc:
            raise SystemExit(
                f"The 'gigaam' ASR backend isn't installed in this environment ({exc}).\n"
                f"  It needs its own virtualenv (newer torch/transformers than the TTS backends allow), e.g.:\n"
                f"    uv venv --python 3.11 .venv-gigaam\n"
                f"    uv pip install --python .venv-gigaam/bin/python -e '.[gigaam]' "
                f"--extra-index-url https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match"
            ) from exc
    raise ValueError(f"unknown ASR backend: {cfg.backend!r} (expected 'parakeet' or 'gigaam')")


# Approximate parameter counts, for estimate_vram_bytes().
_PARAKEET_PARAMS = 600e6
_GIGAAM_PARAMS = 240e6


def estimate_vram_bytes(cfg: ASRConfig) -> int:
    """Roughly how much GPU memory loading `cfg`'s engine takes: its weights
    at the compute type it runs in, plus headroom for activations/workspace.
    Only precise enough to refuse a load that clearly can't fit - see
    Pipeline._load_asr."""
    if cfg.backend == "gigaam":
        weights = _GIGAAM_PARAMS * 4  # fp32 weights (it autocasts to fp16 itself)
    else:
        weights = _PARAKEET_PARAMS * 2  # fp16 on CUDA
    return int(weights * 1.2 + 300 * 2**20)
