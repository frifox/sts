from sts.config import ASRConfig


def _missing_backend(backend: str, exc: ImportError) -> str:
    return (
        f"The '{backend}' ASR backend isn't installed in this environment ({exc}).\n"
        f"  Install it with:  pip install -e '.[{backend}]'\n"
        f"  Unlike the TTS backends, whisper and parakeet don't conflict - nemo_toolkit installs "
        f"alongside either TTS extra's pinned transformers version fine, so this can go in the same "
        f"virtualenv you're already using."
    )


def create_asr(cfg: ASRConfig):
    if cfg.backend == "whisper":
        from sts.asr.whisper_asr import WhisperASR

        return WhisperASR(cfg)
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
    raise ValueError(f"unknown ASR backend: {cfg.backend!r} (expected 'whisper', 'parakeet' or 'gigaam')")


# Approximate parameter counts, for estimate_vram_bytes().
_WHISPER_PARAMS = {
    "tiny": 39e6, "base": 74e6, "small": 244e6, "medium": 769e6,
    "large-v3-turbo": 809e6, "turbo": 809e6, "distil-large-v3": 756e6,
}
_WHISPER_LARGE_PARAMS = 1550e6
_PARAKEET_PARAMS = 600e6
_GIGAAM_PARAMS = 240e6


def estimate_vram_bytes(cfg: ASRConfig) -> int:
    """Roughly how much GPU memory loading `cfg`'s engine takes: its weights
    at the compute type it runs in, plus headroom for activations/workspace.
    Only precise enough to refuse a load that clearly can't fit - see
    Pipeline._load_asr."""
    if cfg.backend == "parakeet":
        weights = _PARAKEET_PARAMS * 2  # fp16 on CUDA
    elif cfg.backend == "gigaam":
        weights = _GIGAAM_PARAMS * 4  # fp32 weights (it autocasts to fp16 itself)
    else:
        params = _WHISPER_PARAMS.get(cfg.model_size, _WHISPER_LARGE_PARAMS)  # large, large-v2, large-v3
        weights = params * (1 if cfg.compute_type.startswith("int8") else 2)
    return int(weights * 1.2 + 300 * 2**20)
