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
    raise ValueError(f"unknown ASR backend: {cfg.backend!r} (expected 'whisper' or 'parakeet')")
