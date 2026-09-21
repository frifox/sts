from sts.config import TTSConfig


def _missing_backend(backend: str, exc: ImportError) -> str:
    other = "qwen3" if backend == "xtts" else "xtts"
    return (
        f"The '{backend}' TTS backend isn't installed in this environment ({exc}).\n"
        f"  Install it with:  pip install -e '.[{backend}]'\n"
        f"  The xtts and qwen3 extras need conflicting transformers versions, so use a separate\n"
        f"  virtualenv per backend (e.g. .venv for xtts, .venv-qwen for qwen3) rather than adding\n"
        f"  it on top of an environment that already has '{other}'."
    )


def create_synthesizer(cfg: TTSConfig):
    if cfg.backend == "xtts":
        try:
            from sts.tts.xtts import XTTSSynthesizer
        except ImportError as exc:
            raise SystemExit(_missing_backend("xtts", exc)) from exc
        return XTTSSynthesizer(cfg)
    if cfg.backend == "qwen3":
        try:
            from sts.tts.qwen3 import Qwen3TTSSynthesizer
        except ImportError as exc:
            raise SystemExit(_missing_backend("qwen3", exc)) from exc
        return Qwen3TTSSynthesizer(cfg)
    raise ValueError(f"unknown TTS backend: {cfg.backend!r} (expected 'xtts' or 'qwen3')")
