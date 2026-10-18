from sts.config import TTSConfig


def _missing_backend(backend: str, exc: ImportError) -> str:
    install = ("pip install -e '.[xtts]'" if backend == "xtts"
               else "pip install -e '.[qwen3]'  (PyTorch), or  pip install -e '.[mlx]'  (MLX, Apple Silicon)")
    return (
        f"The '{backend}' TTS backend isn't installed in this environment ({exc}).\n"
        f"  Install it with:  {install}\n"
        f"  The TTS extras need conflicting transformers versions, so each has a virtualenv of its\n"
        f"  own: .venv-xtts (xtts), .venv-qwen (qwen3), .venv-mlx (mlx).\n"
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
    if cfg.backend == "drama3":
        try:
            from sts.tts.drama3 import Drama3Synthesizer
        except ImportError as exc:
            raise SystemExit(_missing_backend("qwen3", exc)) from exc
        return Drama3Synthesizer(cfg)
    raise ValueError(f"unknown TTS backend: {cfg.backend!r} (expected 'xtts', 'qwen3' or 'drama3')")
