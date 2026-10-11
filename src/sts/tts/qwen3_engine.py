"""Which implementation runs the Qwen3-TTS backend: "torch" (the `qwen-tts`
package, PyTorch - `.venv-qwen`) or "mlx" (`mlx-audio`, Apple's MLX -
`.venv-mlx`, Apple Silicon only, about 2.5x faster there). They need
conflicting transformers versions (qwen-tts pins 4.57.3, mlx-audio needs 5),
so an environment normally has just one; "auto" picks that one, and torch if
both are there. Kept free of heavy imports: the CLI asks before loading anything."""

import importlib.util

ENGINES = ("torch", "mlx")
_MODULES = {"torch": "qwen_tts", "mlx": "mlx_audio"}


def available_engines() -> list[str]:
    return [engine for engine in ENGINES if _installed(_MODULES[engine])]


def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ValueError:  # already imported, without import metadata (a stand-in module): it's there
        return True


def resolve_engine(requested: str = "auto") -> str:
    if requested in ENGINES:
        return requested
    available = available_engines()
    return available[0] if available else "torch"
