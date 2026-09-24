"""Maps the CLI's top-level --device choice to per-component devices, and
decides which CUDA GPUs are new enough for bfloat16/flash-attention.

Pulled out of `sts.cli`/`sts.tts.qwen3` because those modules import the
full pipeline stack (sounddevice, NeMo, torch, transformers, ...)
at import time, none of which needs to be installed just to test this
mapping/decision logic.
"""

import os

# Flash-attention 2 and native bf16 tensor cores both need Ampere or newer.
# Pascal/Turing cards (e.g. a GTX 1080 Ti, compute capability 6.1) support
# neither properly - forcing either on them fails to load or runs badly.
MIN_COMPUTE_CAPABILITY_FOR_BF16_AND_FLASH_ATTENTION = 8


def supports_bf16_and_flash_attention(compute_capability_major: int) -> bool:
    return compute_capability_major >= MIN_COMPUTE_CAPABILITY_FOR_BF16_AND_FLASH_ATTENTION


def detect_device() -> str:
    """Best available torch device: cuda, then mps, else cpu."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_devices(
    device: str,
    asr_device_override: str | None,
    translation_device_override: str | None = None,
    tts_device_override: str | None = None,
) -> tuple[str, str, str]:
    """Returns (asr_device, translation_device, tts_device).

    ASR, translation, and TTS are three independently-loaded models, not one
    that would need splitting *across* GPUs - so multi-GPU here just means
    pinning whole stages to whichever card via these overrides (e.g.
    `--tts-device cuda:1`), each one still running entirely on a single
    device. No model parallelism, no data crossing cards except the small
    text/audio handed between stages already.

    On Apple Silicon (mps), PyTorch ops that Metal doesn't implement yet
    fall back to the CPU instead of crashing (PYTORCH_ENABLE_MPS_FALLBACK),
    at some speed cost for just those ops.
    """
    if device == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    return (
        asr_device_override or device,
        translation_device_override or device,
        tts_device_override or device,
    )
