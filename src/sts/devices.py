"""Maps the CLI's top-level --device choice to per-component devices, and
decides which CUDA GPUs are new enough for bfloat16/flash-attention.

Pulled out of `sts.cli`/`sts.tts.qwen3` because those modules import the
full pipeline stack (sounddevice, faster-whisper, torch, transformers, ...)
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


def resolve_devices(device: str, asr_device_override: str | None) -> tuple[str, str, str]:
    """Returns (asr_device, translation_device, tts_device).

    faster-whisper's CTranslate2 backend has no Metal/MPS support (only
    cpu/cuda), so on Apple Silicon ASR runs on cpu (still fast there -
    CTranslate2's int8 path is well optimized for it) while translation and
    TTS use the GPU via mps.
    """
    if device == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        asr_device = asr_device_override or "cpu"
        if asr_device != "cpu":
            print(f"[device] using --asr-device {asr_device} instead of falling back to cpu for mps")
        else:
            print("[device] mps has no faster-whisper support - ASR runs on cpu, translation/TTS on mps")
        return asr_device, "mps", "mps"
    return (asr_device_override or device), device, device
