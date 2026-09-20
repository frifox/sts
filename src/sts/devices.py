"""Maps the CLI's top-level --device choice to per-component devices.

Pulled out of `sts.cli` because that module imports the full pipeline stack
(sounddevice, faster-whisper, torch, transformers, ...) at import time, none
of which needs to be installed just to test this mapping logic.
"""

import os


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
