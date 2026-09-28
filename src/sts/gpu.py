"""Serializes GPU work on Apple Silicon.

PyTorch's MPS (Metal) backend aborts the whole process with
"failed assertion _status < MTLCommandBufferStatusCommitted" when two threads
submit GPU work at once. The pipeline naturally does exactly that: the
translation thread and the TTS thread both run models on the GPU. Every model
call on `mps` therefore takes this one shared lock. On other devices there is
nothing to share, so callers get their own private lock (or a no-op).
"""

import contextlib
import threading

_mps_lock = threading.RLock()


def gpu_lock(device: str):
    """Context manager guarding model calls on `device`."""
    return _mps_lock if device == "mps" else contextlib.nullcontext()


def model_bytes(*models) -> int:
    """Memory taken by the weights (parameters and buffers) of `models` -
    torch modules, or wrappers holding one as `.model` (Qwen3TTSModel) -
    each tensor counted once even if several models share it. Weights only:
    not the working memory a model uses while it runs."""
    import torch

    seen: set[tuple] = set()
    total = 0

    def visit(obj, depth: int = 0) -> None:
        nonlocal total
        if obj is None or depth > 3:
            return
        if isinstance(obj, torch.nn.Module):
            for tensor in [*obj.parameters(), *obj.buffers()]:
                if tensor.device.type == "meta":
                    continue
                key = (tensor.device.type, tensor.data_ptr(), tensor.numel())
                if key not in seen:
                    seen.add(key)
                    total += tensor.numel() * tensor.element_size()
            return
        visit(getattr(obj, "model", None), depth + 1)

    for model in models:
        visit(model)
    return total
