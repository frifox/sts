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
