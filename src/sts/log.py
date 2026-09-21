"""Optional verbose logging (`sts -v`). Quiet by default: only the
recognized/translated text and lifecycle lines are printed unconditionally."""

import time

_verbose = False
_start = time.monotonic()


def set_verbose(enabled: bool) -> None:
    global _verbose
    _verbose = enabled


def is_verbose() -> bool:
    return _verbose


def vlog(tag: str, message: str) -> None:
    if _verbose:
        print(f"{time.monotonic() - _start:7.2f}s [{tag}] {message}", flush=True)
