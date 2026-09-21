import threading
import time

from sts.gpu import gpu_lock


def test_mps_calls_never_overlap():
    active = 0
    overlaps = 0
    guard = threading.Lock()

    def work():
        nonlocal active, overlaps
        with gpu_lock("mps"):
            with guard:
                active += 1
                overlaps += active > 1
            time.sleep(0.01)
            with guard:
                active -= 1

    threads = [threading.Thread(target=work) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert overlaps == 0


def test_lock_is_reentrant_and_cpu_is_free():
    with gpu_lock("mps"):
        with gpu_lock("mps"):
            pass
    with gpu_lock("cpu"), gpu_lock("cpu"):
        pass
