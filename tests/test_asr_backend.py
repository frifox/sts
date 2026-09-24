import threading

import pytest

from sts.asr import create_asr
from sts.config import ASRConfig, PipelineConfig


def test_create_asr_rejects_unknown_backend():
    with pytest.raises(ValueError, match="unknown ASR backend"):
        create_asr(ASRConfig(backend="bogus"))


def test_create_asr_dispatches_to_parakeet(monkeypatch):
    calls = []
    monkeypatch.setattr("sts.asr.parakeet_asr.ParakeetASR", lambda cfg: calls.append(cfg) or "parakeet-instance")
    assert create_asr(ASRConfig(backend="parakeet")) == "parakeet-instance"
    assert len(calls) == 1


def test_create_asr_parakeet_missing_dependency_gives_install_hint(monkeypatch):
    def raise_import_error(cfg):
        raise ImportError("No module named 'nemo'")

    monkeypatch.setattr("sts.asr.parakeet_asr.ParakeetASR", raise_import_error)
    with pytest.raises(SystemExit, match=r"pip install -e '.\[parakeet\]'"):
        create_asr(ASRConfig(backend="parakeet"))


def _switch_pipeline(monkeypatch, asr_factory):
    from sts.pipeline import Pipeline

    monkeypatch.setattr("sts.pipeline.create_asr", asr_factory)
    # Keeps _load_asr's free-memory check off whatever GPU the test machine has.
    monkeypatch.setattr("sts.pipeline.torch.cuda.is_available", lambda: False)

    class Broadcaster:
        engines = None
        status = None

        def broadcast_engines(self, info):
            self.engines = info

        def broadcast_status(self, **kwargs):
            self.status = kwargs

        def broadcast_stage(self, text, ready=False):
            pass

    class Tts:
        cloned_speaker_wav = None

    p = object.__new__(Pipeline)
    p.cfg = PipelineConfig()
    p.broadcaster = Broadcaster()
    p.tts = Tts()
    p._pending_voice = None
    p._active_speaker_id = None
    p._asr_lock = threading.Lock()
    p._asr_busy = False
    p._persistent_config = None
    from sts.control import RecognitionControl

    p.control = RecognitionControl(16000)
    return p


def test_switch_asr_backend_swaps_the_engine(monkeypatch):
    built = []

    def fake_create_asr(cfg):
        built.append(cfg.backend)
        return f"{cfg.backend}-engine"

    p = _switch_pipeline(monkeypatch, fake_create_asr)
    p.asr = "gigaam-engine"
    p.cfg.asr.backend = "gigaam"

    p._switch_asr_backend("parakeet")

    assert p.asr == "parakeet-engine"
    assert p.cfg.asr.backend == "parakeet"
    assert not p._asr_busy
    assert p.broadcaster.engines["asr"]["backend"] == "parakeet"


def test_switch_asr_backend_is_a_no_op_for_the_same_backend(monkeypatch):
    calls = []
    p = _switch_pipeline(monkeypatch, lambda cfg: calls.append(cfg) or "engine")
    p.asr = "gigaam-engine"
    p.cfg.asr.backend = "gigaam"
    p._asr_busy = True  # set by handle_control before dispatching, as it would be for real

    p._switch_asr_backend("gigaam")

    assert calls == []  # never even tried to build a new one
    assert p.asr == "gigaam-engine"  # untouched
    assert not p._asr_busy


def test_switch_asr_backend_falls_back_when_the_new_backend_is_not_installed(monkeypatch):
    """create_asr() raises SystemExit (not Exception) for a missing optional
    backend - appropriate for a hard CLI-startup failure, but _switch_asr_backend
    must still treat it as recoverable for a live webui toggle rather than
    letting it escape and leave self.asr deleted with nothing to replace it."""

    def create_asr_missing_parakeet(cfg):
        if cfg.backend == "parakeet":
            raise SystemExit("the 'parakeet' ASR backend isn't installed in this environment")
        return "gigaam-engine"

    p = _switch_pipeline(monkeypatch, create_asr_missing_parakeet)
    p.asr = "gigaam-engine"
    p.cfg.asr.backend = "gigaam"

    p._switch_asr_backend("parakeet")

    assert p.cfg.asr.backend == "gigaam"
    assert p.asr == "gigaam-engine"
    assert not p._asr_busy


def test_switch_asr_backend_falls_back_to_the_previous_engine_on_failure(monkeypatch):
    def flaky_create_asr(cfg):
        if cfg.backend == "parakeet":
            raise RuntimeError("out of memory")
        return "gigaam-engine"

    p = _switch_pipeline(monkeypatch, flaky_create_asr)
    p.asr = "gigaam-engine"
    p.cfg.asr.backend = "gigaam"

    p._switch_asr_backend("parakeet")

    # Failed to switch - back on the working engine, not left broken.
    assert p.cfg.asr.backend == "gigaam"
    assert p.asr == "gigaam-engine"
    assert not p._asr_busy
    assert p.broadcaster.engines["asr"]["backend"] == "gigaam"


def test_switch_asr_backend_survives_the_fallback_itself_failing(monkeypatch):
    """Reproduces what actually happened testing this live: switching to
    Parakeet OOM'd, and reloading the previous engine right afterward (before the failed
    attempt's GPU memory had really cleared) OOM'd too. self.asr must end up
    *something* callable, not deleted-and-never-reassigned, or _asr_loop's
    thread dies on the next segment with an AttributeError."""

    def always_fails(cfg):
        raise RuntimeError("CUDA out of memory")

    p = _switch_pipeline(monkeypatch, always_fails)
    p.asr = "gigaam-engine"
    p.cfg.asr.backend = "gigaam"

    p._switch_asr_backend("parakeet")

    assert p.asr.transcribe(None) == ""  # the _NullASR stand-in, not gone entirely
    assert p.cfg.asr.backend == "gigaam"  # reports the engine it *meant* to be on
    assert not p._asr_busy


class _AsrPresenceLock:
    """A real lock that records every release that happens while p.asr is
    missing - i.e. every moment _asr_loop could grab the lock and crash on
    a deleted self.asr."""

    def __init__(self, pipeline):
        self._lock = threading.Lock()
        self._pipeline = pipeline
        self.released_without_asr = 0

    def __enter__(self):
        self._lock.acquire()
        return self

    def __exit__(self, *exc):
        if not hasattr(self._pipeline, "asr"):
            self.released_without_asr += 1
        self._lock.release()
        return False


def test_failed_switch_never_releases_the_lock_while_self_asr_is_missing(monkeypatch):
    """Real crash from live use: switching to parakeet in an environment
    without nemo. _load_asr deletes self.asr *before* create_asr raises, and
    the fallback reload used to happen in a *separate* `with _asr_lock` block
    - so the lock was free in between, _asr_loop (blocked waiting on it)
    grabbed it, found self.asr gone, and its thread died with an
    AttributeError, killing captions until restart. The lock has to be held
    across the attempt *and* the fallback."""

    def create_asr_missing_parakeet(cfg):
        if cfg.backend == "parakeet":
            raise SystemExit("the 'parakeet' ASR backend isn't installed in this environment")
        return "gigaam-engine"

    p = _switch_pipeline(monkeypatch, create_asr_missing_parakeet)
    p.asr = "gigaam-engine"
    p.cfg.asr.backend = "gigaam"
    p._asr_lock = _AsrPresenceLock(p)

    p._switch_asr_backend("parakeet")

    assert p._asr_lock.released_without_asr == 0
    assert p.asr == "gigaam-engine"  # and it did fall back correctly
    assert p.cfg.asr.backend == "gigaam"


def test_failed_switch_and_failed_fallback_never_releases_the_lock_without_an_engine(monkeypatch):
    def always_fails(cfg):
        raise RuntimeError("CUDA out of memory")

    p = _switch_pipeline(monkeypatch, always_fails)
    p.asr = "gigaam-engine"
    p.cfg.asr.backend = "gigaam"
    p._asr_lock = _AsrPresenceLock(p)

    p._switch_asr_backend("parakeet")

    assert p._asr_lock.released_without_asr == 0
    assert p.asr.transcribe(None) == ""  # ended on the _NullASR stand-in, not missing


def test_switch_is_refused_up_front_when_the_gpu_clearly_lacks_room(monkeypatch):
    """The real failure: a switch to Parakeet ran out of GPU memory halfway
    through loading, which left too little for the fallback to the previous engine too.
    Now a switch that can't fit is refused before anything is built, and the
    previous engine comes back (its reload isn't held to the estimate)."""
    built = []

    def fake_create_asr(cfg):
        built.append(cfg.backend)
        return f"{cfg.backend}-engine"

    p = _switch_pipeline(monkeypatch, fake_create_asr)
    monkeypatch.setattr("sts.pipeline.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("sts.pipeline.torch.cuda.mem_get_info", lambda device: (200 * 2**20, 8 * 2**30))
    p.asr = "gigaam-engine"
    p.cfg.asr.backend = "gigaam"

    p._switch_asr_backend("parakeet")

    assert built == ["gigaam"]  # parakeet was never attempted
    assert p.asr == "gigaam-engine"
    assert p.cfg.asr.backend == "gigaam"


def test_switching_releases_the_old_engine_before_building_the_new_one(monkeypatch):
    events = []

    class OldEngine:
        def close(self):
            events.append("closed old")

    def fake_create_asr(cfg):
        events.append(f"built {cfg.backend}")
        return f"{cfg.backend}-engine"

    p = _switch_pipeline(monkeypatch, fake_create_asr)
    p.asr = OldEngine()
    p.cfg.asr.backend = "gigaam"

    p._switch_asr_backend("parakeet")

    assert events == ["closed old", "built parakeet"]


def test_estimate_vram_bytes_covers_each_backend():
    from sts.asr import estimate_vram_bytes

    parakeet = estimate_vram_bytes(ASRConfig(backend="parakeet"))
    gigaam = estimate_vram_bytes(ASRConfig(backend="gigaam"))
    assert 1 * 2**30 < parakeet < 2 * 2**30  # measured: ~1.2 GB loaded
    assert 0.8 * 2**30 < gigaam < 2 * 2**30  # measured: ~0.83 GB loaded
