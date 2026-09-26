"""Start-up percentage: step timings recorded, averaged over the last runs."""

import json

import pytest

import sts.startup_progress as sp
from sts.startup_progress import StartupProgress


class _Settings(dict):
    def get(self, key, default=None):
        return super().get(key, default)

    def set(self, key, value):
        self[key] = value


@pytest.fixture
def clock(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(sp.time, "monotonic", lambda: now[0])
    return now


def _run(settings, clock, steps):
    """One start-up: each (step, seconds) in turn, then done."""
    startup = StartupProgress(settings)
    seen = []
    for step, seconds in steps:
        startup.begin(step)
        seen.append(startup.progress())
        clock[0] += seconds
    startup.finish()
    return seen


def test_the_first_start_up_has_no_percentage_but_is_recorded(clock):
    settings = _Settings()
    assert _run(settings, clock, [("A", 2.0), ("B", 6.0)]) == [None, None]
    assert json.loads(settings["startup_timings"]) == [{"steps": {"A": 2.0, "B": 6.0}, "total": 8.0}]


def test_the_next_start_up_places_each_step_by_its_average(clock):
    settings = _Settings()
    _run(settings, clock, [("A", 2.0), ("B", 6.0)])
    _run(settings, clock, [("A", 4.0), ("B", 8.0)])
    startup = StartupProgress(settings)  # averages: A 3 s, B 7 s, total 10 s
    startup.begin("A")
    assert startup.progress() == {"start": 0.0, "end": 0.3, "step_s": 3.0, "elapsed_s": 0.0}
    clock[0] += 5.0  # A ran long: B still starts where A's average ends
    startup.begin("B")
    assert startup.progress() == {"start": 0.3, "end": 1.0, "step_s": 7.0, "elapsed_s": 0.0}
    clock[0] += 1.5
    assert startup.progress()["elapsed_s"] == 1.5  # for a page connecting mid-step
    startup.finish()
    assert startup.progress() is None


def test_only_the_last_3_start_ups_are_kept(clock):
    settings = _Settings()
    for seconds in (1.0, 2.0, 3.0, 4.0):
        _run(settings, clock, [("A", seconds)])
    assert [run["total"] for run in json.loads(settings["startup_timings"])] == [2.0, 3.0, 4.0]
    startup = StartupProgress(settings)
    assert startup.expected_s("A") == pytest.approx(3.0)


def test_a_step_never_seen_before_takes_no_share(clock):
    settings = _Settings()
    _run(settings, clock, [("A", 4.0)])
    startup = StartupProgress(settings)
    startup.begin("New")
    assert startup.progress()["start"] == startup.progress()["end"] == 0.0


def test_without_settings_nothing_is_saved_or_shown(clock):
    startup = StartupProgress(None)
    startup.begin("A")
    assert startup.progress() is None
    startup.finish()
