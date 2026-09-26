"""How far start-up has got, as a percentage, from how long each step took
the last few times (the caption page shows it next to the stage text,
e.g. "Loading translation model… 11%").

Each step's duration is recorded as start-up runs, and when it's done
("Listening") the timings are saved - the last KEEP_RUNS of them - in the
persistent settings. The next start-up expects each step to take its
average over those runs, and the whole thing their average total. With
nothing recorded yet (the first start-up) there's no percentage.
"""

import json
import time

KEEP_RUNS = 3
SETTINGS_KEY = "startup_timings"


class StartupProgress:
    def __init__(self, persistent_config=None):
        self._settings = persistent_config
        runs = []
        if persistent_config is not None:
            try:
                runs = json.loads(persistent_config.get(SETTINGS_KEY) or "[]")[-KEEP_RUNS:]
            except (ValueError, TypeError):
                runs = []
        self._runs = [run for run in runs if isinstance(run, dict) and run.get("total", 0) > 0]
        self._total_s = sum(run["total"] for run in self._runs) / len(self._runs) if self._runs else 0.0
        self._started = time.monotonic()
        self._timings: dict[str, float] = {}  # this start-up's finished steps
        self._step: str | None = None
        self._step_started = self._started
        self._expected_done_s = 0.0  # expected duration of the steps finished so far
        self._progress: dict | None = None
        self.finished = False

    def expected_s(self, step: str) -> float:
        """The step's average duration over the recorded runs that had it."""
        seen = [run["steps"][step] for run in self._runs if step in run.get("steps", {})]
        return sum(seen) / len(seen) if seen else 0.0

    def begin(self, step: str) -> None:
        """The previous step (if any) is done; `step` starts now."""
        if self.finished:
            return
        now = time.monotonic()
        self._end_step(now)
        self._step, self._step_started = step, now
        if self._total_s <= 0:
            self._progress = None
            return
        expected = self.expected_s(step)
        start = min(1.0, self._expected_done_s / self._total_s)
        self._progress = {
            "start": round(start, 4),
            "end": round(min(1.0, (self._expected_done_s + expected) / self._total_s), 4),
            "step_s": round(expected, 2),
        }

    def progress(self) -> dict | None:
        """For the page: the step's place on the whole start-up (fractions
        `start`-`end`), how long it's expected to take (`step_s`), and how
        long it's been going (`elapsed_s`). None with nothing to go on, or
        once start-up is done."""
        if self._progress is None or self.finished:
            return None
        return {**self._progress, "elapsed_s": round(time.monotonic() - self._step_started, 2)}

    def finish(self) -> None:
        """Start-up is done: saves this run's timings for the next one."""
        if self.finished:
            return
        now = time.monotonic()
        self._end_step(now)
        self.finished = True
        if self._settings is None:
            return
        run = {"steps": self._timings, "total": round(now - self._started, 2)}
        self._settings.set(SETTINGS_KEY, json.dumps((self._runs + [run])[-KEEP_RUNS:], ensure_ascii=False))

    def _end_step(self, now: float) -> None:
        if self._step is None:
            return
        self._timings[self._step] = round(self._timings.get(self._step, 0.0) + now - self._step_started, 2)
        self._expected_done_s += self.expected_s(self._step)
        self._step = None
