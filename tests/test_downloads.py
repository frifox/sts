"""Downloads in progress, seen as growing .incomplete files in the hub cache."""

import os

from sts.downloads import ACTIVE_S, DownloadWatcher
from sts.startup_progress import StartupProgress


def blob(cache, repo, name, size, mtime):
    folder = cache / f"models--{repo.replace('/', '--')}" / "blobs"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_bytes(b"\0" * size)
    os.utime(path, (mtime, mtime))
    return path


def test_a_growing_download_is_reported_with_its_rate_then_cleared(tmp_path):
    seen = []
    watcher = DownloadWatcher(seen.append, tmp_path)
    path = blob(tmp_path, "mlx-community/fish-audio-drama-3-4bit", "abc.incomplete", 2_000_000, 1000.0)
    assert watcher.poll(now=1000.0) == "fish-audio-drama-3-4bit: 2 MB"
    path.write_bytes(b"\0" * 6_000_000)
    os.utime(path, (1002.0, 1002.0))
    assert watcher.poll(now=1002.0) == "fish-audio-drama-3-4bit: 6 MB, 2 MB/s"
    path.rename(path.with_suffix(""))  # done
    assert watcher.poll(now=1003.0) is None
    assert seen == ["fish-audio-drama-3-4bit: 2 MB", "fish-audio-drama-3-4bit: 6 MB, 2 MB/s", None]


def test_a_left_over_partial_download_is_not_a_download(tmp_path):
    seen = []
    blob(tmp_path, "google/madlad400-10b-mt", "old.incomplete", 1000, 1000.0)
    assert DownloadWatcher(seen.append, tmp_path).poll(now=1000.0 + ACTIVE_S + 1) is None
    assert seen == []


def test_no_percentage_while_downloading_and_the_timings_are_not_kept():
    class Settings(dict):
        def set(self, key, value):
            self[key] = value

    settings = Settings(startup_timings='[{"steps": {"a": 2.0}, "total": 2.0}]')
    startup = StartupProgress(settings)
    startup.begin("a")
    assert startup.progress() is not None
    startup.set_downloading(True)
    assert startup.progress() is None
    startup.set_downloading(False)
    assert startup.progress() is not None
    before = settings["startup_timings"]
    startup.finish()
    assert settings["startup_timings"] == before
