"""Model downloads in progress, for the caption page's status text.

Every model sts loads comes from the Hugging Face hub, in whichever process
loads it (translation and TTS each run in their own, see sts.worker), and a
file being downloaded sits in the hub cache as `<blob>.incomplete` until it's
done. So downloads are watched there, from outside: a growing .incomplete
file is a download in progress. (One left behind by an interrupted download
stops growing, and is ignored once it hasn't for ACTIVE_S.)
"""

import os
import threading
import time
from pathlib import Path

ACTIVE_S = 10.0  # an .incomplete file not written for this long isn't being downloaded
POLL_S = 1.0
RATE_WINDOW_S = 5.0


def hub_cache_dir() -> Path:
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        return Path(HF_HUB_CACHE)
    except ImportError:
        home = os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface"
        return Path(home) / "hub"


def _repo_name(folder: str) -> str:
    """models--mlx-community--fish-audio-drama-3-4bit -> fish-audio-drama-3-4bit"""
    return folder.split("--")[-1] if folder.startswith(("models--", "datasets--")) else folder


def _size(nbytes: float) -> str:
    return f"{nbytes / 1e9:.1f} GB" if nbytes >= 1e9 else f"{nbytes / 1e6:.0f} MB"


class DownloadWatcher:
    """Calls `on_change(text)` with what's downloading ("fish-audio-drama-3-4bit:
    4.2 GB, 45 MB/s"), or None once nothing is, whenever that changes."""

    def __init__(self, on_change, cache_dir: Path | None = None):
        self._on_change = on_change
        self._cache_dir = Path(cache_dir) if cache_dir else hub_cache_dir()
        self._sizes: dict[Path, int] = {}  # this download's files: path -> bytes so far
        self._samples: list[tuple[float, int]] = []  # (time, bytes so far), for the rate
        self._text: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def text(self) -> str | None:
        return self._text

    def start(self) -> "DownloadWatcher":
        self._thread = threading.Thread(target=self._run, name="download-watcher", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(POLL_S):
            try:
                self.poll()
            except Exception:  # noqa: BLE001 - a status nicety: never take sts down
                pass

    def poll(self, now: float | None = None) -> str | None:
        now = time.time() if now is None else now
        active: dict[Path, int] = {}
        repos: list[str] = []
        try:
            folders = list(os.scandir(self._cache_dir))
        except OSError:
            folders = []
        for folder in folders:
            try:
                blobs = os.scandir(Path(folder.path) / "blobs")
            except OSError:
                continue
            with blobs:
                for blob in blobs:
                    if not blob.name.endswith(".incomplete"):
                        continue
                    try:
                        stat = blob.stat()
                    except OSError:
                        continue
                    if now - stat.st_mtime <= ACTIVE_S:
                        active[Path(blob.path)] = stat.st_size
                        if _repo_name(folder.name) not in repos:
                            repos.append(_repo_name(folder.name))
        if not active:
            self._sizes.clear()
            self._samples.clear()
            text = None
        else:
            self._sizes.update(active)  # files finished meanwhile keep their last size
            total = sum(self._sizes.values())
            self._samples = [s for s in self._samples if now - s[0] <= RATE_WINDOW_S] + [(now, total)]
            first_t, first_total = self._samples[0]
            rate = (total - first_total) / (now - first_t) if now > first_t else 0.0
            text = f"{', '.join(repos)}: {_size(total)}" + (f", {_size(rate)}/s" if rate > 0 else "")
        if text != self._text:
            self._text = text
            self._on_change(text)
        return text
