"""Where sts keeps what it saves, whichever way it's started."""

import sys
from pathlib import Path


def app_support_dir() -> Path | None:
    """~/Library/Application Support/Live Translate on macOS (where the
    Live Translate app keeps its files), None elsewhere."""
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Live Translate"
    return None


def default_config_db() -> str:
    """The remembered settings (devices, volume, voice, ...): on macOS the app's
    sts.db, so the terminal and the app share them; elsewhere sts.db in the
    working directory."""
    support = app_support_dir()
    return str(support / "sts.db") if support else "sts.db"


def default_voice_library_dir() -> str:
    """The voice library: on macOS the one the Live Translate app uses
    (~/Library/Application Support/Live Translate/voices), so voices made from the
    terminal (sts, sts-add-voices) and in the app are the same ones;
    elsewhere `voices` in the working directory."""
    support = app_support_dir()
    return str(support / "voices") if support else "voices"
