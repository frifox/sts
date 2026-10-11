"""Where sts keeps what it saves, whichever way it's started."""

import sys
from pathlib import Path


def default_voice_library_dir() -> str:
    """The voice library: on macOS the one STS.app uses
    (~/Library/Application Support/STS/voices), so voices made from the
    terminal (sts, sts-add-voices) and in the app are the same ones;
    elsewhere `voices` in the working directory."""
    if sys.platform == "darwin":
        return str(Path.home() / "Library" / "Application Support" / "STS" / "voices")
    return "voices"
