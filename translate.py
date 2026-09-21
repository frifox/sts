#!/usr/bin/env python3
"""Translate a video's Russian speech to English: python translate.py --input talk.mp4

Thin wrapper so the script can be run from a checkout; the logic is in sts/offline.py
(also installed as the `sts-translate` command)."""

from sts.offline import main

if __name__ == "__main__":
    main()
