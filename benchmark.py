#!/usr/bin/env python3
"""Benchmark ASR/translation/TTS speed on this machine: python benchmark.py

Thin wrapper so the script can be run from a checkout; the logic is in
sts/benchmark.py (also installed as the `sts-benchmark` command)."""

from sts.benchmark import main

if __name__ == "__main__":
    main()
