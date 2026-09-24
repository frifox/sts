"""Diagnostic tool: streams an input device straight into Parakeet with no
VAD segmentation and none of the pipeline's commit/confirm logic on top -
just the model's own raw output, printed to stdout exactly as it comes back.

Unlike the live pipeline (which cuts speech into segments on silence, so each
one can be transcribed and handed off independently), this never cuts
anything: captured audio piles into one buffer for the whole run, and that
whole buffer is re-transcribed from scratch every --interval-ms. Parakeet
sees more context each time, and can (and does) revise earlier wording as a
result - printing the *complete* current transcript every cycle, not just
whatever's new, is what makes that revision visible instead of hidden behind
a diff. The buffer only grows, so later re-transcriptions cost more than
earlier ones - by design, since capping or windowing it would just be VAD
under another name. Meant for short exploratory runs, not left running.

Usage:
  sts-caption --input-device 4
  sts-caption --input-device 4 --interval-ms 1000
  sts-caption --list-devices
"""

import argparse
import queue
import sys
import time

import numpy as np

from sts.audio.capture import AudioCapture, list_devices
from sts.config import ASRConfig, AudioConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-device", type=int, default=None, help="ALSA input device index")
    parser.add_argument("--asr-device", default="cuda", help="cuda | cpu | mps (default: cuda)")
    parser.add_argument(
        "--interval-ms", type=int, default=500, help="How often to re-transcribe the whole buffer (default: 500)"
    )
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.list_devices:
        print(list_devices())
        return

    print(f"[caption] loading Parakeet on {args.asr_device}…", file=sys.stderr, flush=True)
    from sts.asr.parakeet_asr import ParakeetASR  # heavy (NeMo + lightning); deferred past --list-devices

    asr = ParakeetASR(ASRConfig(device=args.asr_device))

    capture = AudioCapture(AudioConfig(input_device=args.input_device))
    capture.start()
    print(
        f"[caption] listening on input device {args.input_device!r}; Ctrl+C to stop", file=sys.stderr, flush=True
    )

    buffer: list[np.ndarray] = []
    last_run = time.monotonic()
    interval_s = args.interval_ms / 1000
    try:
        while True:
            try:
                buffer.append(capture.frames.get(timeout=0.1))
            except queue.Empty:
                pass
            now = time.monotonic()
            if buffer and now - last_run >= interval_s:
                last_run = now
                audio = np.concatenate(buffer)
                text = asr.transcribe(audio)
                if text:
                    print(text, flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        capture.stop()


if __name__ == "__main__":
    main()
