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

Press "x" to clear: the buffer (and whatever audio was already captured
into it) is dropped, and recognition starts over from the next thing said -
e.g. to start fresh on a new speaker, or once the growing buffer has made
each re-transcription slow.

Usage:
  sts-caption --input-device 4
  sts-caption --input-device 4 --interval-ms 1000
  sts-caption --list-devices
"""

import argparse
import queue
import sys
import threading
import time
from collections.abc import Callable

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


def watch_keys(on_key: Callable[[str], None]) -> Callable[[], None]:
    """Calls `on_key(char)` for each key pressed in the terminal, without
    waiting for Enter, on a background thread. Returns a function that puts
    the terminal back how it was. Does nothing when stdin isn't a terminal
    (piped input, a service)."""
    if not sys.stdin.isatty():
        return lambda: None
    try:
        import msvcrt  # Windows

        def read() -> str:
            return msvcrt.getwch()

        def restore() -> None:
            pass
    except ImportError:
        import termios
        import tty

        fd = sys.stdin.fileno()
        saved = termios.tcgetattr(fd)
        # cbreak, not raw: keys arrive one at a time and aren't echoed, but
        # Ctrl+C still raises KeyboardInterrupt in the main thread.
        tty.setcbreak(fd)

        def read() -> str:
            return sys.stdin.read(1)

        def restore() -> None:
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)

    def loop() -> None:
        while True:
            try:
                key = read()
            except (OSError, ValueError):  # the terminal went away
                return
            if not key:
                return
            on_key(key)

    threading.Thread(target=loop, name="caption-keys", daemon=True).start()
    return restore


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
        f"[caption] listening on input device {args.input_device!r}; x to clear, Ctrl+C to stop",
        file=sys.stderr,
        flush=True,
    )

    clear = threading.Event()
    restore_terminal = watch_keys(lambda key: clear.set() if key.lower() == "x" else None)

    buffer: list[np.ndarray] = []
    last_run = time.monotonic()
    interval_s = args.interval_ms / 1000
    try:
        while True:
            if clear.is_set():
                clear.clear()
                buffer.clear()
                while True:  # audio captured before the key press belongs to what's being cleared
                    try:
                        capture.frames.get_nowait()
                    except queue.Empty:
                        break
                last_run = time.monotonic()
                print("[caption] cleared - listening for new speech", file=sys.stderr, flush=True)
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
        restore_terminal()
        capture.stop()


if __name__ == "__main__":
    main()
