"""Diagnostic tool: streams an input device straight into Parakeet with no
VAD segmentation and none of the pipeline's commit/confirm logic on top -
just the model's own raw output, printed to stdout exactly as it comes back.

Unlike the live pipeline (which cuts speech into segments on silence, so each
one can be transcribed and handed off independently), this never cuts
anything: captured audio piles into one buffer for the whole run, and that
whole buffer is re-transcribed from scratch every --interval-ms. Parakeet
sees more context each time, and can (and does) revise earlier wording as a
result - showing the *complete* current transcript every cycle, not just
whatever's new, is what makes that revision visible instead of hidden behind
a diff. In a terminal it's one line rewritten in place as it grows and
changes; piped elsewhere, each changed transcript is its own line. The
buffer only grows, so later re-transcriptions cost more than
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
import shutil
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


class LiveLine:
    """Shows the transcript as one line that's rewritten in place as it
    grows or is revised, instead of printing a new line for every
    re-transcription. A transcript longer than the terminal is wide wraps
    onto several rows, so a rewrite moves the cursor back up over all of
    them and clears to the end of the screen before printing the new text.
    When stdout isn't a terminal (piped to a file) there's no cursor to
    move: every changed transcript is printed as its own line instead."""

    def __init__(self, stream=None):
        self.stream = stream or sys.stdout
        self.live = self.stream.isatty()
        self.text = ""
        self._rows = 0  # rows the current text occupies on screen

    def update(self, text: str) -> None:
        if text == self.text:
            return
        self.text = text
        if not self.live:
            print(text, file=self.stream, flush=True)
            return
        width = max(1, shutil.get_terminal_size().columns)
        up = f"\x1b[{self._rows - 1}A" if self._rows > 1 else ""
        self.stream.write(f"\r{up}\x1b[J{text}")
        self.stream.flush()
        self._rows = max(1, -(-len(text) // width))

    def finish(self) -> None:
        """Leaves the current text on screen and moves below it, so whatever
        is printed next starts on a fresh line."""
        if self.live and self._rows:
            self.stream.write("\n")
            self.stream.flush()
        self.text = ""
        self._rows = 0


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

    if args.asr_device.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            sys.exit("[caption] --asr-device cuda, but PyTorch sees no CUDA GPU here; pass --asr-device cpu to run anyway")
    print(f"[caption] loading Parakeet on {args.asr_device}…", file=sys.stderr, flush=True)
    from sts.asr.parakeet_asr import ParakeetASR  # heavy (NeMo + lightning); deferred past --list-devices

    asr = ParakeetASR(ASRConfig(device=args.asr_device))
    param = next(asr.model.parameters())
    print(f"[caption] Parakeet loaded on {param.device} ({param.dtype})", file=sys.stderr, flush=True)

    capture = AudioCapture(AudioConfig(input_device=args.input_device))
    capture.start()
    print(
        f"[caption] listening on input device {args.input_device!r}; x to clear, Ctrl+C to stop",
        file=sys.stderr,
        flush=True,
    )

    clear = threading.Event()
    restore_terminal = watch_keys(lambda key: clear.set() if key.lower() == "x" else None)

    line = LiveLine()
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
                line.finish()  # the cleared text stays on screen, above the new one
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
                    line.update(text)
    except KeyboardInterrupt:
        pass
    finally:
        line.finish()
        restore_terminal()
        capture.stop()


if __name__ == "__main__":
    main()
