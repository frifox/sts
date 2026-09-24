"""Diagnostic tool: streams an input device straight into a speech
recognizer with no VAD segmentation and none of the pipeline's
commit/confirm logic on top - just the model's own raw output, printed to
stdout exactly as it comes back.

The recognizer is Sber's GigaAM-v3 (e2e_rnnt, Russian-only, punctuated
output) by default - run it from .venv-gigaam, which has the newer torch/
transformers its model code needs (see sts.asr.gigaam_asr).
`--asr-backend parakeet` uses NVIDIA's Parakeet TDT instead, from a
virtualenv with NeMo (e.g. .venv-xtts).

Unlike the live pipeline (which cuts speech into segments on silence, so each
one can be transcribed and handed off independently), this never cuts
anything: captured audio piles into one buffer for the whole run, and that
whole buffer is re-transcribed every --interval-ms. The model sees more
context each time, and can (and does) revise earlier wording as a result -
showing the *complete* current transcript every cycle, not just whatever's
new, is what makes that revision visible instead of hidden behind a diff.
In a terminal it's one block, one sentence per line, rewritten in place as
it grows and changes; piped elsewhere, each changed transcript is printed in
full. With Parakeet the buffer is re-transcribed from scratch, so later
re-transcriptions cost more than earlier ones. GigaAM can only take ~25 s at
a time, so it cuts longer audio into ~20 s chunks at quiet moments; finished
chunks are transcribed once, then stay fixed, and only the last one keeps
changing. Meant for short exploratory runs, not left running.

Press "x" to clear: the buffer (and whatever audio was already captured
into it) is dropped, and recognition starts over from the next thing said -
e.g. to start fresh on a new speaker.

Usage:
  .venv-gigaam/bin/sts-caption --input-device 7
  .venv-gigaam/bin/sts-caption --input-device 7 --interval-ms 1000
  .venv-xtts/bin/sts-caption --input-device 7 --asr-backend parakeet
  sts-caption --list-devices
"""

import argparse
import contextlib
import io
import queue
import re
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
    parser.add_argument(
        "--asr-backend", choices=("gigaam", "parakeet"), default="gigaam",
        help="gigaam (GigaAM-v3 e2e_rnnt, Russian-only; run from .venv-gigaam) or parakeet (default: gigaam)",
    )
    parser.add_argument("--asr-device", default="cuda", help="cuda | cpu | mps (default: cuda)")
    parser.add_argument(
        "--interval-ms", type=int, default=500, help="How often to re-transcribe the whole buffer (default: 500)"
    )
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    return parser


# End of a sentence: ., ?, ! or … (possibly several, possibly followed by a
# closing quote/bracket), with more text after it.
_SENTENCE_END = re.compile(r'([.?!…]+["»”)]*)\s+(?=\S)')


def one_sentence_per_line(text: str) -> str:
    """`text` with a line break after each sentence-ending mark that has
    more text after it (none after the last - there's nothing to put on
    the next line yet)."""
    return _SENTENCE_END.sub("\\1\n", text)


def screen_rows(text: str, width: int) -> list[str]:
    """How `text` lays out on a terminal `width` columns wide: each of its
    lines, cut into rows of at most `width` characters."""
    rows = []
    for line in text.split("\n"):
        rows.extend([line[i : i + width] for i in range(0, len(line), width)] or [""])
    return rows


class LiveLine:
    """Shows the transcript as one block, one sentence per line, that's
    rewritten in place as it grows or is revised, instead of printing it all
    again for every re-transcription. The model can revise punctuation as well
    as words, so a sentence break can appear or disappear between two
    re-transcriptions - every update redraws the whole block (from where it
    starts on screen) rather than patching it.

    The one thing a terminal can't do is move the cursor above the top of
    the screen: once the block is taller than the window, its first rows
    have scrolled away and can no longer be rewritten. Those are taken as
    settled ("frozen"), and only the rows below them are redrawn - a
    revision to a sentence that has scrolled out of view isn't shown.

    Anything else printed meanwhile (a capture warning, say) has to go
    through print_above(), which puts it above the block and redraws the
    block under it - written straight to the terminal, it would land in the
    middle of the block and throw off where the next redraw starts (see
    main(), which routes all other output there).

    When stdout isn't a terminal (piped to a file) there's no cursor to
    move: every changed transcript is printed in full instead."""

    def __init__(self, stream=None):
        self.stream = stream or sys.stdout
        self.live = self.stream.isatty()
        self.text = ""
        self._frozen = 0  # leading rows of the block that can't be redrawn any more
        self._drawn = 0  # rows drawn after those, ending at the cursor's row
        self._lock = threading.Lock()  # other output can arrive from any thread

    def update(self, text: str) -> None:
        with self._lock:
            if text == self.text:
                return
            self.text = text
            if not self.live:
                print(one_sentence_per_line(text), file=self.stream, flush=True)
                return
            self._redraw(self._erase())

    def print_above(self, message: str) -> None:
        with self._lock:
            if not self.live or not (self._drawn or self._frozen):
                self.stream.write(message + "\n")
                self.stream.flush()
                return
            prefix = self._erase() + message + "\n"
            self._drawn = 0  # everything still redrawable goes below the message now
            self._redraw(prefix)

    def finish(self) -> None:
        """Leaves the current text on screen and moves below it, so whatever
        is printed next starts on a fresh line."""
        with self._lock:
            if self.live and (self._drawn or self._frozen):
                self.stream.write("\n")
                self.stream.flush()
            self.text = ""
            self._frozen = self._drawn = 0

    def _size(self) -> tuple[int, int]:
        size = shutil.get_terminal_size()
        return max(1, size.columns), max(2, size.lines)

    def _erase(self) -> str:
        """Escape codes that move the cursor to the first drawn row the
        screen still shows and clear from there down. Rows that have
        scrolled off meanwhile are frozen."""
        _, height = self._size()
        on_screen = min(self._drawn, height)
        self._frozen += self._drawn - on_screen
        self._drawn = on_screen
        up = f"\x1b[{on_screen - 1}A" if on_screen > 1 else ""
        return f"\r{up}\x1b[J"

    def _redraw(self, prefix: str) -> None:
        width, _ = self._size()
        rows = screen_rows(one_sentence_per_line(self.text), width)
        rows = rows[min(self._frozen, len(rows) - 1) :]
        self.stream.write(prefix + "\n".join(rows))
        self.stream.flush()
        self._drawn = len(rows)


class _AboveLiveLine(io.TextIOBase):
    """A stdout/stderr stand-in that prints each complete line above the
    live transcript (see LiveLine.print_above)."""

    def __init__(self, line: LiveLine):
        self._line = line
        self._pending = ""

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        self._pending += text
        *complete, self._pending = self._pending.split("\n")
        for message in complete:
            self._line.print_above(message)
        return len(text)


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
    name = {"gigaam": "GigaAM-v3 e2e_rnnt", "parakeet": "Parakeet"}[args.asr_backend]
    print(f"[caption] loading {name} on {args.asr_device}…", file=sys.stderr, flush=True)
    from sts.asr import create_asr  # heavy model imports; deferred past --list-devices

    asr = create_asr(ASRConfig(backend=args.asr_backend, device=args.asr_device))
    param = next(asr.model.parameters())
    print(f"[caption] {name} loaded on {param.device} ({param.dtype})", file=sys.stderr, flush=True)

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
    other_output = _AboveLiveLine(line)
    redirect = contextlib.ExitStack()
    try:
        redirect.enter_context(contextlib.redirect_stdout(other_output))
        redirect.enter_context(contextlib.redirect_stderr(other_output))
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
        redirect.close()
        line.finish()
        restore_terminal()
        capture.stop()


if __name__ == "__main__":
    main()
