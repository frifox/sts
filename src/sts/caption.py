"""Diagnostic tool: streams an input device straight into a speech
recognizer - no VAD segmentation, no translation - and shows what the model
hears, sentence by sentence, as it hears and revises it.

The recognizer is Sber's GigaAM-v3 (e2e_rnnt, Russian-only, punctuated
output), and the captioning (sentence locking) is the live pipeline's own,
sts.transcript - this is the pipeline's captions on their own, without the
web page, translation or voice.

Unlike the live pipeline (which cuts speech into segments on silence), this
never cuts on silence: the audio that isn't settled yet is re-transcribed
every --interval-ms, so the model sees more context each time and can revise
what it heard - which is shown as it happens, one sentence per line,
rewritten in place.

A sentence is final once it ends with a sentence-ending mark and hasn't
changed (text or punctuation) for --final-after-unchanged seconds (default
5), or --final-force-after seconds after its first word (default 30): its audio is
dropped, so it's never re-transcribed or reworded again, and it's printed
once, above a blank line. Piped to a file, only final sentences are printed
(plus whatever is still open on exit). GigaAM takes at most ~25 s
at a time, so a longer stretch that isn't locked yet is cut into ~20 s
chunks at quiet moments. Meant for exploratory runs.

Press "x" to clear: whatever isn't locked yet stays on screen as it is, the
audio behind it is dropped, and recognition starts over from the next thing
said - e.g. to start fresh on a new speaker.

Usage:
  .venv-xtts/bin/sts-caption --input-device 7
  .venv-xtts/bin/sts-caption --input-device 7 --interval-ms 1000
  .venv-xtts/bin/sts-caption --input-device 7 --final-after-unchanged 3 --final-force-after 20
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
from sts.transcript import Transcript


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-device", type=int, default=None, help="ALSA input device index")
    parser.add_argument("--asr-device", default="cuda", help="cuda | cpu | mps (default: cuda)")
    parser.add_argument(
        "--final-after-unchanged", type=float, default=5.0, metavar="S",
        help="Finalize a sentence once it ends with a sentence-ending mark and hasn't changed (text or "
        "punctuation) for S seconds (default: 5; 0 turns this rule off)",
    )
    parser.add_argument(
        "--final-force-after", type=float, default=30.0, metavar="S",
        help="Finalize a sentence S seconds after its first word, even if it's still changing "
        "(default: 30; 0 turns this rule off)",
    )
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

    Text that has become final is passed to update() as `above` and is
    printed above the block the same way, so final text accumulates above
    it, and the block only ever holds what may still change.

    When stdout isn't a terminal (piped to a file) there's no cursor to
    move: only final text is printed, plus whatever is still open when
    finish() is called."""

    def __init__(self, stream=None):
        self.stream = stream or sys.stdout
        self.live = self.stream.isatty()
        self.text = ""
        self._frozen = 0  # leading rows of the block that can't be redrawn any more
        self._drawn = 0  # rows drawn after those, ending at the cursor's row
        self._lock = threading.Lock()  # other output can arrive from any thread

    def update(self, text: str, above: list[str] = ()) -> None:
        """Shows `text` as the block, after printing `above` (text that just
        became final) above it - in one redraw, so the block is never drawn
        still holding what has just moved above it."""
        with self._lock:
            if text == self.text and not above:
                return
            self.text = text
            if not self.live:
                for message in above:
                    self.stream.write(message + "\n")
                self.stream.flush()
                return
            prefix = self._erase()
            if above:
                prefix += "".join(message + "\n" for message in above)
                self._drawn = 0  # everything still redrawable goes below them now
            self._redraw(prefix)

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
            elif not self.live and self.text.strip():
                self.stream.write(one_sentence_per_line(self.text.strip()) + "\n")
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
    name = "GigaAM-v3 e2e_rnnt"
    print(f"[caption] loading {name} on {args.asr_device}…", file=sys.stderr, flush=True)
    from sts.asr import create_asr  # heavy model imports; deferred past --list-devices

    asr = create_asr(ASRConfig(device=args.asr_device))
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
    transcript = Transcript(asr, args.final_after_unchanged, args.final_force_after)
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
                transcript.clear()
                while True:  # audio captured before the key press belongs to what's being cleared
                    try:
                        capture.frames.get_nowait()
                    except queue.Empty:
                        break
                last_run = time.monotonic()
                line.finish()  # the cleared text stays on screen, above the new one
                print("[caption] cleared - listening for new speech", file=sys.stderr, flush=True)
            try:
                transcript.add(capture.frames.get(timeout=0.1))
            except queue.Empty:
                pass
            now = time.monotonic()
            if transcript.has_audio and now - last_run >= interval_s:
                last_run = now
                final, live = transcript.step()
                # The blank line keeps what's final visibly apart from what may still change.
                line.update("\n" + "\n".join(live), above=[sentence.text for sentence in final])
    except KeyboardInterrupt:
        pass
    finally:
        redirect.close()
        line.finish()
        restore_terminal()
        capture.stop()


if __name__ == "__main__":
    main()
