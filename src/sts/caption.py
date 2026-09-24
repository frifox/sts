"""Diagnostic tool: streams an input device straight into a speech
recognizer - no VAD segmentation, no translation - and shows what the model
hears, sentence by sentence, as it hears and revises it.

The recognizer is Sber's GigaAM-v3 (e2e_rnnt, Russian-only, punctuated
output) by default - run it from .venv-gigaam, which has the newer torch/
transformers its model code needs (see sts.asr.gigaam_asr).
`--asr-backend parakeet` uses NVIDIA's Parakeet TDT instead, from a
virtualenv with NeMo (e.g. .venv-xtts).

Unlike the live pipeline (which cuts speech into segments on silence), this
never cuts on silence: the audio that isn't settled yet is re-transcribed
every --interval-ms, so the model sees more context each time and can revise
what it heard - which is shown as it happens, one sentence per line,
rewritten in place.

A sentence is locked - printed once, above a blank line, and never changed
again - once --final-after-sentences more sentences have followed it
(default 2), or it ended --final-after-seconds ago (default 30). Its audio
is then dropped, so later re-transcriptions only cover what's still open.
A "sentence" the model starts with a lowercase letter isn't one - it's the
rest of the previous sentence, cut by a pause - and is joined back onto it
before either rule applies. Piped to a file, only locked sentences are
printed (plus whatever is still open on exit). GigaAM takes at most ~25 s
at a time, so a longer stretch that isn't locked yet is cut into ~20 s
chunks at quiet moments. Meant for exploratory runs.

Press "x" to clear: whatever isn't locked yet stays on screen as it is, the
audio behind it is dropped, and recognition starts over from the next thing
said - e.g. to start fresh on a new speaker.

Usage:
  .venv-gigaam/bin/sts-caption --input-device 7
  .venv-gigaam/bin/sts-caption --input-device 7 --interval-ms 1000
  .venv-gigaam/bin/sts-caption --input-device 7 --final-after-sentences 3 --final-after-seconds 20
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
from dataclasses import dataclass

import numpy as np

from sts.asr import TimedWord
from sts.audio.capture import AudioCapture, list_devices
from sts.config import ASRConfig, AudioConfig
from sts.textclean import clean_transcript


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-device", type=int, default=None, help="ALSA input device index")
    parser.add_argument(
        "--asr-backend", choices=("gigaam", "parakeet"), default="gigaam",
        help="gigaam (GigaAM-v3 e2e_rnnt, Russian-only; run from .venv-gigaam) or parakeet (default: gigaam)",
    )
    parser.add_argument("--asr-device", default="cuda", help="cuda | cpu | mps (default: cuda)")
    parser.add_argument(
        "--final-after-sentences", type=int, default=2, metavar="N",
        help="Lock a sentence once N more sentences have come after it (default: 2; 0 turns this rule off)",
    )
    parser.add_argument(
        "--final-after-seconds", type=float, default=30.0, metavar="S",
        help="Lock a sentence once it ended S seconds ago (default: 30; 0 turns this rule off)",
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


@dataclass
class Sentence:
    """A sentence (or the unfinished tail of one), timed in seconds since
    sts-caption started listening."""

    text: str
    start: float
    end: float
    complete: bool  # ends with a sentence-ending mark


# A word that ends a sentence: ., ?, ! or … last, possibly followed by a
# closing quote/bracket ("ветви»." / "ветви.»").
_ENDS_SENTENCE = re.compile(r'[.?!…]["»”)]*$')


def sentences_from_words(words: list[TimedWord]) -> list[Sentence]:
    sentences: list[Sentence] = []
    current: list[TimedWord] = []
    for word in words:
        current.append(word)
        if _ENDS_SENTENCE.search(word.text):
            sentences.append(Sentence(" ".join(w.text for w in current), current[0].start, current[-1].end, True))
            current = []
    if current:
        sentences.append(Sentence(" ".join(w.text for w in current), current[0].start, current[-1].end, False))
    return sentences


def _starts_lowercase(text: str) -> bool:
    first_letter = next((c for c in text if c.isalpha()), "")
    return first_letter.islower()


def join_continuations(sentences: list[Sentence]) -> list[Sentence]:
    """A "sentence" whose first letter is lowercase isn't one: the model put
    a sentence end where the speaker only paused ("...и умыл ноги всем
    ученикам, он начал." + "с ними беседовать."). It's joined onto the one
    before, whose end mark is dropped."""
    joined: list[Sentence] = []
    for sentence in sentences:
        if joined and _starts_lowercase(sentence.text):
            previous = joined[-1]
            text = f"{re.sub(r'[.?!…]+$', '', previous.text)} {sentence.text}"
            joined[-1] = Sentence(text, previous.start, sentence.end, sentence.complete)
        else:
            joined.append(sentence)
    return joined


def count_final(sentences: list[Sentence], now_s: float, after_sentences: int, after_seconds: float) -> int:
    """How many of the leading `sentences` are final: those with at least
    `after_sentences` sentences after them, or complete and ended at least
    `after_seconds` before `now_s` (0 turns either rule off). Only a
    leading run can be final - a sentence is never locked while one
    before it could still change."""
    count = 0
    for i, sentence in enumerate(sentences):
        by_position = after_sentences > 0 and len(sentences) - 1 - i >= after_sentences
        by_age = after_seconds > 0 and sentence.complete and now_s - sentence.end >= after_seconds
        if not (by_position or by_age):
            break
        count = i + 1
    return count


class Transcript:
    """The audio heard so far and what's been made of it. Each step()
    re-transcribes the audio that isn't locked yet, returns the sentences
    that have just become final (see count_final) and the ones still open,
    and drops the final ones' audio: they're never transcribed - or changed
    - again.

    A little of the locked audio (CONTEXT_S) is kept in front of what's
    still open, so the model hears how the next sentence begins in context;
    words from it (those centered before the cut) are discarded."""

    CONTEXT_S = 1.5

    def __init__(self, asr, after_sentences: int, after_seconds: float, sample_rate: int = 16_000):
        self.asr = asr
        self.after_sentences = after_sentences
        self.after_seconds = after_seconds
        self.sample_rate = sample_rate
        self.clear()

    def clear(self) -> None:
        self._frames: list[np.ndarray] = []
        self._offset_s = 0.0  # when the first buffered sample was heard
        self._cut_s = 0.0  # words centered before this are already final

    def add(self, frame: np.ndarray) -> None:
        self._frames.append(frame)

    @property
    def has_audio(self) -> bool:
        return bool(self._frames)

    def step(self) -> tuple[list[str], list[str]]:
        audio = np.concatenate(self._frames)
        self._frames = [audio]
        words = [
            TimedWord(w.text, w.start + self._offset_s, w.end + self._offset_s)
            for w in self.asr.transcribe_words(audio)
            if (w.start + w.end) / 2 + self._offset_s >= self._cut_s
        ]
        sentences = join_continuations(sentences_from_words(words))
        now_s = self._offset_s + len(audio) / self.sample_rate
        n_final = count_final(sentences, now_s, self.after_sentences, self.after_seconds)
        if n_final:
            self._cut_s = sentences[n_final - 1].end
            drop = int(max(0.0, self._cut_s - self.CONTEXT_S - self._offset_s) * self.sample_rate)
            self._frames = [audio[drop:]]
            self._offset_s += drop / self.sample_rate
        return _texts(sentences[:n_final]), _texts(sentences[n_final:])


def _texts(sentences: list[Sentence]) -> list[str]:
    cleaned = (clean_transcript(sentence.text) for sentence in sentences)
    return [text for text in cleaned if text]


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
    transcript = Transcript(asr, args.final_after_sentences, args.final_after_seconds)
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
                line.update("\n" + "\n".join(live), above=final)
    except KeyboardInterrupt:
        pass
    finally:
        redirect.close()
        line.finish()
        restore_terminal()
        capture.stop()


if __name__ == "__main__":
    main()
