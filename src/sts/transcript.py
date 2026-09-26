"""Turns a stream of audio into sentences, locking each one once it has
settled - the captioning approach shared by the live pipeline and
sts-caption.

The audio that isn't final yet is re-transcribed every step, so the model
keeps hearing more context and can revise what it heard. A sentence is
final - its audio dropped, so it's never re-transcribed or reworded again -
as soon as either of these holds (see count_final):

- it ends with a sentence-ending mark and its text, punctuation included,
  hasn't changed for `after_unchanged` seconds. The wait starts only once
  the mark is there, and any change to the sentence starts it over. Text
  after it doesn't count;
- `force_after` seconds have passed since its first word, however much it's
  still changing, finished or not.
"""

import re
import threading
from dataclasses import dataclass

import numpy as np

from sts.asr import TimedWord
from sts.textclean import clean_transcript


@dataclass
class Sentence:
    """A sentence (or the unfinished tail of one), timed in seconds since
    listening started."""

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
        if not current:
            # A sentence can't start with punctuation - GigaAM sometimes puts a
            # stray "." on the first word it hears (".способны").
            word = TimedWord(word.text.lstrip(".,;:!?… "), word.start, word.end)
            if not word.text:
                continue
        current.append(word)
        if _ENDS_SENTENCE.search(word.text):
            sentences.append(Sentence(" ".join(w.text for w in current), current[0].start, current[-1].end, True))
            current = []
    if current:
        sentences.append(Sentence(" ".join(w.text for w in current), current[0].start, current[-1].end, False))
    return sentences


def count_final(sentences: list[Sentence], now_s: float, unchanged_since: list[float],
                after_unchanged: float, force_after: float) -> int:
    """How many of the leading `sentences` are final: complete ones whose
    text has stayed the same since `unchanged_since[i]` for at least
    `after_unchanged` seconds, and any whose first word was `force_after` or
    more seconds before `now_s` (0 turns either rule off). A sentence's text
    includes its end mark, so it's unchanged only since it got that mark.
    Only a leading run can be final - a sentence is never final while one
    before it could still change."""
    count = 0
    for i, sentence in enumerate(sentences):
        settled = after_unchanged > 0 and sentence.complete and now_s - unchanged_since[i] >= after_unchanged
        forced = force_after > 0 and now_s - sentence.start >= force_after
        if not (settled or forced):
            break
        count = i + 1
    return count


class Transcript:
    """The audio heard so far and what's been made of it. Each step()
    re-transcribes the audio that isn't locked yet, locks the sentences that
    have become final (see count_final), and drops their audio: they're
    never transcribed - or reworded - again.

    A little of the locked audio (CONTEXT_S) is kept in front of what's
    still open, so the model hears how the next sentence begins in context;
    words from it (those centered before the cut) are discarded."""

    CONTEXT_S = 1.5

    def __init__(self, asr, after_unchanged: float = 5.0, force_after: float = 30.0, sample_rate: int = 16_000):
        self.asr = asr
        self.after_unchanged = after_unchanged  # 0 turns it off
        self.force_after = force_after  # 0 turns it off
        # Seconds until the first still-open sentence is finalized by one of
        # the time rules (None when nothing is open or both are off) - as of
        # the last step(); for showing a countdown.
        self.finalize_in_s: float | None = None
        self.sample_rate = sample_rate
        # add() is called from the capture side while step() runs on another
        # thread; the lock covers the buffer, not the (slow) transcription.
        self._lock = threading.Lock()
        self._generation = 0
        self._last_open: list[Sentence] = []
        self.clear()

    def clear(self) -> None:
        with self._lock:
            self._frames: list[np.ndarray] = []
            self._offset_s = 0.0  # when the first buffered sample was heard
            self._cut_s = 0.0  # words centered before this are already locked
            # (text, unchanged since) of each still-open sentence, as of the last step().
            self._open_seen: list[tuple[str, float]] = []
            self.finalize_in_s = None
            self._generation += 1  # a step() running across this drops its result

    def add(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frames.append(frame)

    @property
    def has_audio(self) -> bool:
        return bool(self._frames)

    def step(self) -> tuple[list[Sentence], list[str]]:
        """(sentences that just became final, the text of those still open).
        A final sentence's text is disfluency-cleaned; its start/end still
        say when it was spoken."""
        with self._lock:
            if not self._frames:
                return [], self._open_texts([])
            audio = np.concatenate(self._frames)
            self._frames = []
            generation, offset_s, cut_s = self._generation, self._offset_s, self._cut_s
        # Transcribed without the lock, so audio keeps arriving meanwhile;
        # it's put back behind `audio` below.
        heard = self.asr.transcribe_words(audio)
        with self._lock:
            if generation != self._generation:  # clear() ran meanwhile
                return [], []
            words = [
                TimedWord(w.text, w.start + offset_s, w.end + offset_s)
                for w in heard
                if (w.start + w.end) / 2 + offset_s >= cut_s
            ]
            sentences = sentences_from_words(words)
            now_s = offset_s + len(audio) / self.sample_rate
            # A sentence is the same one as last step's at the same position
            # (the open run always starts right after the last lock); it's
            # unchanged since then if its text is too.
            unchanged_since = [
                self._open_seen[i][1] if i < len(self._open_seen) and self._open_seen[i][0] == sentence.text
                else now_s
                for i, sentence in enumerate(sentences)
            ]
            n_locked = count_final(sentences, now_s, unchanged_since, self.after_unchanged, self.force_after)
            final = sentences[:n_locked]
            still_open = sentences[n_locked:]
            self._open_seen = [(sentence.text, since) for sentence, since in
                               zip(still_open, unchanged_since[n_locked:])]
            self.finalize_in_s = self._finalize_in(still_open, unchanged_since[n_locked:], now_s)
            if n_locked:
                self._cut_s = sentences[n_locked - 1].end
            # Audio before the first still-open word (less CONTEXT_S) is done
            # with: locked, or silence/music nobody said anything in - which
            # would otherwise pile up for as long as nothing is said.
            keep_from_s = (still_open[0].start if still_open else now_s) - self.CONTEXT_S
            drop = int(min(max(0.0, keep_from_s - offset_s), len(audio) / self.sample_rate) * self.sample_rate)
            self._frames = [audio[drop:], *self._frames]
            self._offset_s = offset_s + drop / self.sample_rate
            return _cleaned(final), self._open_texts(still_open)

    def _finalize_in(self, still_open: list[Sentence], unchanged_since: list[float], now_s: float) -> float | None:
        if not still_open:
            return None
        waits = []
        if self.after_unchanged > 0 and still_open[0].complete:
            waits.append(self.after_unchanged - (now_s - unchanged_since[0]))
        if self.force_after > 0:
            waits.append(self.force_after - (now_s - still_open[0].start))
        return max(0.0, min(waits)) if waits else None

    def flush(self) -> list[Sentence]:
        """Whatever was open at the last step(), as final sentences, and
        clears the buffer: for when listening stops, so nothing already
        heard is lost."""
        with self._lock:
            sentences = self._last_open
            self._last_open = []
        self.clear()
        return sentences

    def _open_texts(self, still_open: list[Sentence]) -> list[str]:
        self._last_open = _cleaned(still_open)
        return [sentence.text for sentence in self._last_open]


def _cleaned(sentences: list[Sentence]) -> list[Sentence]:
    """`sentences` with disfluency-cleaned text, minus any left empty."""
    cleaned = (Sentence(clean_transcript(s.text), s.start, s.end, s.complete) for s in sentences)
    return [sentence for sentence in cleaned if sentence.text]
