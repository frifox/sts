"""Turns a stream of audio into sentences, locking each one once it has
settled - the captioning approach shared by the live pipeline and
sts-caption.

The audio that isn't final yet is re-transcribed every step, so the model
keeps hearing more context and can revise what it heard. A sentence is
final - its audio dropped, so it's never re-transcribed or reworded again -
as soon as any of these holds (see count_final):

- it ends with a sentence-ending mark and its text, punctuation included,
  hasn't changed for `after_unchanged` seconds. The wait starts only once
  the mark is there, and any change to the sentence starts it over. Text
  after it doesn't count;
- `force_after` seconds have passed since its first word, however much it's
  still changing, finished or not;
- nothing new has been heard for `after_silence` seconds: the speaker has
  stopped, so everything still open is final, end mark or not (e.g. the
  last words of a sermon, left unpunctuated).

A "sentence" the model starts with a lowercase letter isn't one: it's the
rest of the previous sentence, split by a pause, and is joined back onto it
(its end mark dropped) while both are still open - which changes the
previous one, so its 5 s start over. A sentence that's already final is
never changed.
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


def _starts_lowercase(text: str) -> bool:
    first_letter = next((c for c in text if c.isalpha()), "")
    return first_letter.islower()


def join_continuations(sentences: list[Sentence]) -> list[Sentence]:
    """A "sentence" whose first letter is lowercase isn't one: the model put
    a sentence end where the speaker only paused ("...какие здесь плоды." +
    "и будете думать о том..."). It's joined onto the one before, whose end
    mark is dropped."""
    joined: list[Sentence] = []
    for sentence in sentences:
        if joined and _starts_lowercase(sentence.text):
            previous = joined[-1]
            text = f"{strip_end_mark(previous.text)} {sentence.text}"
            joined[-1] = Sentence(text, previous.start, sentence.end, sentence.complete)
        else:
            joined.append(sentence)
    return joined


def strip_end_mark(text: str) -> str:
    """`text` without its sentence-ending mark (a closing quote/bracket after it stays)."""
    return re.sub(r'[.?!…]+(["»”)]*)$', r"\1", text.rstrip())


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
    # How long before the silence and force rules finalize a sentence they're
    # reported (finalize_timers), as a wait of this length.
    WARN_S = 5.0

    def __init__(self, asr, after_unchanged: float = 5.0, force_after: float = 30.0, sample_rate: int = 16_000,
                 after_silence: float = 10.0):
        self.asr = asr
        self.after_unchanged = after_unchanged  # 0 turns it off
        self.force_after = force_after  # 0 turns it off
        self.after_silence = after_silence  # 0 turns it off
        # Per still-open sentence (the same ones, in the same order, as the
        # texts step() returns): (seconds left, the whole wait) before it's
        # finalized - by the 5 s rule once it has its end mark, or in the last
        # WARN_S before the silence or force rule, whichever is soonest - or
        # None. A later sentence's can run out while an earlier one is still
        # open: it then waits for that one (sentences are final in order).
        # As of the last step(); for the page's finalize indicators.
        self.finalize_timers: list[tuple[float, float] | None] = []
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
            self.finalize_timers = []
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
            sentences = join_continuations(sentences_from_words(words))
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
            if self.after_silence > 0 and sentences and now_s - sentences[-1].end >= self.after_silence:
                n_locked = len(sentences)  # nothing new heard for a while: the speaker has stopped
            final = sentences[:n_locked]
            still_open = sentences[n_locked:]
            self._open_seen = [(sentence.text, since) for sentence, since in
                               zip(still_open, unchanged_since[n_locked:])]
            timers = [self._finalize_timer(still_open, i, since, now_s)
                      for i, since in enumerate(unchanged_since[n_locked:])]
            if n_locked:
                self._cut_s = sentences[n_locked - 1].end
            # Audio before the first still-open word (less CONTEXT_S) is done
            # with: locked, or silence/music nobody said anything in - which
            # would otherwise pile up for as long as nothing is said.
            keep_from_s = (still_open[0].start if still_open else now_s) - self.CONTEXT_S
            drop = int(min(max(0.0, keep_from_s - offset_s), len(audio) / self.sample_rate) * self.sample_rate)
            self._frames = [audio[drop:], *self._frames]
            self._offset_s = offset_s + drop / self.sample_rate
            return _cleaned(final), self._open_texts(still_open, timers)

    @property
    def finalize_timer(self) -> tuple[float, float] | None:
        """The first still-open sentence's entry of finalize_timers."""
        return self.finalize_timers[0] if self.finalize_timers else None

    @property
    def finalize_in_s(self) -> float | None:
        """Seconds left on finalize_timer, if it's running."""
        return self.finalize_timer[0] if self.finalize_timer else None

    def _finalize_timer(self, still_open: list[Sentence], i: int, unchanged_since: float,
                        now_s: float) -> tuple[float, float] | None:
        """still_open[i]'s entry of finalize_timers."""
        sentence = still_open[i]
        timers = []  # (seconds left, whole wait) per rule that's due to be shown
        if self.after_unchanged > 0 and sentence.complete:
            timers.append((self.after_unchanged - (now_s - unchanged_since), self.after_unchanged))
        rules = [(self.after_silence, self.after_silence - (now_s - still_open[-1].end))]
        if sentence.complete:
            # A sentence without its end mark is still being built: only the
            # silence rule (the speaker stopped) is shown for it.
            rules.append((self.force_after, self.force_after - (now_s - sentence.start)))
        for rule_s, left in rules:
            if rule_s > 0 and left <= self.WARN_S:
                timers.append((left, min(self.WARN_S, rule_s)))
        if not timers:
            return None
        left, whole = min(timers)
        return max(0.0, left), whole

    def flush(self) -> list[Sentence]:
        """Whatever was open at the last step(), as final sentences, and
        clears the buffer: for when listening stops, so nothing already
        heard is lost."""
        with self._lock:
            sentences = self._last_open
            self._last_open = []
        self.clear()
        return sentences

    def _open_texts(self, still_open: list[Sentence], timers: list | None = None) -> list[str]:
        """The still-open sentences' cleaned text (sets finalize_timers to
        match: a sentence cleaned away takes its timer with it)."""
        kept = [
            (Sentence(clean_transcript(s.text), s.start, s.end, s.complete), timer)
            for s, timer in zip(still_open, timers or [None] * len(still_open))
        ]
        kept = [(sentence, timer) for sentence, timer in kept if sentence.text]
        self._last_open = [sentence for sentence, _ in kept]
        self.finalize_timers = [timer for _, timer in kept]
        return [sentence.text for sentence in self._last_open]


def _cleaned(sentences: list[Sentence]) -> list[Sentence]:
    """`sentences` with disfluency-cleaned text, minus any left empty."""
    cleaned = (Sentence(clean_transcript(s.text), s.start, s.end, s.complete) for s in sentences)
    return [sentence for sentence in cleaned if sentence.text]
