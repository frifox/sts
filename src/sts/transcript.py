"""Turns a stream of audio into sentences, locking each one once it has
settled - the captioning approach shared by the live pipeline and
sts-caption.

The audio that isn't locked yet is re-transcribed every step, so the model
keeps hearing more context and can revise what it heard. A sentence is
locked once `after_sentences` more sentences have followed it, or it ended
`after_seconds` ago; its audio is then dropped, so it's never re-transcribed
or reworded again. A "sentence" the model starts with a lowercase letter is
the rest of the previous one, split by a pause, and is joined back onto it.
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
    re-transcribes the audio that isn't locked yet, locks the sentences that
    have become final (see count_final), and drops their audio: they're
    never transcribed - or reworded - again.

    A little of the locked audio (CONTEXT_S) is kept in front of what's
    still open, so the model hears how the next sentence begins in context;
    words from it (those centered before the cut) are discarded.

    That context can change the model's mind about a sentence boundary:
    heard right after "...мы об этом рассуждаем", the next sentence may come
    back as "для чего Христос приходил" - a continuation, lowercase - where
    the model heard "Для чего..." as a new sentence before the lock. So the
    most recently locked sentence is held back ("pending") until the one
    after it is locked too: a lowercase continuation still joins onto it
    (see join_continuations), and only then is it returned as final. Until
    then it's returned first among the still-open sentences."""

    CONTEXT_S = 1.5

    def __init__(self, asr, after_sentences: int, after_seconds: float, sample_rate: int = 16_000):
        self.asr = asr
        self.after_sentences = after_sentences
        self.after_seconds = after_seconds
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
            self._pending: Sentence | None = None  # locked, not yet returned as final
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
            n_locked = count_final(sentences, now_s, self.after_sentences, self.after_seconds)
            final = self._lock_in(sentences[:n_locked])
            if n_locked:
                self._cut_s = sentences[n_locked - 1].end
            # Audio before the first still-open word (less CONTEXT_S) is done
            # with: locked, or silence/music nobody said anything in - which
            # would otherwise pile up for as long as nothing is said.
            still_open = sentences[n_locked:]
            keep_from_s = (still_open[0].start if still_open else now_s) - self.CONTEXT_S
            drop = int(min(max(0.0, keep_from_s - offset_s), len(audio) / self.sample_rate) * self.sample_rate)
            self._frames = [audio[drop:], *self._frames]
            self._offset_s = offset_s + drop / self.sample_rate
            return _cleaned(final), self._open_texts(still_open)

    def flush(self) -> list[Sentence]:
        """Everything not yet returned as final - the pending sentence and
        whatever was open at the last step() - as final sentences, and clears
        the buffer: for when listening stops, so nothing already heard is
        lost."""
        with self._lock:
            sentences = self._last_open
            self._last_open = []
        self.clear()
        return sentences

    def _lock_in(self, sentences: list[Sentence]) -> list[Sentence]:
        """Locks `sentences`, returning those now final - see the class docstring."""
        final: list[Sentence] = []
        for sentence in sentences:
            if self._pending is not None and not _starts_lowercase(sentence.text):
                final.append(self._pending)
                self._pending = sentence
            else:
                self._pending = join_continuations([self._pending, sentence])[0] if self._pending else sentence
        return final

    def _open_texts(self, still_open: list[Sentence]) -> list[str]:
        self._last_open = _cleaned(join_continuations(([self._pending] if self._pending else []) + still_open))
        return [sentence.text for sentence in self._last_open]


def _cleaned(sentences: list[Sentence]) -> list[Sentence]:
    """`sentences` with disfluency-cleaned text, minus any left empty."""
    cleaned = (Sentence(clean_transcript(s.text), s.start, s.end, s.complete) for s in sentences)
    return [sentence for sentence in cleaned if sentence.text]
