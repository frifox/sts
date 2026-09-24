"""Transcribes Russian speech with Sber's GigaAM-v3, `e2e_rnnt` variant: a
Russian-only Conformer RNN-T that outputs punctuated, capitalized,
normalized text directly.

Needs its own virtualenv (`pip install -e '.[gigaam]'`, e.g. .venv-gigaam):
its code is loaded from the Hugging Face repo and pins newer torch/
transformers than the TTS backends allow.

GigaAM only takes up to 25 s at a time (longer input is refused - its own
long-form mode needs pyannote and a Hugging Face token). Longer audio is cut
into chunks of at most MAX_CHUNK_S at the quietest moment near each limit,
chosen from the start of the audio onward. Each cut therefore depends only
on audio before it, so re-transcribing a growing buffer (sts-caption) puts
the same cuts in the same places every time: a chunk that's complete comes
out identically on every run, and is cached instead of re-transcribed.
"""

import hashlib
from collections import OrderedDict
from contextlib import contextmanager

import numpy as np
import torch

from sts.asr import TimedWord
from sts.config import ASRConfig
from sts.textclean import clean_transcript

MODEL_NAME = "ai-sage/GigaAM-v3"
REVISION = "e2e_rnnt"
SAMPLE_RATE = 16_000

# Chunk length for audio longer than GigaAM accepts (25 s): each cut is made
# in the quietest stretch between MIN_CHUNK_S and MAX_CHUNK_S after the
# previous one - judged over PAUSE_S of audio, not a single 20 ms frame,
# which could be the brief silence inside a word (a stop consonant's
# closure) and cut it in two: "в Нём" came out as "в Н" that way.
MIN_CHUNK_S = 14.0
MAX_CHUNK_S = 22.0
PAUSE_S = 0.3
_FRAME = SAMPLE_RATE // 50

# Completed chunks' transcripts, kept so a growing buffer only pays for its
# still-open last chunk.
_CACHE_SIZE = 256


def chunk_bounds(audio: np.ndarray) -> list[int]:
    """Sample offsets [0, cut1, cut2, ..., n_samples] splitting `audio` into
    chunks of at most MAX_CHUNK_S - see the module docstring."""
    n_samples = len(audio)
    max_len, min_len = int(MAX_CHUNK_S * SAMPLE_RATE), int(MIN_CHUNK_S * SAMPLE_RATE)
    bounds = [0]
    while n_samples - bounds[-1] > max_len:
        start = bounds[-1]
        window = audio[start + min_len : start + max_len]
        frames = window[: len(window) // _FRAME * _FRAME].reshape(-1, _FRAME)
        span = max(1, int(PAUSE_S * SAMPLE_RATE / _FRAME))
        energy = np.convolve(np.mean(frames**2, axis=1), np.ones(span) / span, mode="same")
        quietest = int(np.argmin(energy))
        bounds.append(start + min_len + quietest * _FRAME + _FRAME // 2)
    bounds.append(n_samples)
    return bounds


@contextmanager
def _without_pyannote_check():
    """transformers refuses to load remote model code whose imports aren't
    all installed - including GigaAM's `import pyannote`, which only its
    long-form mode (not used here) needs, inside that function. It's left
    out of the list of imports transformers checks."""
    import transformers.dynamic_module_utils as dmu

    original = dmu.get_imports

    def get_imports(filename):
        return [name for name in original(filename) if name != "pyannote"]

    dmu.get_imports = get_imports
    try:
        yield
    finally:
        dmu.get_imports = original


class GigaAMASR:
    def __init__(self, cfg: ASRConfig):
        self.cfg = cfg
        from transformers import AutoModel  # the gigaam extra's pinned version

        with _without_pyannote_check():
            model = AutoModel.from_pretrained(MODEL_NAME, revision=REVISION, trust_remote_code=True)
        self.model = model.to(cfg.device).eval()
        self._cache: OrderedDict[bytes, str] = OrderedDict()

    def close(self) -> None:
        """Releases the model's GPU memory now, even if something (a
        traceback, a stray reference) still holds on to this object."""
        self.model.to("cpu")

    def transcribe(self, audio: np.ndarray | str) -> str:
        if isinstance(audio, str):
            audio = _read_audio(audio)
        return clean_transcript(" ".join(word.text for word in self.transcribe_words(audio)))

    def transcribe_words(self, audio: np.ndarray) -> list[TimedWord]:
        """Each word with its timing (see TimedWord), across however many
        chunks the audio needed. Not disfluency-cleaned: the caller decides
        what a sentence is before cleaning its text."""
        audio = np.asarray(audio, dtype=np.float32)
        bounds = chunk_bounds(audio)
        words: list[TimedWord] = []
        for start, end in zip(bounds, bounds[1:]):
            chunk = audio[start:end]
            if end == len(audio):  # still open: more audio may yet join it
                chunk_words = self._chunk_words(chunk)
            else:
                key = hashlib.blake2b(chunk.tobytes(), digest_size=16).digest()
                if key not in self._cache:
                    self._cache[key] = self._chunk_words(chunk)
                    if len(self._cache) > _CACHE_SIZE:
                        self._cache.popitem(last=False)
                self._cache.move_to_end(key)
                chunk_words = self._cache[key]
            offset = start / SAMPLE_RATE
            words.extend(TimedWord(w.text, w.start + offset, w.end + offset) for w in chunk_words)
        return words

    @torch.inference_mode()
    def _chunk_words(self, chunk: np.ndarray) -> list[TimedWord]:
        # What GigaAM's own transcribe() does after reading its file
        # argument with ffmpeg - called directly on the in-memory audio -
        # except for the decoding, which also notes when each token came out.
        inner = self.model.model
        device, dtype = next(inner.parameters()).device, next(inner.parameters()).dtype
        wav = torch.from_numpy(chunk).to(device=device, dtype=dtype).unsqueeze(0)
        length = torch.full([1], wav.shape[-1], device=device)
        encoded, encoded_len = inner(wav, length)
        n_frames = int(encoded_len[0])
        if not n_frames:
            return []
        frame_s = len(chunk) / SAMPLE_RATE / n_frames
        tokens = _greedy_decode(inner.head, inner.decoding, encoded[0].transpose(0, 1).unsqueeze(1), n_frames)
        return _words_from_tokens(tokens, inner.decoding.tokenizer.model, frame_s)


def _greedy_decode(head, decoding, frames: torch.Tensor, n_frames: int) -> list[tuple[int, int]]:
    """(token id, encoder frame it was emitted on) for one sequence: GigaAM's
    own RNNTGreedyDecoding._greedy_decode, which only returns the ids. An
    RNN-T emits a token once it has heard enough of it, so a token's frame
    is roughly where that piece of speech ends."""
    out: list[tuple[int, int]] = []
    state = None
    last_label = None
    for t in range(n_frames):
        f = frames[t, :, :].unsqueeze(1)
        for _ in range(decoding.max_symbols):
            g, hidden = head.decoder.predict(last_label, state)
            k = head.joint.joint(f, g)[0, 0, 0, :].argmax(0).item()
            if k == decoding.blank_id:
                break
            out.append((int(k), t))
            state = hidden
            last_label = torch.tensor([[k]], device=frames.device)
    return out


def _words_from_tokens(tokens: list[tuple[int, int]], sp, frame_s: float) -> list[TimedWord]:
    """Groups SentencePiece tokens into words ("▁" starts a new one) and
    times each from its first to its last token's frame."""
    groups: list[list[tuple[int, int]]] = []
    for token in tokens:
        if not groups or sp.id_to_piece(token[0]).startswith("▁"):
            groups.append([])
        groups[-1].append(token)
    words = []
    for group in groups:
        text = sp.decode([token_id for token_id, _ in group]).strip()
        if text:
            words.append(TimedWord(text, group[0][1] * frame_s, (group[-1][1] + 1) * frame_s))
    return words


def _read_audio(path: str) -> np.ndarray:
    """A file (e.g. a --clone reference clip) as 16 kHz mono float32."""
    from math import gcd

    import soundfile as sf
    from scipy.signal import resample_poly

    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if rate != SAMPLE_RATE:
        divisor = gcd(rate, SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // divisor, rate // divisor).astype(np.float32)
    return audio
