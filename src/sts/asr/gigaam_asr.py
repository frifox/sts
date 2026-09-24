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

from sts.config import ASRConfig
from sts.textclean import clean_transcript

MODEL_NAME = "ai-sage/GigaAM-v3"
REVISION = "e2e_rnnt"
SAMPLE_RATE = 16_000

# Chunk length for audio longer than GigaAM accepts (25 s): each cut is made
# at the quietest 20 ms frame between MIN_CHUNK_S and MAX_CHUNK_S after the
# previous one.
MIN_CHUNK_S = 14.0
MAX_CHUNK_S = 22.0
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
        quietest = int(np.argmin(np.mean(frames**2, axis=1)))
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
        audio = np.asarray(audio, dtype=np.float32)
        bounds = chunk_bounds(audio)
        texts = []
        for start, end in zip(bounds, bounds[1:]):
            chunk = audio[start:end]
            if end == len(audio):  # still open: more audio may yet join it
                texts.append(self._transcribe_chunk(chunk))
                continue
            key = hashlib.blake2b(chunk.tobytes(), digest_size=16).digest()
            if key not in self._cache:
                self._cache[key] = self._transcribe_chunk(chunk)
                if len(self._cache) > _CACHE_SIZE:
                    self._cache.popitem(last=False)
            self._cache.move_to_end(key)
            texts.append(self._cache[key])
        return clean_transcript(" ".join(text for text in texts if text))

    @torch.inference_mode()
    def _transcribe_chunk(self, chunk: np.ndarray) -> str:
        # What GigaAM's own transcribe() does after reading its file
        # argument with ffmpeg - called directly on the in-memory audio.
        inner = self.model.model
        device, dtype = next(inner.parameters()).device, next(inner.parameters()).dtype
        wav = torch.from_numpy(chunk).to(device=device, dtype=dtype).unsqueeze(0)
        length = torch.full([1], wav.shape[-1], device=device)
        encoded, encoded_len = inner(wav, length)
        return inner.decoding.decode(inner.head, encoded, encoded_len)[0].strip()


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
