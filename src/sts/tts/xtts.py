"""Synthesizes English speech with Coqui XTTS v2.

XTTS supports both a set of built-in speakers and zero-shot voice cloning
from a short reference clip, which covers the stretch goal (clone the live
speaker's voice from ~10-20s of their own audio) with no separate model.

`cfg.device` is passed straight to `.to(...)`, so "cuda", "cpu", and "mps"
(Apple Silicon) all work as far as PyTorch is concerned - unlike Qwen3-TTS,
there's no flash-attention/bfloat16 choice baked into this backend to worry
about per device.
"""

import queue
import re
import threading

import numpy as np
import soundfile as sf
import torch
import torchaudio
from TTS.api import TTS
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsArgs, XttsAudioConfig
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers.generation.streamers import BaseStreamer

from sts.config import TTSConfig
from sts.audioquality import level_gain, match_level
from sts.gpu import gpu_lock
from sts.textclean import split_for_tts
from sts.tts.voice_library import VoiceEntry

SAMPLE_RATE = 24_000  # XTTS v2 output rate

# Coqui TTS 0.22 predates PyTorch 2.6's weights_only=True default for
# torch.load, so its own checkpoint (which pickles these config classes)
# fails to load unless they're allowlisted.
if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])


# XTTS asserts on more than 400 tokens per sentence and warns past 250 characters
# for English. Text is spoken one sentence at a time; only a sentence longer than
# this is cut further, at clause boundaries.
MAX_CHARS_PER_CHUNK = 220

# Live playback streams each piece as XTTS generates it: audio is decoded
# every this many GPT tokens (about 46 ms of speech each, so ~1 s per chunk),
# and the first chunk can play while the rest is still being generated.
STREAM_CHUNK_TOKENS = 20
# How much of a streamed piece is held back to set its loudness from (see
# stream_chunks) - about one chunk, so it costs next to no extra delay.
LEVEL_WINDOW_S = 0.8
# Samples cross-faded between consecutive streamed chunks (XTTS's own default).
STREAM_OVERLAP_SAMPLES = 1024
# Xtts.inference()'s own defaults, used for a cloned voice (see _voice_conditioning).
_INFERENCE_SETTINGS = {
    "temperature": 0.75, "length_penalty": 1.0, "repetition_penalty": 10.0, "top_k": 50, "top_p": 0.85,
}


class _TokenQueue(BaseStreamer):
    """Receives GPT audio codes from transformers' generate() as they're
    sampled (on the generating thread) and hands them to the consuming one."""

    def __init__(self):
        self.items: queue.Queue = queue.Queue()
        self._prompt_skipped = False

    def put(self, value):
        if not self._prompt_skipped:  # generate() first passes the prompt itself
            self._prompt_skipped = True
            return
        self.items.put(value.reshape(-1).tolist())

    def end(self):
        self.items.put(None)


class _Cancelled(StoppingCriteria):
    def __init__(self, event: threading.Event):
        self.event = event

    def __call__(self, input_ids, scores, **kwargs):
        return self.event.is_set()


def split_for_xtts(text: str, limit: int = MAX_CHARS_PER_CHUNK) -> list[str]:
    return split_for_tts(text, limit)


def _load_with_soundfile(path, *args, **kwargs):
    """Drop-in for `torchaudio.load` returning (channels x frames float32 tensor, sample rate).

    torchaudio 2.9+ dropped its own decoders and now requires torchcodec (and
    ffmpeg) for `load`. Coqui XTTS calls `torchaudio.load` to read the
    reference clip for cloning, and our reference clips are plain wav files
    that soundfile (already a dependency) reads fine.
    """
    data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(data.T.copy()), sample_rate


torchaudio.load = _load_with_soundfile


class XTTSSynthesizer:
    def __init__(self, cfg: TTSConfig):
        self.cfg = cfg
        self.tts = TTS(cfg.model_name).to(cfg.device)
        self.cloned_speaker_wav: str | None = cfg.speaker_wav
        self.default_voice = cfg.resolve_voice(cfg.default_voice)
        self._conditioning_cache: dict = {}

    def set_default_voice(self, voice: str) -> None:
        """Picks which built-in voice is used when no cloned voice is active."""
        voice = self.cfg.resolve_voice(voice)
        if voice not in self.cfg.all_voices():
            raise ValueError(f"unknown built-in voice {voice!r}; choose from {self.cfg.all_voices()}")
        self.default_voice = voice

    def clone_voice(self, reference_wav_path: str, ref_text: str | None = None) -> None:
        """Registers a reference clip (10-20s is enough) to clone for
        subsequent synthesis. `ref_text` is accepted for interface parity
        with `Qwen3TTSSynthesizer` but unused - XTTS's cloning doesn't need
        a transcript of the reference clip."""
        self.cloned_speaker_wav = reference_wav_path

    def clone_from_entry(self, entry: VoiceEntry) -> None:
        self.clone_voice(str(entry.wav_path), entry.ref_text)

    def reset_voice(self) -> None:
        self.cloned_speaker_wav = None

    def synthesize_stream(self, text: str):
        """Yields (audio, sample_rate) per chunk as each is synthesized, so
        playback can start before a long sentence has been fully rendered."""
        for chunk in split_for_xtts(text):
            audio = self._synthesize_piece(chunk)
            if self.cfg.target_level:
                audio = match_level(audio, SAMPLE_RATE, self.cfg.target_level)
            yield audio, SAMPLE_RATE

    def stream_chunks(self, text: str):
        """Yields (audio, sample_rate, seamless) about a second at a time as
        XTTS generates it - for live playback, where the first English is
        then heard about one chunk's generation time after the sentence
        arrives, instead of after the whole sentence has been rendered.
        `seamless` is False for the first chunk of each piece (it gets the
        usual pause before it) and True for the rest, which continue it.

        Loudness is matched per piece like synthesize_stream, but only the
        start of the piece exists when its first chunk has to go out, so the
        gain is set from its first LEVEL_WINDOW_S (the whole piece, if that's
        shorter - e.g. "Amen.", the case the matching matters most for) and
        applied to the rest, peak-limited."""
        for piece in split_for_xtts(text):
            yield from self._stream_piece(piece)

    def _stream_piece(self, text: str):
        target = self.cfg.target_level
        gain = None if target else 1.0
        held: list[np.ndarray] = []
        first = True
        for audio in self._generate_stream(text):
            if gain is None:
                held.append(audio)
                if sum(len(part) for part in held) < LEVEL_WINDOW_S * SAMPLE_RATE:
                    continue
                audio, held = np.concatenate(held), []
                gain = level_gain(audio, SAMPLE_RATE, target)
            yield np.clip(audio * gain, -0.99, 0.99).astype(np.float32), SAMPLE_RATE, not first
            first = False
        if held:  # the whole piece was shorter than the level window
            yield match_level(np.concatenate(held), SAMPLE_RATE, target), SAMPLE_RATE, not first

    def _generate_stream(self, text: str):
        """Raw audio for one piece, yielded every STREAM_CHUNK_TOKENS codes.

        Coqui's own Xtts.inference_stream() can't be used: its bundled
        generation loop (TTS/tts/layers/xtts/stream_generator.py) calls
        transformers internals that no longer exist in the versions this
        project installs. This builds the same thing from the parts the
        regular, non-streaming inference() uses: GPT generate() runs on a
        worker thread with a standard transformers streamer; every chunk's
        worth of codes, the latents for all codes so far are recomputed
        (the GPT is causal, so a prefix's latents are exactly what the full
        pass would produce for it) and decoded, and each new stretch of
        audio is cross-faded onto the last with XTTS's own handle_chunks().
        Given the same codes, the result is what inference() would decode,
        apart from those cross-faded joins."""
        model = self.tts.synthesizer.tts_model
        gpt_latent, speaker_embedding, settings = self._voice_conditioning()
        gpt_latent, speaker_embedding = gpt_latent.to(model.device), speaker_embedding.to(model.device)
        language = self.cfg.language.split("-")[0]
        text_tokens = torch.IntTensor(model.tokenizer.encode(text.strip().lower(), lang=language)).unsqueeze(0)
        text_tokens = text_tokens.to(model.device)
        text_len = torch.tensor([text_tokens.shape[-1]], device=model.device)
        tokens, cancelled = _TokenQueue(), threading.Event()

        def generate() -> None:
            try:
                with torch.inference_mode(), gpu_lock(self.cfg.device):
                    model.gpt.generate(
                        cond_latents=gpt_latent, text_inputs=text_tokens, input_tokens=None,
                        do_sample=True, num_beams=1, num_return_sequences=1, output_attentions=False,
                        streamer=tokens, stopping_criteria=StoppingCriteriaList([_Cancelled(cancelled)]),
                        **settings,
                    )
            except BaseException as exc:  # handed to the consumer, which re-raises it
                tokens.items.put(exc)
                tokens.items.put(None)

        threading.Thread(target=generate, name="xtts-gpt", daemon=True).start()
        codes: list[int] = []
        decoded = 0
        wav_prev = wav_overlap = None
        try:
            while True:
                item = tokens.items.get()
                if isinstance(item, BaseException):
                    raise item
                done = item is None
                if not done:
                    codes.extend(item)
                if not codes or (not done and len(codes) - decoded < STREAM_CHUNK_TOKENS):
                    if done:
                        return
                    continue
                decoded = len(codes)
                with torch.inference_mode(), gpu_lock(self.cfg.device):
                    code_tensor = torch.tensor([codes], device=model.device)
                    latents = model.gpt(
                        text_tokens, text_len, code_tensor,
                        torch.tensor([code_tensor.shape[-1] * model.gpt.code_stride_len], device=model.device),
                        cond_latents=gpt_latent, return_attentions=False, return_latent=True,
                    )
                    wav_gen = model.hifigan_decoder(latents, g=speaker_embedding).squeeze()
                    chunk, wav_prev, wav_overlap = model.handle_chunks(
                        wav_gen, wav_prev, wav_overlap, STREAM_OVERLAP_SAMPLES
                    )
                    if done and wav_overlap is not None:
                        chunk = torch.cat([chunk, wav_overlap])  # the end of the piece: nothing left to fade into
                yield chunk.float().cpu().numpy().ravel()
                if done:
                    return
        finally:
            cancelled.set()  # e.g. Reset stopped consuming mid-piece: don't keep generating

    def _voice_conditioning(self):
        """(gpt_cond_latent, speaker_embedding, generation settings) for the
        active voice - the same ones the non-streaming path below uses: a
        clone's latents with inference()'s own default settings, or a
        built-in speaker's with the model config's (what tts.tts() applies)."""
        if self.cloned_speaker_wav:
            gpt_latent, speaker_embedding = self._clone_conditioning(self.cloned_speaker_wav)
            return gpt_latent, speaker_embedding, dict(_INFERENCE_SETTINGS)
        model = self.tts.synthesizer.tts_model
        speaker = model.speaker_manager.speakers[self.default_voice]
        config = model.config
        settings = {
            "temperature": config.temperature,
            "length_penalty": config.length_penalty,
            "repetition_penalty": config.repetition_penalty,
            "top_k": config.top_k,
            "top_p": config.top_p,
        }
        return speaker["gpt_cond_latent"], speaker["speaker_embedding"], settings

    def synthesize(self, text: str, pause_s: float = 0.35) -> tuple[np.ndarray, int]:
        """Whole text as one clip, with `pause_s` of silence between sentences."""
        pieces = [audio for audio, _ in self.synthesize_stream(text)]
        if not pieces:
            return np.zeros(0, dtype=np.float32), SAMPLE_RATE
        gap = np.zeros(int(pause_s * SAMPLE_RATE), dtype=np.float32)
        return np.concatenate([part for i, piece in enumerate(pieces) for part in ((gap, piece) if i else (piece,))]), SAMPLE_RATE

    def _clone_conditioning(self, wav_path: str):
        """(gpt_cond_latent, speaker_embedding) for the active clone, with its
        style latent blended toward the accent voice's. Cached per (clip, accent)."""
        accent_speaker = self.cfg.resolve_voice(self.cfg.accent_voice or self.default_voice)
        strength = self.cfg.accent_strength
        key = (wav_path, accent_speaker, strength)
        if key not in self._conditioning_cache:
            model = self.tts.synthesizer.tts_model
            gpt_latent, speaker_embedding = model.get_conditioning_latents([wav_path])
            if strength > 0:
                accent_latent = model.speaker_manager.speakers[accent_speaker]["gpt_cond_latent"]
                accent_latent = accent_latent.to(gpt_latent.device, gpt_latent.dtype)
                gpt_latent = strength * accent_latent + (1 - strength) * gpt_latent
            self._conditioning_cache[key] = (gpt_latent, speaker_embedding)
        return self._conditioning_cache[key]

    def _synthesize_piece(self, text: str) -> np.ndarray:
        with gpu_lock(self.cfg.device):
            return self._synthesize_piece_locked(text)

    def _synthesize_piece_locked(self, text: str) -> np.ndarray:
        if self.cloned_speaker_wav:
            gpt_latent, speaker_embedding = self._clone_conditioning(self.cloned_speaker_wav)
            out = self.tts.synthesizer.tts_model.inference(
                text, self.cfg.language, gpt_latent, speaker_embedding
            )
            wav = out["wav"]
        else:
            wav = self.tts.tts(
                text=text, speaker=self.default_voice, language=self.cfg.language
            )
        return np.asarray(wav, dtype=np.float32)
