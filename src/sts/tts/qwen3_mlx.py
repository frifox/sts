"""Qwen3-TTS on Apple's MLX (the `mlx-audio` package, `.venv-mlx`), behind the
same interface as `qwen_tts.Qwen3TTSModel` - the part of it Qwen3TTSSynthesizer
uses - so everything around generation (the English reference for a non-English
voice, the length checks, tone and level matching) works the same either way.

Same models, converted to MLX (mlx-community/...-bf16): measured on 17 sermon
sentences on Apple Silicon, the 1.7B model generated at 0.25-0.29x real time,
against 0.71-0.72x with PyTorch on mps - the per-step overhead of PyTorch's
Metal backend, not the model's size, is what made it slow (the 0.6B model was
hardly faster there).

Not supported here: trained LoRA voices (a PyTorch/peft adapter); zero-shot
cloned voices, the preset voices and voice design are.
"""

import logging
import threading
from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 24_000


def mlx_model_name(name: str) -> str:
    """The MLX conversion of a Qwen3-TTS model: "Qwen/Qwen3-TTS-12Hz-1.7B-Base" ->
    "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16". Anything else (an MLX model
    named directly) is used as is."""
    if name.startswith("Qwen/Qwen3-TTS-"):
        return "mlx-community/" + name.split("/", 1)[1] + "-bf16"
    return name


@dataclass
class _ClonePrompt:
    ref_audio: object  # mx.array at the model's sample rate
    ref_text: str | None  # None: speaker embedding only (x-vector)
    speaker_audio: object = None  # whose voice, if not ref_audio's (see with_speaker)
    ref_codes: object = None  # ref_audio as codec tokens, once worked out (see _prime_stream_decoder)


class MLXQwen3TTSModel:
    def __init__(self, model):
        self._model = model
        self.model = None  # (no torch module: see weight_bytes)

    @classmethod
    def from_pretrained(cls, name: str, **_torch_options) -> "MLXQwen3TTSModel":
        from mlx_audio.tts.utils import load_model

        _quiet_config_type_warning()
        return cls(load_model(mlx_model_name(name)))

    @property
    def sample_rate(self) -> int:
        return int(getattr(self._model, "sample_rate", SAMPLE_RATE))

    def weight_bytes(self) -> int:
        from mlx.utils import tree_flatten

        return sum(array.nbytes for _, array in tree_flatten(self._model.parameters()))

    def create_voice_clone_prompt(self, ref_audio, ref_text=None, x_vector_only_mode=False) -> _ClonePrompt:
        from mlx_audio.utils import load_audio

        audio = load_audio(ref_audio, sample_rate=self.sample_rate) if isinstance(ref_audio, str) else ref_audio
        return _ClonePrompt(audio, None if x_vector_only_mode else ref_text)

    def with_speaker(self, prompt: _ClonePrompt, speaker: _ClonePrompt) -> _ClonePrompt:
        """`prompt`'s clip + text (how to speak), in `speaker`'s voice."""
        return _ClonePrompt(prompt.ref_audio, prompt.ref_text, speaker.speaker_audio or speaker.ref_audio)

    def speaker_vector(self, prompt: _ClonePrompt) -> np.ndarray:
        audio = prompt.speaker_audio if prompt.speaker_audio is not None else prompt.ref_audio
        return np.asarray(self._extract_speaker_embedding(audio), dtype=np.float32).reshape(-1)

    def _extract_speaker_embedding(self, audio):
        model = self._model
        original = model.__class__.extract_speaker_embedding
        return original(model, audio)

    def generate_voice_clone(self, text, language="English", voice_clone_prompt=None, non_streaming_mode=True, **sampling):
        # ICL (clip + transcript) always runs non-streaming text input in mlx-audio.
        prompt = voice_clone_prompt
        model = self._model
        if prompt.speaker_audio is not None:
            # mlx-audio takes the speaker embedding from ref_audio; here it's another clip's.
            embedding = self._extract_speaker_embedding(prompt.speaker_audio)
            model.extract_speaker_embedding = lambda _audio: embedding
        try:
            return self._collect(model.generate(
                text=text, ref_audio=prompt.ref_audio, ref_text=prompt.ref_text,
                lang_code=language.lower(), split_pattern="", **_mlx_sampling(sampling),
            ))
        finally:
            model.__dict__.pop("extract_speaker_embedding", None)

    def stream_voice_clone(self, text, voice_clone_prompt, interval_s, language="English", **sampling):
        """generate_voice_clone, streamed: the audio (np.float32) in pieces as
        it's generated, one about every `interval_s` of audio. Closing the
        generator stops generating."""
        prompt = voice_clone_prompt
        model = self._model
        if prompt.speaker_audio is not None:
            embedding = self._extract_speaker_embedding(prompt.speaker_audio)
            model.extract_speaker_embedding = lambda _audio: embedding
        if prompt.ref_text is not None:
            self._prime_stream_decoder(prompt)
        try:
            yield from self._pieces(model.generate(
                text=text, ref_audio=prompt.ref_audio, ref_text=prompt.ref_text,
                lang_code=language.lower(), split_pattern="", stream=True, streaming_interval=interval_s,
                **_mlx_sampling(sampling),
            ))
        finally:
            model.__dict__.pop("extract_speaker_embedding", None)
            _prime.codes = None

    def _prime_stream_decoder(self, prompt: _ClonePrompt) -> None:
        """Has the next streamed take decoded with the reference clip as its
        context, as a whole take is. Unstreamed, mlx-audio decodes a clone's
        tokens after the reference clip's own and cuts that part off, so the
        codec decoder already has the voice when the sentence starts; streamed,
        it resets the decoder and starts on the sentence cold, and its first
        second or so comes out with the pitch wavering in small, sudden steps
        (like auto-tune). So the reset that starts the stream is followed by
        decoding the reference clip's tokens, whose audio is thrown away."""
        import mlx.core as mx

        if prompt.ref_codes is None:
            audio = prompt.ref_audio
            audio = audio.reshape(1, 1, -1) if audio.ndim < 3 else audio  # as mlx-audio encodes it: [1, 1, samples]
            prompt.ref_codes = self._model.speech_tokenizer.encode(audio)  # [1, groups, time]
            mx.eval(prompt.ref_codes)
        _install_prime_hook()
        _prime.codes = prompt.ref_codes

    def stream_custom_voice(self, text, speaker, interval_s, language="English", instruct=None, **sampling):
        """generate_custom_voice, streamed (see stream_voice_clone)."""
        yield from self._pieces(self._model.generate_custom_voice(
            text=text, speaker=speaker, language=language, instruct=instruct,
            stream=True, streaming_interval=interval_s, **_mlx_sampling(sampling),
        ))

    @staticmethod
    def _pieces(results):
        try:
            for result in results:
                audio = np.asarray(result.audio, dtype=np.float32).reshape(-1)
                if audio.size:
                    yield audio
        finally:
            close = getattr(results, "close", None)
            if close:
                close()

    def generate_custom_voice(self, text, speaker, language="English", instruct=None, **sampling):
        return self._collect(self._model.generate_custom_voice(
            text=text, speaker=speaker, language=language, instruct=instruct, **_mlx_sampling(sampling),
        ))

    def generate_voice_design(self, text, instruct, language="English", **sampling):
        return self._collect(self._model.generate_voice_design(
            text=text, instruct=instruct, language=language, **_mlx_sampling(sampling),
        ))

    def _collect(self, results) -> tuple[list[np.ndarray], int]:
        audio = [np.asarray(result.audio, dtype=np.float32) for result in results]
        return [np.concatenate(audio) if audio else np.zeros(0, dtype=np.float32)], self.sample_rate


# The codec tokens the next streaming decoder reset is primed with (see
# MLXQwen3TTSModel._prime_stream_decoder); used once, by that reset.
_prime = threading.local()


def _install_prime_hook() -> None:
    """Makes the streaming decoder's reset decode `_prime.codes`, if set, and
    throw that audio away. On the class: the decoder object itself doesn't take
    new attributes."""
    from mlx_audio.tts.models.qwen3_tts.speech_tokenizer import Qwen3TTSSpeechTokenizerDecoder as Decoder

    if getattr(Decoder, "_sts_primes", False):
        return
    reset = Decoder.reset_streaming_state

    def reset_streaming_state(self):
        import mlx.core as mx

        reset(self)
        codes, _prime.codes = getattr(_prime, "codes", None), None
        if codes is not None:
            mx.eval(self.streaming_step(codes))

    Decoder.reset_streaming_state = reset_streaming_state
    Decoder._sts_primes = True


def _mlx_sampling(sampling: dict) -> dict:
    """qwen_tts's sampling options, as mlx-audio names them (it has no separate
    settings for the sub-talker)."""
    out = {}
    if "temperature" in sampling:
        out["temperature"] = sampling["temperature"]
    if "top_k" in sampling:
        out["top_k"] = sampling["top_k"]
    if "max_new_tokens" in sampling:
        out["max_tokens"] = sampling["max_new_tokens"]
    return out


class _DropConfigTypeWarning(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "to instantiate a model of type" not in record.getMessage()


def _quiet_config_type_warning() -> None:
    """mlx-audio reads each model's config through transformers' generic config
    class, which warns "You are using a model of type `qwen3_tts` to
    instantiate a model of type ``..." on every load - harmless (the model
    loads and speaks fine), but it looked like an error in the console. Only
    that message is dropped; transformers' other warnings still show."""
    target = logging.getLogger("transformers.configuration_utils")
    if not any(isinstance(f, _DropConfigTypeWarning) for f in target.filters):
        target.addFilter(_DropConfigTypeWarning())


def clear_cache() -> None:
    import mlx.core as mx

    mx.clear_cache()
