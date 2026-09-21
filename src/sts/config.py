"""Central configuration for the pipeline. Values can be overridden via CLI flags."""

from dataclasses import dataclass, field


@dataclass
class AudioConfig:
    input_device: int | None = None
    output_device: int | None = None
    sample_rate: int = 16_000
    channels: int = 1
    block_ms: int = 32  # frame fed to the VAD; Silero needs exactly 512 samples at 16 kHz (32 ms)


@dataclass
class VADConfig:
    # Silero VAD speech probability threshold.
    threshold: float = 0.5
    # How much trailing silence (ms) closes out a speech segment.
    min_silence_ms: int = 700
    # Segments shorter than this are dropped as noise.
    min_speech_ms: int = 250
    # Hard cap so a run-on speaker doesn't stall translation/TTS.
    max_segment_ms: int = 20_000


@dataclass
class ASRConfig:
    model_size: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "float16"
    source_language: str = "ru"
    # Nudges Whisper's decoding toward Baptist/religious vocabulary.
    initial_prompt: str = (
        "Проповедь. Библия, церковь, молитва, благодать, покаяние, "
        "крещение, спасение, Господь, Иисус Христос, вера, грех."
    )


@dataclass
class TranslationConfig:
    model_name: str = "facebook/nllb-200-distilled-1.3B"
    source_lang: str = "rus_Cyrl"
    target_lang: str = "eng_Latn"
    device: str = "cuda"
    apply_glossary: bool = True


@dataclass
class TTSConfig:
    backend: str = "xtts"  # "xtts" or "qwen3"
    model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    qwen3_model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    device: str = "cuda"
    language: str = "en"
    # Built-in (non-cloned) XTTS voices, used when no cloned/trained voice is
    # active. XTTS only - Qwen3-TTS's cloning model has no built-in speakers.
    builtin_voices: dict[str, str] = field(
        default_factory=lambda: {"male": "Damien Black", "female": "Claribel Dervla"}
    )
    default_voice: str = "male"  # key into builtin_voices
    # When set, clone this reference speaker instead (stretch goal).
    speaker_wav: str | None = None
    # Where per-speaker reference clips are saved for reuse across sessions.
    voice_library_dir: str = "voices"


@dataclass
class PipelineConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
