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
    # Drop a segment when Whisper both thinks it is probably not speech AND is
    # unsure of its own transcription (guards against hallucinated text on noise).
    no_speech_threshold: float = 0.6
    min_avg_logprob: float = -1.0
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
    # Qwen3-TTS's CustomVoice variant: named preset speakers, no reference clip
    # needed. Used when no cloned/trained voice is active. `qwen3_style_instruct`
    # steers delivery (ignored by the 0.6B variant).
    qwen3_custom_model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    # VoiceDesign variant: invents a voice from a text description (the "Create
    # Voice" dialog). Loaded only while that dialog is in use.
    qwen3_design_model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    qwen3_style_instruct: str = (
        "Speak in a calm, warm, measured tone, like a gentle preacher delivering a sermon. "
        "American English accent."
    )
    device: str = "cuda"
    language: str = "en"
    # Built-in (non-cloned) XTTS voices offered in the page's dropdown, used
    # when no cloned/trained voice is active. Hand-picked for a US-sounding
    # accent. XTTS only - Qwen3-TTS's cloning model has no built-in speakers.
    voice_groups: dict[str, list[str]] = field(
        default_factory=lambda: {
            "Male": [
                "Adde Michal", "Craig Gutsy", "Damien Black",
                "Dionisio Schuyler", "Eugenio Mataracı", "Ilkin Urbano",
            ],
            "Female": [
                "Ana Florence", "Andrew Chipper", "Barbora MacLean", "Brenda Stern",
                "Chandra MacFarland", "Daisy Studious", "Maja Ruoho",
            ],
        }
    )
    voice_aliases: dict[str, str] = field(
        default_factory=lambda: {"male": "Adde Michal", "female": "Ana Florence"}
    )
    default_voice: str = "Adde Michal"  # an XTTS speaker name from voice_groups (or "male"/"female")
    # Cloned voices inherit the accent of their reference clip (here: a Russian
    # speaker), so English cloned from it comes out accented. XTTS's style/accent
    # conditioning is separate from timbre, so the clone's style is blended
    # toward a built-in voice's. `accent_voice` is an XTTS speaker name (or
    # "male"/"female"), or None to follow the selected default voice. Strength
    # 0 disables the blend.
    accent_voice: str | None = None
    accent_strength: float = 0.7
    # When set, clone this reference speaker instead (stretch goal).
    speaker_wav: str | None = None

    def use_backend_voices(self) -> None:
        """Swaps in the preset-voice list for the selected backend. Call after
        `backend` is set, and before applying any --default-voice override."""
        if self.backend == "qwen3":
            self.voice_groups = {"Male": ["Aiden", "Ryan"]}
            self.voice_aliases = {"male": "Aiden"}
            self.default_voice = "Aiden"

    def all_voices(self) -> list[str]:
        return [name for names in self.voice_groups.values() for name in names]

    def resolve_voice(self, name: str) -> str:
        """Expands the "male"/"female" shorthands; anything else is returned as-is."""
        return self.voice_aliases.get(name.strip().lower(), name)
    # Where per-speaker reference clips are saved for reuse across sessions.
    voice_library_dir: str = "voices"


@dataclass
class PipelineConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
