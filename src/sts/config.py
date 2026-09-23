"""Central configuration for the pipeline. Values can be overridden via CLI flags."""

from dataclasses import dataclass, field


@dataclass
class AudioConfig:
    input_device: int | None = None
    output_device: int | None = None
    # "alsa" reads `input_device` via sounddevice, as always; "browser" instead
    # accepts frames pushed over the websocket by a connected browser's
    # microphone (see Pipeline._handle_mic_frame). Switched at runtime by the
    # caption page's "Use this browser's mic" control / the input dropdown.
    source: str = "alsa"
    sample_rate: int = 16_000
    channels: int = 1
    # Minimum silence between one translated text and the next when they play
    # back to back (e.g. while playback is catching up).
    pause_between_s: float = 1.0
    # Silence between the sentences of one spoken text (each sentence is
    # synthesized separately so it gets its own natural intonation).
    pause_within_s: float = 0.35
    block_ms: int = 32  # frame fed to the VAD; Silero needs exactly 512 samples at 16 kHz (32 ms)


@dataclass
class VADConfig:
    # Silero VAD speech probability threshold.
    threshold: float = 0.5
    # How much trailing silence (ms) closes out a speech segment.
    min_silence_ms: int = 500
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
    # Load the preset-voice model at start-up (live use, where the preset is the
    # default voice). False loads it only if it is actually needed, which saves
    # several GB when a cloned voice is used throughout (video translation).
    qwen3_preload_presets: bool = True
    # VoiceDesign variant: invents a voice from a text description (the "Create
    # Voice" dialog). Loaded only while that dialog is in use.
    qwen3_design_model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    # Qwen3-TTS samples its output randomly (model default temperature 0.9), so
    # intonation differs from one sentence to the next. Lower = steadier, more
    # uniform delivery; too low can sound flat or loop. Applies to both the
    # main generator and the "sub-talker" that shapes fine acoustic detail.
    qwen3_temperature: float = 0.6
    qwen3_top_k: int = 30
    # Qwen3 sometimes speaks the start and end of a sentence and skips the middle.
    # Long input makes it likelier, so text is synthesized in pieces of at most
    # this many characters, and a piece whose audio is implausibly short (or
    # long) for its length is regenerated, up to `qwen3_max_retries` times.
    # Length is judged per *character*: measured on a cloned voice it varied
    # 0.055-0.079 s/char across sentences, far steadier than per word (0.23-0.68)
    # or per syllable. Once a voice has a track record, its own pace sets the bar.
    qwen3_max_chars: int = 140
    qwen3_min_seconds_per_char: float = 0.04   # absolute floor (typical is about 0.065)
    qwen3_max_seconds_per_char: float = 0.16
    # How a reference clip conditions the voice. "icl" feeds the clip *and its
    # transcript* to the model (closest voice match, but a long or foreign-language
    # transcript is a common cause of skipped speech); "xvector" uses only a speaker
    # embedding (steadier, slightly less faithful). "auto" picks xvector when the
    # transcript isn't English (e.g. a Russian speaker's clip) and icl otherwise.
    qwen3_clone_mode: str = "auto"
    qwen3_max_retries: int = 2
    # Voice cloning also copies a reference's recording conditions, and even a good
    # reference sometimes yields a dull, distant-sounding take. A take whose clarity
    # (see sts.audioquality) is more than `qwen3_clarity_margin_db` below that voice's
    # own typical take is regenerated (up to `qwen3_quality_retries` times) and the
    # clearest take kept. There is deliberately no absolute floor: clarity depends on the
    # voice (a deep voice measures duller than a bright one) and on the sentence, so a
    # take is only ever compared with that voice's own previous takes, and nothing is
    # judged until it has 3 of them. 0 retries turns this off.
    qwen3_quality_retries: int = 0  # off by default: each retry is a full extra generation, and it did not fix what listeners noticed
    qwen3_clarity_margin_db: float = 4.0
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
    # Saved voice (library id) that is active at start-up and after Reset: the
    # bundled "Default" voice. None starts on the built-in `default_voice` instead.
    startup_voice: str | None = "default"
    # Speech level every spoken clip is matched to (RMS of its louder frames; 0.10 is about -20 dBFS,
    # a normal speech level). The models speak short phrases quieter than long sentences, which
    # sounds distant. None leaves the clips as generated.
    target_level: float | None = 0.10
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
