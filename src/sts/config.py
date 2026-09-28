"""Central configuration for the pipeline. Values can be overridden via CLI flags."""

from dataclasses import dataclass, field


@dataclass
class AudioConfig:
    input_device: int | None = None
    output_device: int | None = None
    # Software gain applied to audio written to the local sound card only (not
    # to a browser stream, which has its own volume) - set from the Output
    # dropdown's slider, 0.0-1.0.
    volume: float = 1.0
    # "alsa" reads `input_device` via sounddevice, as always; "browser" instead
    # accepts frames pushed over the websocket by a connected browser's
    # microphone (see Pipeline._handle_mic_frame); "none" listens to nothing
    # (the local input turned off). Switched at runtime by the caption page's
    # "Use this browser's mic" control / the input device list.
    source: str = "alsa"
    # False: nothing plays on the local output device - the voice still goes
    # to browsers streaming it, paced as if it played (see AudioPlayback.play).
    output_enabled: bool = True
    sample_rate: int = 16_000
    channels: int = 1
    # Minimum silence between one translated text and the next when they play
    # back to back (e.g. while playback is catching up).
    pause_between_s: float = 1.0
    # Silence between the sentences of one spoken text (each sentence is
    # synthesized separately so it gets its own natural intonation).
    pause_within_s: float = 0.35
    # Catch-up: when a sentence starts being spoken `catchup_start_s` or more
    # after its caption became final, playback is sped up - pitch preserved -
    # until a sentence starts within `catchup_stop_s`, then returns to normal
    # speed. (Seconds, not sentences; the time before a caption is final
    # doesn't count.) The speed-up grows with the lateness, from
    # `catchup_min_rate` up to `catchup_max_rate` at `catchup_full_s` late
    # or more, and is chosen once per sentence. `catchup_max_rate` of 1.0
    # disables it.
    catchup_start_s: float = 8.0
    catchup_stop_s: float = 3.0
    catchup_full_s: float = 20.0
    catchup_min_rate: float = 1.1
    # Temporarily disabled (1.0 = never speed up): all speech at normal speed.
    # 1.35 re-enables it with the thresholds above.
    catchup_max_rate: float = 1.0
    # Last resort, for when even full speed-up can't keep up (the English
    # runs much longer than the Russian for a long stretch): once this many
    # seconds of synthesized speech are waiting to play, the oldest waiting
    # sentences are skipped. None never skips.
    max_backlog_s: float | None = 60.0
    block_ms: int = 32  # frame fed to the VAD; Silero needs exactly 512 samples at 16 kHz (32 ms)


@dataclass
class VADConfig:
    """Silero VAD, which only collects the live speaker's speech for "Train
    voice" (captions don't depend on pauses - see ASRConfig)."""

    # Silero VAD speech probability threshold.
    threshold: float = 0.5
    # How much trailing silence (ms) closes out a speech segment.
    min_silence_ms: int = 500
    # Segments shorter than this are dropped as noise.
    min_speech_ms: int = 250
    # Hard cap on one segment's length.
    max_segment_ms: int = 20_000


@dataclass
class ASRConfig:
    device: str = "cuda"
    # Live captions (sts.transcript): what isn't final yet is re-transcribed
    # every `interval_ms`. A sentence is final - sent on to translation and
    # voicing, and never changed again - once it ends with a sentence-ending
    # mark and its text (punctuation included) hasn't changed for
    # `final_after_unchanged_seconds`, or `final_force_after_seconds` after
    # its first word however much it's still changing. Once nothing new has
    # been heard for `final_after_silence_seconds`, everything still open is
    # final, end mark or not. 0 turns any of these off.
    interval_ms: int = 500
    final_after_unchanged_seconds: float = 5.0
    final_force_after_seconds: float = 0.0  # off for now; 30 caps a run-on sentence's wait
    final_after_silence_seconds: float = 10.0


@dataclass
class TranslationConfig:
    # MADLAD-400 checkpoint (see sts.translate.translator). bf16 memory:
    # 10b ~21GB, 7b ~15GB, 3b ~6GB.
    model_name: str = "google/madlad400-10b-mt"
    target_lang: str = "en"  # MADLAD's <2xx> target tag; the source language is detected
    device: str = "cuda"
    # Off: the placeholder masking (sts.translate.glossary) was built for
    # NLLB. MADLAD misspells the placeholders, so every masked sentence fell
    # back to a second, plain translation (twice the time, same output), and
    # masking words mid-sentence breaks its grammar and its recall of Bible
    # wording anyway. MADLAD renders the glossary's terms correctly on its own.
    apply_glossary: bool = False


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
    # Qwen3 sometimes speaks the start and end of a sentence and skips the middle
    # (long input makes it likelier). Each translated sentence is generated whole
    # - one generation, however long - and one whose audio is implausibly short
    # (or long) for its length is regenerated, up to `qwen3_max_retries` times.
    # Length is judged per *character*: measured on a cloned voice it varied
    # 0.055-0.079 s/char across sentences, far steadier than per word (0.23-0.68)
    # or per syllable. Once a voice has a track record, its own pace sets the bar.
    qwen3_min_seconds_per_char: float = 0.04   # absolute floor (typical is about 0.065)
    qwen3_max_seconds_per_char: float = 0.16
    # How a reference clip conditions the voice. "icl" feeds the clip *and its
    # transcript* to the model - even a Russian transcript for English speech;
    # "xvector" uses only a speaker embedding. xvector used to be picked for
    # non-English transcripts, for fear of skipped words (none in 64 test takes),
    # but it carries nothing of how the reference was spoken or recorded, and
    # with it about 1 statement in 5 ended on a non-falling, "to be continued"
    # pitch, against about 1 in 16 with icl. ("auto" = "icl".) The whole
    # sentence is always given to the model up front (non_streaming_mode), not
    # fed in as it speaks, so it knows where the sentence ends.
    qwen3_clone_mode: str = "icl"
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
    # Equalize each take gently toward the voice's recent average tonal balance
    # (sts.audioquality.match_tone): evens out takes that sound closer to or further
    # from the mic than the one before.
    qwen3_tone_match: bool = True
    # A statement ends on a falling pitch; now and then Qwen3 ends one level or
    # rising, and it sounds unfinished. A sentence ending in "." whose pitch over its
    # last 0.35 s isn't at least `-qwen3_ending_fall_st` semitones below its median is
    # regenerated (up to `qwen3_ending_retries` times; about 1 sentence in 10 with
    # icl cloning) and the take ending lowest kept. 0 turns this off.
    qwen3_ending_retries: int = 1
    qwen3_ending_fall_st: float = -1.0
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
