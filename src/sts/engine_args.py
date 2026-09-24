"""Command-line options and config building shared by `sts` (live) and
`translate.py` (offline video): which models to run, on what device, and how
the TTS voice behaves."""

import argparse
import importlib.util

from sts.config import PipelineConfig
from sts.devices import detect_device, resolve_devices
from sts.log import set_verbose


def add_engine_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--device",
        default=None,
        choices=["cuda", "cpu", "mps"],
        help="Compute device for translation/TTS, and ASR unless --asr-device overrides it "
        "('mps' targets Apple Silicon GPUs). Default: auto-detect (cuda, then mps, then cpu)",
    )
    parser.add_argument(
        "--asr-device",
        default=None,
        help="Override just the ASR device: 'cpu' (e.g. to free VRAM for translation/TTS), or a specific "
        "GPU like 'cuda:1' if you have more than one card. faster-whisper only supports cpu/cuda, never mps.",
    )
    parser.add_argument(
        "--translation-device",
        default=None,
        help="Override just the translation device, e.g. 'cuda:1' to run it on a second GPU instead of "
        "sharing one with ASR/TTS. Each stage is an independent model - this pins the whole stage to a "
        "card, not model parallelism, so there's no NVLink/bandwidth requirement between cards.",
    )
    parser.add_argument(
        "--tts-device",
        default=None,
        help="Override just the TTS device, e.g. 'cuda:1' to run voice synthesis on a second GPU instead "
        "of sharing one with ASR/translation.",
    )
    parser.add_argument("--whisper-model", default="large-v3", help="faster-whisper model size/name")
    parser.add_argument(
        "--asr-compute-type",
        default=None,
        help="faster-whisper quantization. Default: float16 on GPU, int8 on CPU. 'int8_float16' (GPU only) "
        "trades a little accuracy for a smaller VRAM footprint and faster inference - useful when ASR has "
        "to share an 8GB card with translation and TTS.",
    )
    parser.add_argument(
        "--translation-model",
        default=None,
        metavar="HF_MODEL",
        help="Hugging Face NLLB model used for translation (default: facebook/nllb-200-distilled-1.3B). "
        "e.g. facebook/nllb-200-3.3B is larger and slower (about 2x the memory: ~13GB in fp32 on mps) "
        "for a modest quality gain; facebook/nllb-200-distilled-600M is smaller and faster.",
    )
    parser.add_argument(
        "--tts-backend",
        default=None,
        choices=["xtts", "qwen3"],
        help="Voice cloning/synthesis backend. Default: whichever is installed in the current Python "
        "environment (xtts needs the 'xtts' extra, qwen3 the 'qwen3' extra; they conflict, so each "
        "lives in its own virtualenv)",
    )
    parser.add_argument(
        "--default-voice",
        default=None,
        help="Built-in (non-cloned) voice used when no trained/cloned voice is active: an XTTS speaker name, "
        "e.g. 'Craig Gutsy', or male/female (xtts default: Adde Michal; qwen3: Aiden, Ryan)",
    )
    parser.add_argument(
        "--accent-voice",
        default=None,
        help="Built-in voice whose accent cloned voices are steered toward: male/female, or any XTTS "
        "speaker name such as 'Andrew Chipper' (default: the selected --default-voice)",
    )
    parser.add_argument(
        "--accent-strength",
        type=float,
        default=0.7,
        help="How far a cloned voice's accent is pulled toward the accent voice: 0 disables, 1 fully adopts it, "
        "above 1 exaggerates it (can get unstable) (default 0.7)",
    )
    parser.add_argument(
        "--qwen-temperature",
        type=float,
        default=None,
        help="Qwen3-TTS sampling temperature: lower = more consistent intonation between sentences, "
        "higher = more varied (default 0.6; the model's own default is 0.9)",
    )
    parser.add_argument(
        "--qwen-clone-mode",
        choices=["auto", "icl", "xvector"],
        default=None,
        help="How Qwen3 clones a saved voice: 'icl' = reference clip + transcript (closest match, but can skip "
        "words), 'xvector' = speaker embedding only (steadier), 'auto' = xvector unless the transcript is English (default)",
    )
    parser.add_argument(
        "--qwen-preload-presets",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Load Qwen3-TTS's CustomVoice preset model at start-up alongside the base cloning model "
        "(needed for the preset speaker dropdown). --no-qwen-preload-presets saves several GB of VRAM "
        "when only a cloned/trained voice is used throughout, e.g. a live service on the bundled "
        "Default voice (default: on)",
    )
    parser.add_argument(
        "--qwen-quality-retries",
        type=int,
        default=None,
        help="How many times Qwen3 may regenerate a take that sounds duller (muffled, distant) than that "
        "voice's usual; each retry costs a full extra generation, so it adds latency (default 1 live, 2 for "
        "translate.py; 0 turns it off)",
    )
    parser.add_argument(
        "--sentence-pause-s",
        type=float,
        default=None,
        help="Silence between the sentences of one spoken translation (seconds, default 0.35)",
    )
    parser.add_argument(
        "--voice-library-dir",
        default="voices",
        help="Directory where trained/selectable speaker voices are saved",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Log pipeline activity to stdout: speech segments, ASR/translation/TTS timings, audio device events",
    )


# The Python module each backend's extra installs.
_BACKEND_MODULES = {"xtts": "TTS", "qwen3": "qwen_tts"}


def installed_backends() -> list[str]:
    return [name for name, module in _BACKEND_MODULES.items() if importlib.util.find_spec(module) is not None]


def default_tts_backend(prefer: str = "xtts") -> str:
    """The TTS backend to use when --tts-backend isn't given: the one installed in
    this environment. If both are (unusual: their transformers requirements
    conflict) or neither is, `prefer` decides (the latter then fails with an
    install hint when the backend loads)."""
    installed = installed_backends()
    return installed[0] if len(installed) == 1 else prefer


def build_engine_config(args: argparse.Namespace, prefer_backend: str = "xtts") -> tuple[PipelineConfig, tuple[str, str, str]]:
    """Applies the shared options to a fresh PipelineConfig. Returns it together
    with the resolved (asr, translation, tts) devices."""
    set_verbose(args.verbose)

    # transformers' generation code emits accurate but purely informational
    # logger.warning_once() calls (a v4.50 deprecation heads-up, an
    # attention-mask reminder) on every fresh model load and every so often
    # thereafter (e.g. XTTS's GPT2-based model on each Reset); they add
    # nothing to the live journal, and this is transformers' own documented
    # verbosity control rather than a stream-redirection hack.
    try:
        import transformers

        transformers.utils.logging.set_verbosity_error()
    except ImportError:
        pass
    devices = resolve_devices(
        args.device or detect_device(), args.asr_device, args.translation_device, args.tts_device
    )
    asr_device, translation_device, tts_device = devices

    cfg = PipelineConfig()
    cfg.asr.device = asr_device
    cfg.asr.compute_type = args.asr_compute_type or ("float16" if asr_device == "cuda" else "int8")
    cfg.asr.model_size = args.whisper_model
    cfg.translation.device = translation_device
    if args.translation_model:
        cfg.translation.model_name = args.translation_model
    cfg.tts.device = tts_device
    cfg.tts.backend = args.tts_backend or default_tts_backend(prefer_backend)
    if not args.tts_backend:
        print(f"[tts] using the {cfg.tts.backend} backend (the one installed in this environment)", flush=True)
    cfg.tts.use_backend_voices()
    cfg.tts.voice_library_dir = args.voice_library_dir
    if args.qwen_clone_mode is not None:
        cfg.tts.qwen3_clone_mode = args.qwen_clone_mode
    if args.qwen_preload_presets is not None:
        cfg.tts.qwen3_preload_presets = args.qwen_preload_presets
    if args.qwen_quality_retries is not None:
        cfg.tts.qwen3_quality_retries = max(0, args.qwen_quality_retries)
    if args.qwen_temperature is not None:
        cfg.tts.qwen3_temperature = args.qwen_temperature
    if args.sentence_pause_s is not None:
        cfg.audio.pause_within_s = args.sentence_pause_s
    if args.default_voice:
        cfg.tts.default_voice = cfg.tts.resolve_voice(args.default_voice)
        if cfg.tts.default_voice not in cfg.tts.all_voices():
            raise SystemExit(
                f"unknown --default-voice {args.default_voice!r}; choose from: {', '.join(cfg.tts.all_voices())}"
            )
    cfg.tts.accent_voice = args.accent_voice
    cfg.tts.accent_strength = args.accent_strength
    return cfg, devices
