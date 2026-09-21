"""Command-line options and config building shared by `sts` (live) and
`translate.py` (offline video): which models to run, on what device, and how
the TTS voice behaves."""

import argparse

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
        help="Override just the ASR device (e.g. force 'cpu' when --device is cuda/mps for some other reason). "
        "faster-whisper only supports cpu/cuda, never mps.",
    )
    parser.add_argument("--whisper-model", default="large-v3", help="faster-whisper model size/name")
    parser.add_argument(
        "--tts-backend",
        default="xtts",
        choices=["xtts", "qwen3"],
        help="Voice cloning/synthesis backend (qwen3 needs the 'qwen3' extra: pip install -e '.[qwen3]')",
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


def build_engine_config(args: argparse.Namespace) -> tuple[PipelineConfig, tuple[str, str, str]]:
    """Applies the shared options to a fresh PipelineConfig. Returns it together
    with the resolved (asr, translation, tts) devices."""
    set_verbose(args.verbose)
    devices = resolve_devices(args.device or detect_device(), args.asr_device)
    asr_device, translation_device, tts_device = devices

    cfg = PipelineConfig()
    cfg.asr.device = asr_device
    cfg.asr.compute_type = "float16" if asr_device == "cuda" else "int8"
    cfg.asr.model_size = args.whisper_model
    cfg.translation.device = translation_device
    cfg.tts.device = tts_device
    cfg.tts.backend = args.tts_backend
    cfg.tts.use_backend_voices()
    cfg.tts.voice_library_dir = args.voice_library_dir
    if args.qwen_clone_mode is not None:
        cfg.tts.qwen3_clone_mode = args.qwen_clone_mode
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
