"""Entry point: `sts` starts the live pipeline, `sts --list-devices` lists sound cards."""

import argparse
import threading
import time

from sts.audio.capture import list_devices
from sts.config import PipelineConfig
from sts.devices import detect_device, resolve_devices
from sts.log import set_verbose, vlog
from sts.pipeline import Pipeline
from sts.web.server import CaptionBroadcaster, run_server


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live Russian -> English speech pipeline")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    parser.add_argument("--input-device", type=int, default=None, help="Input device index")
    parser.add_argument("--output-device", type=int, default=None, help="Output device index")
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
    parser.add_argument(
        "--whisper-model", default="large-v3", help="faster-whisper model size/name"
    )
    parser.add_argument(
        "--clone",
        metavar="WAV_PATH",
        default=None,
        help="Reference clip (10-20s) of the speaker to clone their voice for TTS output",
    )
    parser.add_argument(
        "--clone-name",
        default=None,
        help="Save the --clone reference clip into the voice library under this name, so it's "
        "reusable from the caption page's speaker dropdown in later sessions instead of just "
        "this run. Zero-shot cloning is often good enough on its own - this just means not "
        "reprocessing the file every time.",
    )
    parser.add_argument(
        "--web", action="store_true", help="Serve live original/translated captions over HTTP"
    )
    parser.add_argument("--web-host", default="0.0.0.0", help="Caption web server bind address")
    parser.add_argument("--web-port", type=int, default=8765, help="Caption web server port")
    parser.add_argument(
        "--tts-backend",
        default="xtts",
        choices=["xtts", "qwen3"],
        help="Voice cloning/synthesis backend (qwen3 needs the 'qwen3' extra: pip install -e '.[qwen3]')",
    )
    parser.add_argument(
        "--default-voice",
        default="male",
        choices=["male", "female"],
        help="Built-in (non-cloned) voice used when no trained/cloned voice is active (xtts backend)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Log pipeline activity to stdout: speech segments, ASR/translation/TTS timings, audio device events",
    )
    parser.add_argument(
        "--voice-library-dir",
        default="voices",
        help="Directory where trained/selectable speaker voices are saved",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.list_devices:
        print(list_devices())
        return

    set_verbose(args.verbose)
    asr_device, translation_device, tts_device = resolve_devices(args.device or detect_device(), args.asr_device)

    cfg = PipelineConfig()
    cfg.audio.input_device = args.input_device
    cfg.audio.output_device = args.output_device
    cfg.asr.device = asr_device
    cfg.asr.compute_type = "float16" if asr_device == "cuda" else "int8"
    cfg.asr.model_size = args.whisper_model
    cfg.translation.device = translation_device
    cfg.tts.device = tts_device
    cfg.tts.backend = args.tts_backend
    cfg.tts.voice_library_dir = args.voice_library_dir
    cfg.tts.default_voice = args.default_voice

    broadcaster = None
    if args.web:
        broadcaster = CaptionBroadcaster()
        threading.Thread(
            target=run_server, args=(broadcaster, args.web_host, args.web_port), daemon=True
        ).start()
        print(f"Captions: http://{args.web_host}:{args.web_port}/")

    vlog("init", f"loading models (whisper={cfg.asr.model_size} on {asr_device}, translation on {translation_device}, "
         f"tts={cfg.tts.backend} on {tts_device})")
    pipeline = Pipeline(cfg, broadcaster=broadcaster)
    vlog("init", "models loaded")
    if args.clone:
        pipeline.clone_voice(args.clone, name=args.clone_name)
    if broadcaster:
        broadcaster.on_control = pipeline.handle_control

    pipeline.start()
    print("Pipeline running. Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
