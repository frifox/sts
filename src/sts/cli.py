"""Entry point: `sts` starts the live pipeline, `sts --list-devices` lists sound cards."""

import argparse
import threading
import time

from sts.audio.capture import list_devices
from sts.config import PipelineConfig
from sts.pipeline import Pipeline
from sts.web.server import CaptionBroadcaster, run_server


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live Russian -> English speech pipeline")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    parser.add_argument("--input-device", type=int, default=None, help="Input device index")
    parser.add_argument("--output-device", type=int, default=None, help="Output device index")
    parser.add_argument(
        "--device", default="cuda", choices=["cuda", "cpu"], help="Compute device for all models"
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

    cfg = PipelineConfig()
    cfg.audio.input_device = args.input_device
    cfg.audio.output_device = args.output_device
    cfg.asr.device = args.device
    cfg.asr.compute_type = "float16" if args.device == "cuda" else "int8"
    cfg.asr.model_size = args.whisper_model
    cfg.translation.device = args.device
    cfg.tts.device = args.device
    cfg.tts.backend = args.tts_backend
    cfg.tts.voice_library_dir = args.voice_library_dir

    broadcaster = None
    if args.web:
        broadcaster = CaptionBroadcaster()
        threading.Thread(
            target=run_server, args=(broadcaster, args.web_host, args.web_port), daemon=True
        ).start()
        print(f"Captions: http://{args.web_host}:{args.web_port}/")

    pipeline = Pipeline(cfg, broadcaster=broadcaster)
    if args.clone:
        pipeline.clone_voice(args.clone)
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
