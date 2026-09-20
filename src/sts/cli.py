"""Entry point: `sts` starts the live pipeline, `sts --list-devices` lists sound cards."""

import argparse
import time

from sts.audio.capture import list_devices
from sts.config import PipelineConfig
from sts.pipeline import Pipeline


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

    pipeline = Pipeline(cfg)
    if args.clone:
        pipeline.clone_voice(args.clone)

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
