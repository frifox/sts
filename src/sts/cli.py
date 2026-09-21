"""Entry point: `sts` starts the live pipeline, `sts --list-devices` lists sound cards."""

import argparse
import threading
import time

from sts.audio.capture import list_devices
from sts.engine_args import add_engine_arguments, build_engine_config
from sts.log import vlog
from sts.pipeline import Pipeline
from sts.web.server import CaptionBroadcaster, run_server


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live Russian -> English speech pipeline")
    add_engine_arguments(parser)
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    parser.add_argument("--input-device", type=int, default=None, help="Input device index")
    parser.add_argument("--output-device", type=int, default=None, help="Output device index")
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
        "--web",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Serve the live captions page and operator controls over HTTP (on by default; --no-web disables)",
    )
    parser.add_argument("--web-host", default="0.0.0.0", help="Caption web server bind address")
    parser.add_argument("--web-port", type=int, default=8765, help="Caption web server port")
    parser.add_argument(
        "--pause-s",
        type=float,
        default=None,
        help="Minimum silence between consecutive spoken translations (seconds, default 1.0; 0 disables)",
    )
    parser.add_argument(
        "--min-silence-ms",
        type=int,
        default=None,
        help="Pause length (ms) that ends a caption segment; lower = shorter, more frequent captions (default 700)",
    )
    parser.add_argument(
        "--max-segment-s",
        type=float,
        default=None,
        help="Longest a single caption segment may run (seconds) before it is cut regardless of pauses (default 20)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.list_devices:
        print(list_devices())
        return

    cfg, (asr_device, translation_device, tts_device) = build_engine_config(args)
    cfg.audio.input_device = args.input_device
    if args.default_voice or args.clone:
        cfg.tts.startup_voice = None  # an explicit voice choice on the command line wins over the bundled Default
    cfg.audio.output_device = args.output_device
    if args.pause_s is not None:
        cfg.audio.pause_between_s = args.pause_s
    if args.min_silence_ms is not None:
        cfg.vad.min_silence_ms = args.min_silence_ms
    if args.max_segment_s is not None:
        cfg.vad.max_segment_ms = int(args.max_segment_s * 1000)

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
