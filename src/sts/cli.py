"""Entry point: `sts` starts the live pipeline, `sts --list-devices` lists sound cards."""

import argparse
import sys
import threading
import time
from pathlib import Path

from sts.audio.capture import list_devices
from sts.engine_args import add_engine_arguments, build_engine_config
from sts.log import vlog
from sts.pipeline import Pipeline
from sts.web.server import CaptionBroadcaster, run_server

# Where --https-port looks for a cert/key when --tls-cert/--tls-key aren't
# given - generate one with the command the --https-port help text prints.
DEFAULT_TLS_CERT = "certs/cert.pem"
DEFAULT_TLS_KEY = "certs/key.pem"


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
        "--https-port",
        type=int,
        default=None,
        help="Also serve the caption page over HTTPS on this port, alongside plain HTTP on "
        "--web-port. Needed for the browser-mic input: getUserMedia only works from a secure "
        "context, which a plain http:// LAN address isn't. Uses a self-signed cert by default "
        f"(generate one with: openssl req -x509 -newkey rsa:2048 -nodes -keyout {DEFAULT_TLS_KEY} "
        f"-out {DEFAULT_TLS_CERT} -days 3650 -subj '/CN=sts'); browsers will show a one-time "
        "warning to click through per device.",
    )
    parser.add_argument(
        "--tls-cert", default=DEFAULT_TLS_CERT, help=f"TLS certificate for --https-port (default: {DEFAULT_TLS_CERT})"
    )
    parser.add_argument(
        "--tls-key", default=DEFAULT_TLS_KEY, help=f"TLS private key for --https-port (default: {DEFAULT_TLS_KEY})"
    )
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
        if args.https_port is not None:
            if not Path(args.tls_cert).is_file() or not Path(args.tls_key).is_file():
                print(
                    f"error: --https-port needs a cert and key; didn't find both "
                    f"{args.tls_cert!r} and {args.tls_key!r}. Generate a self-signed pair with:\n"
                    f"  openssl req -x509 -newkey rsa:2048 -nodes -keyout {args.tls_key} "
                    f"-out {args.tls_cert} -days 3650 -subj '/CN=sts'"
                )
                sys.exit(1)
        broadcaster = CaptionBroadcaster()
        threading.Thread(
            target=run_server,
            args=(broadcaster, args.web_host, args.web_port),
            kwargs={
                "https_port": args.https_port,
                "ssl_certfile": args.tls_cert if args.https_port else None,
                "ssl_keyfile": args.tls_key if args.https_port else None,
            },
            daemon=True,
        ).start()
        print(f"Captions: http://{args.web_host}:{args.web_port}/")
        if args.https_port is not None:
            print(f"Captions (HTTPS, needed for the browser mic): https://{args.web_host}:{args.https_port}/")

    vlog("init", f"loading models (whisper={cfg.asr.model_size} on {asr_device}, translation on {translation_device}, "
         f"tts={cfg.tts.backend} on {tts_device})")
    pipeline = Pipeline(cfg, broadcaster=broadcaster)
    vlog("init", "models loaded")
    if args.clone:
        pipeline.clone_voice(args.clone, name=args.clone_name)
    if broadcaster:
        broadcaster.on_control = pipeline.handle_control
        broadcaster.on_mic_frame = pipeline._handle_mic_frame

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
