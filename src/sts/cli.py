"""Entry point: `sts` starts the live pipeline, `sts --list-devices` lists sound cards."""

import argparse
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

from sts.audio.capture import list_devices
from sts.audio.devices import check_input, check_output
from sts.engine_args import add_engine_arguments, build_engine_config
from sts.log import vlog
from sts.persistent_config import PersistentConfig
from sts.startup_progress import StartupProgress
from sts.pipeline import Pipeline
from sts.web.server import CaptionBroadcaster, run_server

# Where --https-port looks for a cert/key when --tls-cert/--tls-key aren't
# given; a self-signed pair is made there on first start (see ensure_tls_cert).
DEFAULT_TLS_CERT = "certs/cert.pem"
DEFAULT_TLS_KEY = "certs/key.pem"
DEFAULT_WEB_PORT = 80
DEFAULT_HTTPS_PORT = 443


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
    parser.add_argument(
        "--web-port", type=int, default=DEFAULT_WEB_PORT,
        help=f"Caption page port, plain HTTP (default {DEFAULT_WEB_PORT})",
    )
    parser.add_argument(
        "--https-port",
        type=int,
        default=DEFAULT_HTTPS_PORT,
        help=f"Also serve the caption page over HTTPS on this port (default {DEFAULT_HTTPS_PORT}; 0 turns it "
        "off). Needed for the browser-mic input from other devices: browsers only allow the mic on a "
        "secure page (https, or localhost). Uses a self-signed cert, made on first start if it doesn't "
        "exist (with openssl); browsers show a one-time warning to click through per device.",
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
        "--final-after-unchanged",
        type=float,
        default=None,
        metavar="S",
        help="Finalize a caption sentence - send it on to translation and voicing, and never change it again - "
        "once it ends with a sentence-ending mark and hasn't changed (text or punctuation) for S seconds "
        "(default 5; 0 turns this rule off)",
    )
    parser.add_argument(
        "--final-force-after",
        type=float,
        default=None,
        metavar="S",
        help="Finalize a caption sentence S seconds after its first word, even if it's still changing "
        "(default 0: off; e.g. 30)",
    )
    parser.add_argument(
        "--final-after-silence",
        type=float,
        default=None,
        metavar="S",
        help="Finalize everything still open, end mark or not, once nothing new has been heard for S seconds "
        "(default 10; 0 turns this rule off)",
    )
    parser.add_argument(
        "--caption-interval-ms",
        type=int,
        default=None,
        help="How often the not-yet-locked audio is re-transcribed, updating the live caption (default 500)",
    )
    parser.add_argument(
        "--config-db",
        default="sts.db",
        help="SQLite file that remembers the operator's input/output device, output volume, and active voice "
        "across restarts (default: sts.db in the working directory). An explicit --input-device/--output-device/ "
        "--default-voice/--clone still wins over what's saved. Pass '' to disable persistence.",
    )
    return parser


def ensure_tls_cert(cert: str, key: str) -> bool:
    """Whether HTTPS can go ahead with this cert/key: they exist, or - the
    default location - a self-signed pair has just been made there. A
    cert/key named on the command line that's missing is an error."""
    if Path(cert).is_file() and Path(key).is_file():
        return True
    if (cert, key) != (DEFAULT_TLS_CERT, DEFAULT_TLS_KEY):
        print(f"error: didn't find the TLS cert/key {cert!r} / {key!r} for --https-port")
        sys.exit(1)
    Path(cert).parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", key, "-out", cert,
             "-days", "3650", "-subj", "/CN=sts"],
            check=True, capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"[web] couldn't make a self-signed certificate for HTTPS ({exc}); continuing without HTTPS")
        return False
    print(f"[web] made a self-signed certificate for HTTPS: {cert}")
    return True


def port_unavailable(host: str, port: int) -> str | None:
    """Why the caption page can't listen on `port` (in use, not allowed), or
    None. Checked up front: the server runs on a background thread, where
    failing to listen would just end that thread, silently."""
    try:
        with socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind((host, port))
    except OSError as exc:
        return exc.strerror or str(exc)
    return None


def page_hosts(web_host: str) -> list[str]:
    """Where the caption page can be opened: listening on every interface
    (0.0.0.0 / ::), each of this machine's IPv4 addresses, loopback first;
    otherwise just `web_host`."""
    if web_host not in ("0.0.0.0", "::", ""):
        return [web_host]
    try:
        import psutil

        addresses = {
            address.address
            for addresses in psutil.net_if_addrs().values()
            for address in addresses
            if address.family == socket.AF_INET
        }
    except Exception:
        return [web_host]
    return sorted(addresses, key=lambda a: (not a.startswith("127."), [int(part) for part in a.split(".")])) or [web_host]


def saved_device(persistent_config, key: str, check) -> int | None:
    """The audio device index saved under `key`, or None (the system default)
    if nothing is saved - or if what's saved can no longer be used: devices
    come and go (a USB interface unplugged, a Bluetooth headset off) and
    their indices shift, so a saved index can point at nothing at all, or at
    something that can't capture/play what the pipeline needs. Left
    unchecked that only surfaced as a PortAudioError from pipeline.start(),
    after every model had already loaded. The saved value is kept, not
    cleared, so the device is picked up again once it's back. Only ever used
    for a *saved* value - an explicit --input-device/--output-device is the
    operator asking for that exact device, so those still fail loudly."""
    if persistent_config is None:
        return None
    saved = persistent_config.get(key)
    if not saved or saved == "off":  # "off": that side is turned off (see main); no device to restore
        return None
    try:
        device = int(saved)
        check(device)
    except Exception as exc:
        print(f"[config] saved {key.replace('_', ' ')} {saved!r} can't be used ({exc}); using the system default")
        return None
    return device


def main() -> None:
    args = build_parser().parse_args()

    if args.list_devices:
        print(list_devices())
        return

    cfg, (asr_device, translation_device, tts_device) = build_engine_config(args)

    persistent_config = PersistentConfig(args.config_db) if args.config_db else None
    startup = StartupProgress(persistent_config)
    startup.begin("Starting…")
    if persistent_config is not None:
        saved_donor = persistent_config.get("qwen3_accent_donor")
        if saved_donor is not None and args.qwen_accent_donor is None:
            cfg.tts.qwen3_accent_donor = saved_donor  # set from the caption page last time
        saved_volume = persistent_config.get("output_volume")
        if saved_volume:
            try:
                cfg.audio.volume = max(0, min(100, int(saved_volume))) / 100
            except ValueError:
                pass

    # A side saved as "off" (clicked off in the caption page's device list)
    # starts off - unless the command line names a device for it.
    if args.input_device is None and persistent_config is not None and persistent_config.get("input_device") == "off":
        cfg.audio.source = "none"
    if args.output_device is None and persistent_config is not None and persistent_config.get("output_device") == "off":
        cfg.audio.output_enabled = False
    cfg.audio.input_device = (
        args.input_device
        if args.input_device is not None
        else saved_device(
            persistent_config, "input_device",
            lambda device: check_input(device, cfg.audio.sample_rate, cfg.audio.channels),
        )
    )
    if args.default_voice or args.clone:
        cfg.tts.startup_voice = None  # an explicit voice choice on the command line wins over the bundled Default
    cfg.audio.output_device = (
        args.output_device
        if args.output_device is not None
        else saved_device(persistent_config, "output_device", lambda device: check_output(device, 24_000))
    )
    if args.pause_s is not None:
        cfg.audio.pause_between_s = args.pause_s
    if args.final_after_unchanged is not None:
        cfg.asr.final_after_unchanged_seconds = max(0.0, args.final_after_unchanged)
    if args.final_force_after is not None:
        cfg.asr.final_force_after_seconds = max(0.0, args.final_force_after)
    if args.final_after_silence is not None:
        cfg.asr.final_after_silence_seconds = max(0.0, args.final_after_silence)
    if args.caption_interval_ms is not None:
        cfg.asr.interval_ms = max(100, args.caption_interval_ms)

    broadcaster = None
    if args.web:
        https_port = args.https_port or None  # 0: off
        if https_port and not ensure_tls_cert(args.tls_cert, args.tls_key):
            https_port = None
        port_error = port_unavailable(args.web_host, args.web_port)
        if port_error:
            print(f"error: can't serve the caption page on port {args.web_port} ({port_error}). "
                  f"Pick another with --web-port, e.g. --web-port 8765.")
            sys.exit(1)
        if https_port:
            port_error = port_unavailable(args.web_host, https_port)
            if port_error:
                print(f"[web] can't serve HTTPS on port {https_port} ({port_error}); continuing without it "
                      f"(pick another with --https-port, or 0 to turn it off)")
                https_port = None
        args.https_port = https_port
        broadcaster = CaptionBroadcaster()
        broadcaster.startup = startup
        broadcaster.https_port = args.https_port
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
        http_port = "" if args.web_port == 80 else f":{args.web_port}"
        for host in page_hosts(args.web_host):
            print(f"Captions: http://{host}{http_port}/")

    vlog("init", f"loading models (asr=gigaam on {asr_device}, translation on {translation_device}, "
         f"tts={cfg.tts.backend} on {tts_device})")
    pipeline = Pipeline(cfg, broadcaster=broadcaster, persistent_config=persistent_config, startup=startup)
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
        if persistent_config is not None:
            persistent_config.close()


if __name__ == "__main__":
    main()
