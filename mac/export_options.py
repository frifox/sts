"""Writes sts's command-line options as JSON for the app's Configuration
form: run by build.sh with the bundled Python, so the form always matches
the sts that's bundled."""

import argparse
import json
import sys

from sts.cli import build_parser

# Set by the app itself, or meaningless inside it (only MLX is installed).
HIDDEN = {
    "help", "list_devices", "web", "config_db", "voice_library_dir", "tls_cert", "tls_key",
    "tts_backend", "qwen_engine",
}

# What sts does when an option whose argparse default is None is left out
# (the help text says it in prose). An option missing from here and from
# argparse's defaults fails the build, so new options get one.
DEFAULTS = {
    "device": "auto (mps on this Mac)",
    "asr_device": "same as --device",
    "translation_device": "same as --device",
    "tts_device": "same as --device",
    "translation_model": "google/madlad400-10b-mt",
    "default_voice": "the bundled Default voice",
    "accent_voice": "the default voice",
    "qwen_temperature": "0.6",
    "qwen_accent_donor": "none, or last set on the page",
    "qwen_clone_mode": "auto",
    "qwen_preload_presets": "on",
    "qwen_quality_retries": "0 (off)",
    "sentence_pause_s": "0.35",
    "input_device": "last used, or system default",
    "output_device": "last used, or system default",
    "clone": "none",
    "clone_name": "none",
    "pause_s": "1.0",
    "final_after_unchanged": "3",
    "final_force_after": "0 (off)",
    "final_after_silence": "7",
    "caption_interval_ms": "500",
    "web_host": "127.0.0.1 (this computer only; see --all-interfaces)",
}


def described_default(action: argparse.Action) -> str:
    if isinstance(action, argparse._StoreTrueAction):
        return "off"
    if action.default not in (None, argparse.SUPPRESS):
        if isinstance(action.default, bool):
            return "on" if action.default else "off"
        return str(action.default)
    if action.dest not in DEFAULTS:
        sys.exit(f"export_options.py: say what --{action.dest.replace('_', '-')} defaults to in DEFAULTS")
    return DEFAULTS[action.dest]


def main() -> None:
    options = []
    for action in build_parser()._actions:
        if action.dest in HIDDEN or not action.option_strings:
            continue
        flag = max(action.option_strings, key=len)  # --verbose, not -v
        if isinstance(action, argparse.BooleanOptionalAction):
            kind = "optional-bool"
            flag = next(s for s in action.option_strings if not s.startswith("--no-"))
        elif isinstance(action, argparse._StoreTrueAction):
            kind = "flag"
        elif action.choices:
            kind = "choice"
        else:
            kind = "text"
        options.append({
            "key": action.dest,
            "flag": flag,
            "kind": kind,
            "choices": [str(c) for c in action.choices or []],
            "type": action.type.__name__ if action.type in (int, float) else "str",
            "default": described_default(action),
            "help": " ".join((action.help or "").split()),
        })
    json.dump(options, sys.stdout, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
