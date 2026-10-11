"""Voices made from the terminal and in STS.app are the same library on macOS."""

from pathlib import Path

from sts import paths


def test_on_macos_the_voice_library_is_the_app_s(monkeypatch):
    monkeypatch.setattr(paths.sys, "platform", "darwin")
    assert paths.default_voice_library_dir() == str(Path.home() / "Library/Application Support/STS/voices")


def test_elsewhere_it_is_voices_in_the_working_directory(monkeypatch):
    monkeypatch.setattr(paths.sys, "platform", "linux")
    assert paths.default_voice_library_dir() == "voices"


def test_the_flag_still_wins():
    import argparse

    from sts.engine_args import add_engine_arguments, build_engine_config

    parser = argparse.ArgumentParser()
    add_engine_arguments(parser)
    cfg, _ = build_engine_config(parser.parse_args(["--device", "cpu", "--voice-library-dir", "/tmp/elsewhere"]))
    assert cfg.tts.voice_library_dir == "/tmp/elsewhere"
