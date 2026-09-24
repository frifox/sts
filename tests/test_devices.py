import os

from sts.devices import resolve_devices, supports_bf16_and_flash_attention


def test_cuda_maps_all_components_to_cuda():
    assert resolve_devices("cuda", None) == ("cuda", "cuda", "cuda")


def test_cpu_maps_all_components_to_cpu():
    assert resolve_devices("cpu", None) == ("cpu", "cpu", "cpu")


def test_mps_maps_all_components_to_mps():
    # ASR included: Parakeet is PyTorch, unlike the faster-whisper this once
    # had to route to cpu for lack of any Metal support.
    assert resolve_devices("mps", None) == ("mps", "mps", "mps")


def test_mps_sets_pytorch_fallback_env_var():
    os.environ.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
    resolve_devices("mps", None)
    assert os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"


def test_asr_device_override_is_respected_for_cuda():
    assert resolve_devices("cuda", "cpu") == ("cpu", "cuda", "cuda")


def test_asr_device_override_is_respected_for_mps():
    asr, translation, tts = resolve_devices("mps", "cuda")
    assert asr == "cuda"
    assert translation == "mps"
    assert tts == "mps"


def test_translation_and_tts_device_overrides_pin_independent_gpus():
    # Each stage is an independent model, so a second card is just another
    # override, same shape as --asr-device - no model-parallelism needed.
    asr, translation, tts = resolve_devices("cuda", None, "cuda:1", "cuda:1")
    assert asr == "cuda"
    assert translation == "cuda:1"
    assert tts == "cuda:1"


def test_all_three_stages_can_be_pinned_to_different_devices():
    asr, translation, tts = resolve_devices("cuda", "cpu", "cuda:0", "cuda:1")
    assert (asr, translation, tts) == ("cpu", "cuda:0", "cuda:1")


def test_translation_and_tts_overrides_apply_under_mps_too():
    asr, translation, tts = resolve_devices("mps", None, "cuda:1", "cuda:1")
    assert asr == "mps"
    assert translation == "cuda:1"
    assert tts == "cuda:1"


def test_no_overrides_still_defaults_every_stage_to_the_single_shared_device():
    # The default (no second card): unchanged behavior, everything on one GPU.
    assert resolve_devices("cuda", None, None, None) == ("cuda", "cuda", "cuda")


def test_pascal_gtx_1080_ti_does_not_support_bf16_or_flash_attention():
    assert supports_bf16_and_flash_attention(6) is False  # compute capability 6.1


def test_turing_does_not_support_bf16_or_flash_attention():
    assert supports_bf16_and_flash_attention(7) is False  # compute capability 7.5


def test_ampere_and_newer_support_bf16_and_flash_attention():
    assert supports_bf16_and_flash_attention(8) is True  # e.g. RTX 3090
    assert supports_bf16_and_flash_attention(9) is True  # e.g. H100/H200


def test_voice_resolution():
    from sts.config import TTSConfig

    cfg = TTSConfig()
    assert cfg.resolve_voice("male") == "Adde Michal"
    assert cfg.resolve_voice("Female") == "Ana Florence"
    assert cfg.resolve_voice("Craig Gutsy") == "Craig Gutsy"
    assert cfg.default_voice in cfg.all_voices()
    assert len(cfg.all_voices()) == 13
    assert all(cfg.resolve_voice(alias) in cfg.all_voices() for alias in cfg.voice_aliases)


def test_qwen3_backend_voices():
    from sts.config import TTSConfig

    cfg = TTSConfig()
    cfg.backend = "qwen3"
    cfg.use_backend_voices()
    assert cfg.all_voices() == ["Aiden", "Ryan"]
    assert cfg.resolve_voice(cfg.default_voice) == "Aiden"
    assert cfg.resolve_voice("male") == "Aiden"
    xtts = TTSConfig()
    xtts.use_backend_voices()
    assert "Adde Michal" in xtts.all_voices()


def test_qwen3_text_splitting_and_plausibility():
    import importlib.util, sys, types

    # sts.tts.qwen3 imports qwen_tts at module level; load just the pure helpers.
    stub = types.ModuleType("qwen_tts")
    stub.Qwen3TTSModel = object
    sys.modules.setdefault("qwen_tts", stub)
    from sts.tts.qwen3 import speech_length_plausible, split_for_qwen

    text = "Short one. " + ("This is a long clause that keeps going, " * 8).strip(", ") + "."
    pieces = split_for_qwen(text, 140)
    assert all(len(p) <= 140 for p in pieces)
    assert " ".join(pieces).replace(" ", "") == text.replace(" ", "")  # nothing dropped
    assert split_for_qwen("  ", 140) == []
    text = "One two three four five six seven eight nine ten."  # 50 characters
    assert speech_length_plausible(text, 3.2, 0.04, 0.16)
    assert not speech_length_plausible(text, 1.0, 0.04, 0.16)   # skipped
    assert not speech_length_plausible(text, 12.0, 0.04, 0.16)  # babble
    assert speech_length_plausible("Amen.", 0.6, 0.04, 0.16)    # too short to judge


def test_qwen3_clone_mode_selection():
    import sys, types

    stub = types.ModuleType("qwen_tts")
    stub.Qwen3TTSModel = object
    sys.modules.setdefault("qwen_tts", stub)
    from sts.tts.qwen3 import transcript_is_english, use_icl_conditioning

    ru, en = "Которые Он сказал во время Его последней беседы", "Brothers and sisters, we gather today."
    assert not transcript_is_english(ru) and transcript_is_english(en) and not transcript_is_english(None)
    assert not use_icl_conditioning("auto", ru)   # Russian reference -> embedding only
    assert use_icl_conditioning("auto", en)       # English (designed voice) -> full conditioning
    assert use_icl_conditioning("icl", ru) and not use_icl_conditioning("xvector", en)
    assert not use_icl_conditioning("auto", None)


def test_tts_backend_is_inferred_from_the_environment(monkeypatch):
    import importlib.util

    from sts import engine_args

    def installed(*modules):
        monkeypatch.setattr(importlib.util, "find_spec", lambda name: object() if name in modules else None)

    installed("qwen_tts")
    assert engine_args.default_tts_backend() == "qwen3"           # only Qwen3 here -> Qwen3, even for live use
    installed("TTS")
    assert engine_args.default_tts_backend(prefer="qwen3") == "xtts"
    installed("TTS", "qwen_tts")
    assert engine_args.default_tts_backend(prefer="qwen3") == "qwen3"   # both: the caller's preference
    assert engine_args.default_tts_backend() == "xtts"
    installed()
    assert engine_args.default_tts_backend() == "xtts"            # neither: falls through to the install hint


def test_clarity_bar_follows_the_voice():
    import sys, types

    stub = types.ModuleType("qwen_tts")
    stub.Qwen3TTSModel = object
    sys.modules.setdefault("qwen_tts", stub)
    from sts.tts.qwen3 import clarity_bar

    assert clarity_bar([], 4.0) is None and clarity_bar([-14.0, -14.0], 4.0) is None   # nothing judged without a track record
    assert clarity_bar([-14.0, -13.5, -14.5], 4.0) == -18.0                            # a clear voice: 4 dB below its own median
    assert clarity_bar([-22.0, -21.0, -23.0], 4.0) == -26.0                            # a muffled voice isn't retried forever


def test_translation_model_flag_overrides_the_default():
    import argparse

    from sts.config import TranslationConfig
    from sts.engine_args import add_engine_arguments, build_engine_config

    parser = argparse.ArgumentParser()
    add_engine_arguments(parser)

    default_cfg, _ = build_engine_config(parser.parse_args(["--device", "cpu", "--tts-backend", "xtts"]))
    assert default_cfg.translation.model_name == TranslationConfig().model_name  # untouched without the flag

    cfg, _ = build_engine_config(
        parser.parse_args(["--device", "cpu", "--tts-backend", "xtts", "--translation-model", "facebook/nllb-200-3.3B"])
    )
    assert cfg.translation.model_name == "facebook/nllb-200-3.3B"
