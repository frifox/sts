import os

from sts.devices import resolve_devices, supports_bf16_and_flash_attention


def test_cuda_maps_all_components_to_cuda():
    assert resolve_devices("cuda", None) == ("cuda", "cuda", "cuda")


def test_cpu_maps_all_components_to_cpu():
    assert resolve_devices("cpu", None) == ("cpu", "cpu", "cpu")


def test_mps_forces_asr_to_cpu_but_keeps_translation_and_tts_on_mps():
    asr, translation, tts = resolve_devices("mps", None)
    assert asr == "cpu"
    assert translation == "mps"
    assert tts == "mps"


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
