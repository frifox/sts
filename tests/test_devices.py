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
