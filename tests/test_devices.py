import os

from sts.devices import resolve_devices


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
