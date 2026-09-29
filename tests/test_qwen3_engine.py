"""Choosing what runs Qwen3-TTS: PyTorch (qwen-tts) or MLX (mlx-audio)."""

from sts.gpu import model_bytes
from sts.tts import qwen3_engine
from sts.tts.qwen3_mlx import mlx_model_name


def test_the_mlx_conversion_of_each_qwen3_model():
    assert mlx_model_name("Qwen/Qwen3-TTS-12Hz-1.7B-Base") == "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"
    assert mlx_model_name("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice") == "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16"
    assert mlx_model_name("mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit") == "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit"


def test_auto_picks_the_engine_this_environment_has(monkeypatch):
    monkeypatch.setattr(qwen3_engine, "available_engines", lambda: ["mlx"])
    assert qwen3_engine.resolve_engine("auto") == "mlx"
    monkeypatch.setattr(qwen3_engine, "available_engines", lambda: ["torch", "mlx"])
    assert qwen3_engine.resolve_engine("auto") == "torch"
    assert qwen3_engine.resolve_engine("mlx") == "mlx"


def test_an_mlx_model_s_weights_are_counted():
    class _MLXModel:
        model = None

        def weight_bytes(self):
            return 1234

    assert model_bytes(_MLXModel()) == 1234
