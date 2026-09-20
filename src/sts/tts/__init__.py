from sts.config import TTSConfig


def create_synthesizer(cfg: TTSConfig):
    if cfg.backend == "xtts":
        from sts.tts.xtts import XTTSSynthesizer

        return XTTSSynthesizer(cfg)
    if cfg.backend == "qwen3":
        from sts.tts.qwen3 import Qwen3TTSSynthesizer

        return Qwen3TTSSynthesizer(cfg)
    raise ValueError(f"unknown TTS backend: {cfg.backend!r} (expected 'xtts' or 'qwen3')")
