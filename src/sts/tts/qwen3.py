"""Synthesizes English speech with Qwen3-TTS, either zero-shot cloning a
reference clip or using a real trained LoRA adapter for a repeat speaker.

Zero-shot API per github.com/QwenLM/Qwen3-TTS (Apache-2.0, `pip install
qwen-tts`): `model.create_voice_clone_prompt(ref_audio=..., ref_text=...,
x_vector_only_mode=...)` builds a reusable prompt once; `generate_voice_clone`
consumes it. `ref_text` is the reference clip's own transcript and
materially improves quality; `x_vector_only_mode` drops that requirement at
some quality cost, and is the fallback when no transcript is available.

LoRA loading (`_load_lora_model` and its helpers below) mirrors
`finetuning/infer_lora_custom_voice.py` from
github.com/cheeweijie/qwen3-tts-lora-finetuning (Apache-2.0), the community
patch used by `training/qwen3_lora/` to actually train these adapters - see
that directory for the training side. A LoRA speaker is selected with
`generate_custom_voice(speaker=...)`, not `generate_voice_clone`: the
trained checkpoint bakes the voice into a named speaker slot (with its own
embedding) inside the model rather than being conditioned on a reference
clip at synthesis time.

Each cached LoRA speaker is a second, full model load (base weights + the
adapter merged in) - unlike the zero-shot path, which reuses one base model
for every reference clip. On a memory-constrained consumer GPU, keeping more
than a couple of trained speakers active at once will add up; there's no
cache eviction here, only what `reset_voice()`/picking a different speaker
overwrites as the *active* one (cached entries stay loaded for fast
re-selection).

When no cloned/trained voice is active, a preset speaker from Qwen3-TTS's
separate CustomVoice model is used instead (default: Aiden, with a calm
"preacher" style instruction), so no reference clip is needed to get sound.
It's a second model load alongside the Base model.

This backend is unverified against real hardware in this repo (no GPU
available in dev) - the call shapes come directly from the two upstream
projects' example code, not from having actually run it.

Device support: `cfg.device` is passed straight through as `device_map`, so
"cuda", "cpu", and "mps" (Apple Silicon) all work as far as loading the
model goes. bfloat16 and flash-attention aren't requirements of the model
itself, though, and aren't even a safe default for every CUDA GPU: both need
Ampere or newer (compute capability >= 8) - an older card like a GTX 1080 Ti
(Pascal, compute 6.1) supports neither properly, and would fail to load or
run badly if forced onto that path. `_dtype_for_device`/`_attn_impl_for_device`
below check the actual GPU's compute capability (via
`sts.devices.supports_bf16_and_flash_attention`, kept import-light so it's
unit-testable without a GPU) and fall back to float16 + SDPA attention on
mps or an older/absent GPU, same reasoning as the Metal case either way.
"""

import gc
import threading

import numpy as np
import torch
from qwen_tts import Qwen3TTSModel

from sts.config import TTSConfig
from sts.devices import supports_bf16_and_flash_attention
from sts.tts.voice_library import VoiceEntry


def _cuda_supports_bf16_and_flash_attention() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _minor = torch.cuda.get_device_capability()
    return supports_bf16_and_flash_attention(major)


def _dtype_for_device(device: str) -> torch.dtype:
    if device == "cuda":
        return torch.bfloat16 if _cuda_supports_bf16_and_flash_attention() else torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def _attn_impl_for_device(device: str) -> str:
    if device == "cuda" and _cuda_supports_bf16_and_flash_attention():
        return "flash_attention_2"
    return "sdpa"


class Qwen3TTSSynthesizer:
    def __init__(self, cfg: TTSConfig):
        self.cfg = cfg
        self.model = Qwen3TTSModel.from_pretrained(
            cfg.qwen3_model_name,
            device_map=cfg.device,
            dtype=_dtype_for_device(cfg.device),
            attn_implementation=_attn_impl_for_device(cfg.device),
        )
        self._custom_model = Qwen3TTSModel.from_pretrained(
            cfg.qwen3_custom_model_name,
            device_map=cfg.device,
            dtype=_dtype_for_device(cfg.device),
            attn_implementation=_attn_impl_for_device(cfg.device),
        )
        self.default_voice = cfg.resolve_voice(cfg.default_voice)
        # MPS/CUDA models aren't safe to run from two threads at once, and the
        # voice-design dialog synthesizes previews while the live pipeline runs.
        self._lock = threading.Lock()
        self._design_model = None
        self._design_load_lock = threading.Lock()
        self._prompt = None
        self.cloned_speaker_wav: str | None = None  # set whenever a zero-shot clone is active
        self._lora_models: dict[str, tuple[Qwen3TTSModel, str]] = {}  # speaker_id -> (model, speaker_name)
        self._active_lora_speaker_id: str | None = None

    # -- voice design ("Create Voice" dialog) --------------------------------

    @property
    def design_loaded(self) -> bool:
        return self._design_model is not None

    def design_reference(self, instruct: str, ref_text: str) -> tuple[np.ndarray, int]:
        """Invents a voice from `instruct` and returns it speaking `ref_text`.

        Each call designs a *different* voice, so the returned clip is used as
        the reference for cloning (`preview`, and the saved library voice)
        rather than re-running the description for every sentence.
        """
        with self._design_load_lock:
            if self._design_model is None:
                self._design_model = Qwen3TTSModel.from_pretrained(
                    self.cfg.qwen3_design_model_name,
                    device_map=self.cfg.device,
                    dtype=_dtype_for_device(self.cfg.device),
                    attn_implementation=_attn_impl_for_device(self.cfg.device),
                )
        with self._lock:
            wavs, sample_rate = self._design_model.generate_voice_design(
                text=ref_text, instruct=instruct, language="English"
            )
        return np.asarray(wavs[0], dtype=np.float32), sample_rate

    def unload_designer(self) -> None:
        """Frees the VoiceDesign model when the dialog closes (it's a multi-GB model)."""
        with self._design_load_lock:
            self._design_model = None
        gc.collect()
        if self.cfg.device == "mps":
            torch.mps.empty_cache()
        elif self.cfg.device == "cuda":
            torch.cuda.empty_cache()

    def preview(self, text: str, reference_wav_path: str, ref_text: str) -> tuple[np.ndarray, int]:
        """Speaks `text` in the voice of a reference clip, without changing the active voice."""
        with self._lock:
            prompt = self.model.create_voice_clone_prompt(
                ref_audio=reference_wav_path, ref_text=ref_text, x_vector_only_mode=False
            )
            wavs, sample_rate = self.model.generate_voice_clone(
                text=text, language="English", voice_clone_prompt=prompt
            )
        return np.asarray(wavs[0], dtype=np.float32), sample_rate

    def clone_voice(self, reference_wav_path: str, ref_text: str | None = None) -> None:
        with self._lock:
            self._prompt = self.model.create_voice_clone_prompt(
                ref_audio=reference_wav_path,
                ref_text=ref_text,
                x_vector_only_mode=ref_text is None,
            )
        self.cloned_speaker_wav = reference_wav_path
        self._active_lora_speaker_id = None

    def clone_from_entry(self, entry: VoiceEntry) -> None:
        if entry.kind == "lora":
            self._activate_lora(entry)
        else:
            self.clone_voice(str(entry.wav_path), entry.ref_text)

    def _activate_lora(self, entry: VoiceEntry) -> None:
        if entry.speaker_id not in self._lora_models:
            self._lora_models[entry.speaker_id] = _load_lora_model(self.cfg, entry)
        self._active_lora_speaker_id = entry.speaker_id
        self._prompt = None
        self.cloned_speaker_wav = None

    def set_default_voice(self, voice: str) -> None:
        voice = self.cfg.resolve_voice(voice)
        if voice not in self.cfg.all_voices():
            raise ValueError(f"unknown preset voice {voice!r}; choose from {self.cfg.all_voices()}")
        self.default_voice = voice

    def reset_voice(self) -> None:
        self._prompt = None
        self.cloned_speaker_wav = None
        self._active_lora_speaker_id = None

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        if not text.strip():
            return np.zeros(0, dtype=np.float32), 24_000

        with self._lock:
            if self._active_lora_speaker_id:
                lora_model, speaker_name = self._lora_models[self._active_lora_speaker_id]
                wavs, sample_rate = lora_model.generate_custom_voice(
                    text=text, speaker=speaker_name, language="English"
                )
            elif self._prompt is not None:
                wavs, sample_rate = self.model.generate_voice_clone(
                    text=text, language="English", voice_clone_prompt=self._prompt
                )
            else:
                wavs, sample_rate = self._custom_model.generate_custom_voice(
                    text=text,
                    speaker=self.default_voice.lower(),
                    language="English",
                    instruct=self.cfg.qwen3_style_instruct or None,
                )
        return np.asarray(wavs[0], dtype=np.float32), sample_rate


def _load_lora_model(cfg: TTSConfig, entry: VoiceEntry) -> tuple[Qwen3TTSModel, str]:
    from peft import PeftModel

    base = Qwen3TTSModel.from_pretrained(
        cfg.qwen3_model_name,
        device_map=cfg.device,
        dtype=_dtype_for_device(cfg.device),
        attn_implementation=_attn_impl_for_device(cfg.device),
    )
    peft_model = PeftModel.from_pretrained(base.model, str(entry.adapter_dir))
    _set_lora_scale(peft_model, entry.lora_scale)
    base.model = peft_model.merge_and_unload()

    core_model = _resolve_core_model(base.model)
    _apply_speaker_config(core_model, str(entry.adapter_dir), entry.speaker_name)
    _apply_speaker_embedding(core_model, str(entry.adapter_dir), entry.speaker_name)
    core_model.eval()

    return base, entry.speaker_name


def _resolve_core_model(model):
    if hasattr(model, "talker"):
        return model
    if hasattr(model, "model") and hasattr(model.model, "talker"):
        return model.model
    if hasattr(model, "base_model"):
        base = model.base_model
        if hasattr(base, "talker"):
            return base
        if hasattr(base, "model") and hasattr(base.model, "talker"):
            return base.model
    return model


def _apply_speaker_config(core_model, adapter_dir: str, speaker_name: str, speaker_id: int = 3000) -> None:
    import json
    import os

    config_path = os.path.join(adapter_dir, "config.json")
    if os.path.isfile(config_path):
        with open(config_path, encoding="utf-8") as f:
            config_dict = json.load(f)
        talker_config = config_dict.get("talker_config", {})
        spk_id_map = talker_config.get("spk_id", {})
        spk_is_dialect = talker_config.get("spk_is_dialect", {})
        if isinstance(spk_id_map, dict):
            core_model.config.talker_config.spk_id.update(spk_id_map)
        if isinstance(spk_is_dialect, dict):
            core_model.config.talker_config.spk_is_dialect.update(spk_is_dialect)

    if speaker_name not in core_model.config.talker_config.spk_id:
        core_model.config.talker_config.spk_id[speaker_name] = speaker_id
        core_model.config.talker_config.spk_is_dialect[speaker_name] = False

    core_model.config.tts_model_type = "custom_voice"
    core_model.tts_model_type = "custom_voice"
    core_model.supported_speakers = core_model.config.talker_config.spk_id.keys()


def _apply_speaker_embedding(core_model, adapter_dir: str, speaker_name: str) -> None:
    import os

    from safetensors.torch import load_file

    embedding_path = os.path.join(adapter_dir, "speaker_embedding.safetensors")
    payload = load_file(embedding_path)
    spk_id = core_model.config.talker_config.spk_id[speaker_name]
    weight = core_model.talker.model.codec_embedding.weight
    weight.data[spk_id] = payload["target_speaker_embedding"].to(device=weight.device, dtype=weight.dtype)


def _set_lora_scale(peft_model, scale: float) -> None:
    if scale == 1.0:
        return
    from peft.tuners.lora import LoraLayer

    adapter = getattr(peft_model, "active_adapter", None)
    for module in peft_model.modules():
        if isinstance(module, LoraLayer):
            if adapter is None:
                module.scale_layer(scale)
            elif isinstance(adapter, (list, tuple)):
                for name in adapter:
                    module.set_scale(name, scale)
            else:
                module.set_scale(adapter, scale)
