"""Russian -> English translation via NLLB-200, with glossary term forcing."""

import re

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from sts.config import TranslationConfig
from sts.gpu import gpu_lock
from sts.textclean import collapse_repeats
from sts.translate import glossary

# A glossary placeholder the model mangled into a run of Z's, or any token repeated in a stuck loop.
_DEGENERATE = re.compile(r"Z{6,}|ZZGLOSS|(\b\w+\b)(?:[\s,.]+\1\b){5,}", re.IGNORECASE)


# One- and two-word utterances are what NLLB handles worst: given «Аминь.» on its
# own it invents a sentence ("I'm not going to."). These common church replies
# are translated exactly instead.
SHORT_PHRASES = {
    "аминь": "Amen.",
    "аллилуйя": "Hallelujah.",
    "алиллуйя": "Hallelujah.",
    "слава богу": "Praise God.",
    "слава господу": "Glory to the Lord.",
    "спасибо": "Thank you.",
    "благодарю": "Thank you.",
    "да": "Yes.",
    "нет": "No.",
    "хорошо": "Good.",
    "пожалуйста": "Please.",
}


def short_phrase(text: str) -> str | None:
    """The exact English for a known one- or two-word utterance, else None."""
    key = re.sub(r"[^\w\s]", "", text.lower()).strip()
    return SHORT_PHRASES.get(re.sub(r"\s+", " ", key))


def split_sentences(text: str, min_words: int = 2) -> list[str]:
    """Splits on sentence-ending punctuation; fragments shorter than
    `min_words` are merged into the previous sentence rather than translated alone."""
    sentences: list[str] = []
    for part in re.split(r"(?<=[.!?…])\s+", text.strip()):
        if not part:
            continue
        if sentences and len(part.split()) < min_words:
            sentences[-1] = f"{sentences[-1]} {part}"
        else:
            sentences.append(part)
    return sentences


class Translator:
    def __init__(self, cfg: TranslationConfig):
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name, src_lang=cfg.source_lang
        )
        # fp16 on cuda roughly halves VRAM versus the default fp32 load - matters
        # once ASR and TTS are also sharing the same GPU (observed OOM on an 8GB
        # card loading NLLB fp32 alongside faster-whisper + Qwen3-TTS).
        # `torch_dtype`, not `dtype`: sts supports transformers back to 4.40
        # (the xtts extra pins <4.50), and `dtype` only exists on newer
        # releases - `torch_dtype` still works everywhere, just deprecated
        # (with a one-time warning) on the newest ones.
        dtype = torch.float16 if cfg.device == "cuda" else torch.float32
        self.model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name, torch_dtype=dtype).to(
            cfg.device
        )
        self.model.eval()

        # transformers has moved this attribute around across NLLB releases.
        if hasattr(self.tokenizer, "lang_code_to_id"):
            self._target_id = self.tokenizer.lang_code_to_id[cfg.target_lang]
        else:
            self._target_id = self.tokenizer.convert_tokens_to_ids(cfg.target_lang)

    def translate(self, text: str) -> str:
        if not text.strip():
            return ""

        # NLLB is trained on single sentences and silently drops content when
        # handed several at once, so a speech segment is translated per sentence.
        return " ".join(self._translate_sentence(sentence) for sentence in split_sentences(text)).strip()

    def _translate_sentence(self, text: str) -> str:
        known = short_phrase(text)
        if known:
            return known
        result = ""
        if self.cfg.apply_glossary:
            source, placeholder_values = glossary.mask(text)
            result = self._generate(source)
            if _DEGENERATE.search(result) or not glossary.all_placeholders_present(result, placeholder_values):
                # The model garbled or dropped a placeholder (typical when a
                # sentence is mostly glossary terms); retry on the plain text.
                result = ""
            elif placeholder_values:
                result = glossary.unmask(result, placeholder_values)
        if not result:
            result = self._generate(text)
        return collapse_repeats(result)

    def _generate(self, source: str) -> str:
        inputs = self.tokenizer(source, return_tensors="pt")
        input_tokens = inputs["input_ids"].shape[1]
        with gpu_lock(self.cfg.device):
            generated = self.model.generate(
                **inputs.to(self.cfg.device),
                forced_bos_token_id=self._target_id,
                # Translations are about as long as their source; a hard cap
                # relative to the input stops a runaway loop early.
                max_new_tokens=min(256, int(input_tokens * 2) + 10),
                no_repeat_ngram_size=4,
            )
            generated = generated.cpu()
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
