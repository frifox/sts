"""Russian -> English translation via MADLAD-400 (optionally with glossary term forcing).

MADLAD-400 (Google's T5-based translation model) was chosen over NLLB-200
after comparing both on a recorded sermon (translate.py): it reproduces
Bible passages in standard English wording (mostly KJV) on its own, without
the reading being announced or looked up, and is the most accurate on the
preacher's own speech too. Any MADLAD checkpoint works (3b/7b/10b-mt).
"""

import re

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

from sts.config import TranslationConfig
from sts.gpu import gpu_lock
from sts.textclean import collapse_repeats
from sts.translate import glossary

# A glossary placeholder the model mangled into a run of Z's, or any token repeated in a stuck loop.
_DEGENERATE = re.compile(r"Z{6,}|ZZGLOSS|(\b\w+\b)(?:[\s,.]+\1\b){5,}", re.IGNORECASE)


# One- and two-word utterances are what translation models handle worst: NLLB, given «Аминь.» on its
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
        self.tokenizer = T5Tokenizer.from_pretrained(cfg.model_name)
        # bf16 halves the memory of fp32 (10b: ~21GB instead of ~43GB). Not
        # fp16: T5 was trained in bf16 and its activations overflow fp16.
        # `torch_dtype`, not `dtype`: sts supports transformers back to 4.40
        # (the xtts extra pins <4.50), and `dtype` only exists on newer
        # releases - `torch_dtype` still works everywhere, just deprecated
        # (with a one-time warning) on the newest ones.
        dtype = torch.float32 if cfg.device == "cpu" else torch.bfloat16
        self.model = T5ForConditionalGeneration.from_pretrained(cfg.model_name, torch_dtype=dtype).to(
            cfg.device
        )
        self.model.eval()
        # MADLAD detects the source language itself; only the target is named.
        self._prefix = f"<2{cfg.target_lang}> "

    def translate(self, text: str) -> str:
        return " ".join(self.translate_sentences(text)).strip()

    def translate_sentences(self, text: str):
        """Translates `text` one sentence at a time, yielding each translation
        the moment it's ready instead of waiting for the whole passage.

        Translation models are trained on single sentences and can silently
        drop content when handed several at once (NLLB did), so text is
        translated per sentence either way; `translate()` above just joins
        them back into one string."""
        if not text.strip():
            return
        for sentence in split_sentences(text):
            yield self._translate_sentence(sentence)

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
        inputs = self.tokenizer(self._prefix + source, return_tensors="pt")
        input_tokens = inputs["input_ids"].shape[1]
        with gpu_lock(self.cfg.device):
            generated = self.model.generate(
                **inputs.to(self.cfg.device),
                # Translations are about as long as their source; a hard cap
                # relative to the input stops a runaway loop early.
                max_new_tokens=min(256, int(input_tokens * 2) + 10),
                no_repeat_ngram_size=4,
            )
            generated = generated.cpu()
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
