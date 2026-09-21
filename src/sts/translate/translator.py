"""Russian -> English translation via NLLB-200, with glossary term forcing."""

import re

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from sts.config import TranslationConfig
from sts.textclean import collapse_repeats
from sts.translate import glossary

# A glossary placeholder the model mangled into a run of Z's, or any token repeated in a stuck loop.
_DEGENERATE = re.compile(r"Z{6,}|ZZGLOSS|(\b\w+\b)(?:[\s,.]+\1\b){5,}", re.IGNORECASE)


def split_sentences(text: str, min_words: int = 3) -> list[str]:
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
        self.model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name).to(cfg.device)
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
        result = ""
        if self.cfg.apply_glossary:
            source, placeholder_values = glossary.mask(text)
            result = self._generate(source)
            if placeholder_values and not _DEGENERATE.search(result):
                result = glossary.unmask(result, placeholder_values)
            elif _DEGENERATE.search(result):
                result = ""  # the placeholders confused the model; retry on the plain text
        if not result:
            result = self._generate(text)
        return collapse_repeats(result)

    def _generate(self, source: str) -> str:
        inputs = self.tokenizer(source, return_tensors="pt").to(self.cfg.device)
        input_tokens = inputs["input_ids"].shape[1]
        generated = self.model.generate(
            **inputs,
            forced_bos_token_id=self._target_id,
            # Translations are about as long as their source; a hard cap
            # relative to the input stops a runaway loop early.
            max_new_tokens=min(256, int(input_tokens * 2) + 10),
            no_repeat_ngram_size=4,
        )
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
