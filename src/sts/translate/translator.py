"""Russian -> English translation via NLLB-200, with glossary term forcing."""

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from sts.config import TranslationConfig
from sts.translate import glossary


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

        source = text
        placeholder_values: list[str] = []
        if self.cfg.apply_glossary:
            source, placeholder_values = glossary.mask(text)

        inputs = self.tokenizer(source, return_tensors="pt").to(self.cfg.device)
        generated = self.model.generate(
            **inputs, forced_bos_token_id=self._target_id, max_new_tokens=256
        )
        result = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

        if self.cfg.apply_glossary and placeholder_values:
            result = glossary.unmask(result, placeholder_values)
        return result
