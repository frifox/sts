from sts.translate import glossary


def test_mask_unmask_roundtrip_with_inflected_forms():
    # Genitive/prepositional case endings, not the glossary's dictionary form.
    text = "Сегодня мы поговорим о благодати и покаянии."
    masked, translations = glossary.mask(text)

    assert "благодат" not in masked.lower()
    assert "покаян" not in masked.lower()

    restored = glossary.unmask(masked, translations)
    assert "grace" in restored
    assert "repentance" in restored


def test_mask_prefers_longer_phrases():
    text = "Вечеря Господня в это воскресенье."
    masked, translations = glossary.mask(text)
    restored = glossary.unmask(masked, translations)

    assert "the Lord's Supper" in restored


def test_mask_is_case_insensitive():
    text = "ГОСПОДЬ благ."
    masked, translations = glossary.mask(text)
    assert masked != text
    restored = glossary.unmask(masked, translations)
    assert "the Lord" in restored


def test_no_terms_present_is_a_noop():
    text = "Это просто предложение без специальных терминов."
    masked, translations = glossary.mask(text)
    assert masked == text
    assert translations == []
