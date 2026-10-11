"""Caption, Translate and Voice are independent: with Caption off, sentences
already captioned are still translated (and voiced)."""

import queue

from sts.config import PipelineConfig
from sts.control import RecognitionControl
from sts.pipeline import Pipeline


def test_translation_carries_on_with_caption_off():
    p = Pipeline.__new__(Pipeline)
    p.cfg = PipelineConfig()
    p.control = RecognitionControl(sample_rate=16_000)
    p._transcripts = queue.Queue()
    translated = []
    p._translate_sentence = lambda text, *args: translated.append(text)
    p.control.stop()  # Caption off
    p._transcripts.put(("Последнее предложение.", 7, 2.0, False, 0.0))
    p._translate_step()
    assert translated == ["Последнее предложение."]
