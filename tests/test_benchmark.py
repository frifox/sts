from sts.benchmark import format_rtf, rtf


def test_rtf_faster_than_realtime():
    assert abs(rtf(processing_s=5.0, audio_s=10.0) - 0.5) < 1e-9


def test_rtf_slower_than_realtime():
    assert abs(rtf(processing_s=20.0, audio_s=10.0) - 2.0) < 1e-9


def test_format_rtf_labels_faster_and_slower():
    assert "faster" in format_rtf(0.5)
    assert "slower" in format_rtf(1.5)
