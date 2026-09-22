import pytest

from sts.benchmark import (
    StageResult,
    coqui_model_cache_dir,
    coqui_model_is_cached,
    first_line,
    format_rtf,
    format_table,
    is_not_installed,
    result_row,
    rtf,
)


def test_rtf_faster_than_realtime():
    assert abs(rtf(processing_s=5.0, audio_s=10.0) - 0.5) < 1e-9


def test_rtf_slower_than_realtime():
    assert abs(rtf(processing_s=20.0, audio_s=10.0) - 2.0) < 1e-9


def test_format_rtf_labels_faster_and_slower():
    assert "faster" in format_rtf(0.5)
    assert "slower" in format_rtf(1.5)


def test_first_line_takes_only_the_first_line():
    assert first_line("one\ntwo\nthree") == "one"


def test_first_line_empty_string():
    assert first_line("") == ""


def test_format_table_aligns_columns_to_widest_cell():
    table = format_table(["A", "BB"], [["1", "22"], ["333", "4"]])
    lines = table.splitlines()
    assert len(lines) == 4  # header, separator, 2 rows
    # every column is padded to its widest cell ("333", 3 chars) so the 2-space
    # gap before column 2 lands at the same offset on every line
    gap_start = len("333")
    for line in [lines[0], lines[2], lines[3]]:  # header + the 2 data rows
        assert line[gap_start:gap_start + 2] == "  "


def test_result_row_ok_stage_has_numeric_columns():
    r = StageResult("asr", "cpu", True, detail="9.4s clip, RTF 1.95x", model_load_s=1.0, warmup_s=2.0, runs_s=[3.0, 3.2])
    row = result_row(r)
    assert row[:3] == ["asr", "cpu", "OK"]
    assert row[3] == "1.00" and row[4] == "2.00"
    assert row[-1] == "9.4s clip, RTF 1.95x"


def test_result_row_failed_stage_has_dashes():
    r = StageResult("tts:xtts", "cuda", False, detail="not installed: no module named TTS\nmore detail")
    row = result_row(r)
    assert row[:3] == ["tts:xtts", "cuda", "SKIP"]
    assert row[3:7] == ["-", "-", "-", "-"]
    assert row[-1] == "not installed: no module named TTS"  # only the first line


def test_is_not_installed_true_for_missing_backend():
    r = StageResult("tts:xtts", "cuda", False, detail="not installed: no module named TTS\nmore detail")
    assert is_not_installed(r)


def test_is_not_installed_false_for_a_genuine_error():
    r = StageResult("tts:qwen3", "cuda", False, detail="OutOfMemoryError('CUDA out of memory')")
    assert not is_not_installed(r)


def test_is_not_installed_false_for_a_successful_stage():
    r = StageResult("asr", "cpu", True, detail="9.4s clip, RTF 1.95x", model_load_s=1.0, warmup_s=2.0, runs_s=[3.0])
    assert not is_not_installed(r)


def test_coqui_model_cache_dir_replaces_slashes(monkeypatch, tmp_path):
    pytest.importorskip("TTS")  # coqui_model_cache_dir calls into Coqui's own path logic
    monkeypatch.setenv("TTS_HOME", str(tmp_path))
    d = coqui_model_cache_dir("tts_models/multilingual/multi-dataset/xtts_v2")
    assert d == tmp_path / "tts" / "tts_models--multilingual--multi-dataset--xtts_v2"


def test_coqui_model_is_cached_false_when_dir_missing(monkeypatch, tmp_path):
    pytest.importorskip("TTS")
    monkeypatch.setenv("TTS_HOME", str(tmp_path))
    assert not coqui_model_is_cached("tts_models/multilingual/multi-dataset/xtts_v2")


def test_coqui_model_is_cached_false_when_dir_empty(monkeypatch, tmp_path):
    pytest.importorskip("TTS")
    monkeypatch.setenv("TTS_HOME", str(tmp_path))
    d = coqui_model_cache_dir("tts_models/multilingual/multi-dataset/xtts_v2")
    d.mkdir(parents=True)
    assert not coqui_model_is_cached("tts_models/multilingual/multi-dataset/xtts_v2")


def test_coqui_model_is_cached_true_once_populated(monkeypatch, tmp_path):
    pytest.importorskip("TTS")
    monkeypatch.setenv("TTS_HOME", str(tmp_path))
    d = coqui_model_cache_dir("tts_models/multilingual/multi-dataset/xtts_v2")
    d.mkdir(parents=True)
    (d / "model.pth").write_bytes(b"fake weights")
    assert coqui_model_is_cached("tts_models/multilingual/multi-dataset/xtts_v2")
