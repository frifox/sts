from pathlib import Path

from sts.asr.whisper_asr import TimedSegment
from sts.offline import Unit, default_output_path, group_units, pick_reference_window, plan_timeline


def seg(start, end, text="слово", logprob=-0.2):
    return TimedSegment(start, end, text, 0.01, logprob)


def test_default_output_name():
    assert default_output_path(Path("/v/talk.mp4")) == Path("/v/talk-english.mp4")
    assert default_output_path(Path("sermon.final.mkv")) == Path("sermon.final-english.mkv")
    assert default_output_path(Path("clip.MOV")) == Path("clip-english.MOV")


def test_segments_merge_until_a_sentence_ends():
    segments = [seg(0, 2, "Мы собрались"), seg(2.1, 4, "здесь сегодня."), seg(4.5, 6, "Аминь.")]
    units = group_units(segments)
    assert [(u.start, u.end, u.text) for u in units] == [
        (0, 4, "Мы собрались здесь сегодня."),
        (4.5, 6, "Аминь."),
    ]


def test_long_pause_and_max_length_split_units():
    assert len(group_units([seg(0, 2, "Один"), seg(5, 7, "два")])) == 2  # 3 s pause
    long_run = [seg(i * 5, i * 5 + 4.9, "и") for i in range(6)]  # never ends a sentence
    assert all(u.end - u.start <= 20 for u in group_units(long_run, max_len=20))


def test_reference_window_prefers_the_longest_confident_run():
    units = [
        Unit(0, 4, "a", -0.2), Unit(4, 9, "b", -0.9),          # b is low confidence: breaks the run
        Unit(10, 16, "c", -0.2), Unit(16, 24, "d", -0.3),       # 14 s run
        Unit(30, 35, "e", -0.2),
    ]
    assert pick_reference_window(units, max_seconds=20) == (2, 3)
    assert pick_reference_window([Unit(0, 3, "x", -2.0)]) is None
    assert pick_reference_window([Unit(0, 3, "x", -0.1)]) == (0, 0)  # short is better than nothing


def test_reference_window_respects_max_length():
    units = [Unit(i * 8, i * 8 + 8, "u", -0.1) for i in range(5)]
    first, last = pick_reference_window(units, max_seconds=20)
    assert units[last].end - units[first].start <= 20


def test_timeline_keeps_original_start_when_it_fits():
    placements = plan_timeline([0.0, 10.0, 20.0], [3.0, 4.0, 2.0])
    assert [p.start for p in placements] == [0.0, 10.0, 20.0]
    assert all(p.rate == 1.0 for p in placements)


def test_timeline_speeds_up_a_clip_that_overruns_the_next():
    placements = plan_timeline([0.0, 5.0], [6.0, 2.0], max_speedup=1.3)
    assert 1.0 < placements[0].rate <= 1.3            # 6 s must fit in ~4.85 s -> capped at 1.3
    assert placements[1].start >= 0.0 + 6.0 / placements[0].rate  # second is pushed back, not overlapped


def test_timeline_pushes_later_clips_then_catches_up():
    placements = plan_timeline([0.0, 4.0, 60.0], [9.0, 3.0, 2.0], max_speedup=1.3)
    assert placements[1].start > 4.0                  # pushed by the long first clip
    assert placements[2].start == 60.0                # a long pause absorbs the delay


def test_timeline_never_overlaps_clips():
    starts = [0.0, 1.0, 2.0, 3.0]
    durations = [3.0, 3.0, 3.0, 3.0]
    placements = plan_timeline(starts, durations, max_speedup=1.3)
    for a, b, d in zip(placements, placements[1:], durations):
        assert b.start >= a.start + d / a.rate


def test_sentences_from_words_get_their_own_timing():
    from types import SimpleNamespace as W

    from sts.asr.whisper_asr import sentences_from_words

    words = [W(start=0.0, end=0.5, word=" Мы"), W(start=0.5, end=1.0, word=" здесь."),
             W(start=5.0, end=5.4, word=" Потом"), W(start=5.4, end=6.0, word=" пауза."),
             W(start=9.0, end=9.5, word=" Конец")]  # trailing words without an ending
    assert sentences_from_words(words) == [(0.0, 1.0, "Мы здесь."), (5.0, 6.0, "Потом пауза."), (9.0, 9.5, "Конец")]
    assert sentences_from_words([]) == []


def test_pace_bar_follows_the_voice():
    import sys, types

    stub = types.ModuleType("qwen_tts")
    stub.Qwen3TTSModel = object
    sys.modules.setdefault("qwen_tts", stub)
    from sts.tts.qwen3 import min_plausible_seconds_per_word

    assert min_plausible_seconds_per_word([], 0.2) == 0.2                      # no record yet: absolute floor
    assert min_plausible_seconds_per_word([0.28, 0.28, 0.29], 0.2) < 0.21      # fast voice: 0.28 s/word is fine
    assert abs(min_plausible_seconds_per_word([0.5, 0.5, 0.5], 0.2) - 0.35) < 1e-9  # slower voice: stricter bar
