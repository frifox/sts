from pathlib import Path

from sts.asr.whisper_asr import TimedSegment
from sts.offline import Unit, default_output_path, group_units, pick_reference_units, pick_reference_window, plan_timeline


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
    from sts.tts.qwen3 import min_plausible_seconds_per_char, speech_length_plausible

    assert min_plausible_seconds_per_char([0.06] * 4, 0.04) == 0.04               # too little history: absolute floor
    assert abs(min_plausible_seconds_per_char([0.06] * 5, 0.04) - 0.045) < 1e-9   # 75% of the voice's median
    assert min_plausible_seconds_per_char([0.03] * 5, 0.04) == 0.04               # never below the floor


def test_measured_sentences_are_not_flagged_but_a_skip_is():
    """Real measurements from a cloned Qwen3 voice: (characters, seconds) of correct, complete speech."""
    from sts.tts.qwen3 import min_plausible_seconds_per_char, speech_length_plausible

    measured = [(104, 5.7), (100, 6.4), (103, 6.6), (94, 7.0), (90, 5.6), (99, 6.5),
                (84, 5.3), (66, 5.2), (85, 5.4), (99, 6.2), (42, 2.9), (45, 3.0)]
    for i, (chars, seconds) in enumerate(measured):
        history = [s / c for j, (c, s) in enumerate(measured) if j != i]
        bar = min_plausible_seconds_per_char(history, 0.04)
        assert speech_length_plausible("x" * chars, seconds, bar, 0.16), (chars, seconds)
    # the same 100-character sentence with a third of it skipped
    history = [s / c for c, s in measured]
    bar = min_plausible_seconds_per_char(history, 0.04)
    assert not speech_length_plausible("x" * 100, 6.4 * 0.64, bar, 0.16)


def test_speaker_reference_prefers_clear_short_sentences_and_avoids_blurry_long_ones():
    units = [
        Unit(0, 5, "muffled", -0.2, clarity=-28.0),
        Unit(10, 20, "poor transcription", -1.5, clarity=-10.0),
        Unit(30, 55, "one 25 s blur", -0.2, clarity=-12.0),          # far too long to be one person's sentence
        Unit(60, 66, "clear", -0.2, clarity=-14.0),
        Unit(70, 78, "also clear", -0.2, clarity=-15.0),
    ]
    picked = pick_reference_units(units, [0, 1, 2, 3, 4])
    assert picked == [3, 4]                                          # the clear ones, in time order; 14 s is enough
    assert 1 not in picked and 2 not in picked                       # low confidence / blurry long unit left out


def test_speaker_reference_falls_back_gracefully():
    units = [Unit(0, 2, "a", -0.2), Unit(5, 6, "b", -0.9)]
    assert pick_reference_units(units, [0, 1]) == [0, 1]            # little speech: use all of it
    assert pick_reference_units(units, []) == []
    long_only = [Unit(0, 25, "long", -0.2, clarity=-15.0)]
    assert pick_reference_units(long_only, [0]) == [0]              # a long unit beats having no reference


def test_dull_takes_are_judged_against_the_voices_own_median():
    from sts.offline import find_dull_takes

    clear_voice = {0: -14.0, 1: -13.5, 2: -14.5, 3: -21.0, 4: -14.2}          # one take is muffled
    bar, dull = find_dull_takes(clear_voice)
    assert dull == [3] and abs(bar - (-18.2)) < 1e-9

    muffled_voice = {0: -22.0, 1: -21.5, 2: -22.5, 3: -21.8}                   # muffled every time: nothing to retake
    assert find_dull_takes(muffled_voice)[1] == []

    assert find_dull_takes({0: -14.0, 1: -25.0})[1] == []                      # too few takes to judge
    even = {0: -10.0, 1: -12.0, 2: -14.0, 3: -30.0}                            # even count: median is the middle pair's mean
    assert find_dull_takes(even)[0] == -13.0 - 4.0 and find_dull_takes(even)[1] == [3]
    assert find_dull_takes({0: -30.0, 1: -25.0, 2: -14.0, 3: -13.0, 4: -14.0}, margin_db=4.0)[1] == [0, 1]   # worst first


def test_single_speaker_reference_prefers_the_clearer_stretch():
    units = [Unit(0, 6, "a", -0.2, clarity=-27.0), Unit(6.5, 13, "b", -0.2, clarity=-27.0),      # muffled, 13 s
             Unit(40, 47, "c", -0.2, clarity=-13.0), Unit(47.5, 54, "d", -0.2, clarity=-13.0)]   # clear, 14 s
    assert pick_reference_window(units) == (2, 3)
