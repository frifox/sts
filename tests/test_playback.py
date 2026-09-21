from sts.audio.playback import pause_needed


def test_no_pause_before_first_clip():
    assert pause_needed(now=100.0, scheduled_end=0.0, gap=1.0) == 0.0


def test_full_pause_when_playback_is_catching_up():
    # earlier audio is still playing (ends in the future) -> a full gap after it
    assert pause_needed(now=100.0, scheduled_end=103.0, gap=1.0) == 1.0


def test_partial_pause_when_previous_just_ended():
    assert abs(pause_needed(now=100.4, scheduled_end=100.0, gap=1.0) - 0.6) < 1e-9


def test_no_pause_after_a_long_natural_gap():
    assert pause_needed(now=110.0, scheduled_end=100.0, gap=1.0) == 0.0
