import numpy as np

from sts.diarize import (
    DEFAULT_DISTANCE_THRESHOLD,
    SAMPLE_RATE,
    SpeakerEncoder,
    assign_speakers,
    cluster_embeddings,
    relabel_by_first_appearance,
)


def voiceprint(direction: int, jitter: float, rng, dims: int = 32) -> np.ndarray:
    """A unit vector near axis `direction`: a stand-in for one speaker's voiceprint."""
    v = np.zeros(dims)
    v[direction] = 1.0
    v = v + rng.normal(0, jitter, dims)
    return v / np.linalg.norm(v)


def test_relabel_numbers_speakers_by_first_appearance():
    assert relabel_by_first_appearance([7, 7, 3, 7, 9, 3]) == [0, 0, 1, 0, 2, 1]


def test_clusters_separate_speakers_and_keep_one_speaker_together():
    rng = np.random.default_rng(0)
    vectors = np.stack([voiceprint(d, 0.05, rng) for d in (0, 0, 0, 1, 1, 1, 2, 2, 2)])
    labels = cluster_embeddings(vectors, [6.0] * 9)
    assert labels == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert cluster_embeddings(vectors[:3], [6.0] * 3) == [0, 0, 0]            # one person stays one


def test_known_speaker_count_is_respected():
    rng = np.random.default_rng(1)
    vectors = np.stack([voiceprint(d, 0.05, rng) for d in (0, 0, 1, 1, 2, 2)])
    assert cluster_embeddings(vectors, [6.0] * 6, num_speakers=2) in ([0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1], [0, 0, 1, 1, 0, 0])
    assert len(set(cluster_embeddings(vectors, [6.0] * 6, num_speakers=3))) == 3
    assert cluster_embeddings(vectors, [6.0] * 6, num_speakers=1) == [0] * 6


def test_a_group_with_almost_no_speech_is_not_a_speaker():
    rng = np.random.default_rng(2)
    vectors = np.stack([voiceprint(d, 0.03, rng) for d in (0, 0, 0, 0, 1)])
    labels = cluster_embeddings(vectors, [8.0, 8.0, 8.0, 8.0, 1.6])          # the odd one out speaks 1.6 s in total
    assert len(set(labels)) == 1                                              # folded into the real speaker


class FakeEncoder:
    """Returns a voiceprint chosen by the audio's value, so no model is needed."""

    def __init__(self, rng):
        self.rng = rng

    def embed(self, audio):
        if len(audio) < SAMPLE_RATE // 2:
            return None
        return voiceprint(int(round(float(audio[0]))), 0.03, self.rng)


def make_units(spec):
    """spec: list of (start, end, speaker_axis)."""
    from sts.offline import Unit

    audio = np.zeros(int(max(e for _, e, _ in spec) * SAMPLE_RATE) + SAMPLE_RATE, dtype=np.float32)
    units = []
    for start, end, axis in spec:
        audio[int(start * SAMPLE_RATE): int(end * SAMPLE_RATE)] = axis
        units.append(Unit(start, end, "текст", -0.2))
    return units, audio


def test_assign_speakers_handles_short_and_tiny_sentences():
    units, audio = make_units([
        (0, 6, 0), (7, 13, 0),          # speaker A
        (14, 20, 1), (21, 27, 1),       # speaker B
        (28, 29.0, 0),                  # short (1 s): fingerprinted, closest voice is A
        (30, 30.3, 1),                  # tiny (0.3 s): takes whoever spoke just before
    ])
    result = assign_speakers(units, audio, FakeEncoder(np.random.default_rng(3)))
    assert result.labels == [0, 0, 1, 1, 0, 0]
    assert result.count == 2


def test_the_default_threshold_is_in_the_measured_gap():
    # Measured with the real model: same speaker <= 0.33 apart even for 3 s clips, but
    # two very similar different voices were 0.31-0.35 apart. The threshold stays below 0.36.
    assert 0.25 <= DEFAULT_DISTANCE_THRESHOLD <= 0.35


def test_real_encoder_gives_unit_length_voiceprints_and_ignores_tiny_clips():
    encoder = SpeakerEncoder()
    tone = (0.1 * np.sin(2 * np.pi * 150 * np.arange(SAMPLE_RATE * 2) / SAMPLE_RATE)).astype(np.float32)
    embedding = encoder.embed(tone)
    assert embedding.shape == (256,) and abs(float(np.linalg.norm(embedding)) - 1.0) < 1e-4
    assert encoder.embed(tone[: SAMPLE_RATE // 4]) is None
