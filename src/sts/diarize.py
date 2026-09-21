"""Works out who is speaking when, so each person in a multi-speaker video can
be given their own cloned voice.

Each sentence is turned into a speaker embedding (a 256-number "voiceprint")
by WeSpeaker's ResNet34 model, a small public ONNX model that needs no login,
and the embeddings are grouped by similarity. Sentences whose voiceprints are
close belong to the same person. No speaker count needs to be given, though
`num_speakers` fixes it when known (more reliable).

Limits: one speaker per sentence (a change of speaker in the middle of a
sentence isn't split), overlapping speech isn't separated, and very short
sentences are unreliable so they are assigned to the closest voice found.
"""

from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16_000
MODEL_REPO = "Wespeaker/wespeaker-voxceleb-resnet34-LM"
MODEL_FILE = "voxceleb_resnet34_LM.onnx"

# Sentences shorter than this give unreliable voiceprints: they don't take part
# in clustering, they are assigned to the closest speaker afterwards.
MIN_EMBED_SECONDS = 1.5
# A "speaker" with less speech than this is most likely a few noisy sentences
# rather than a person, and is merged into the closest real speaker.
MIN_SPEAKER_SECONDS = 4.0
# Cosine distance above which two groups of sentences are different people.
# (Calibrated on real and synthetic voices; see tests/test_diarize.py.)
DEFAULT_DISTANCE_THRESHOLD = 0.32


class SpeakerEncoder:
    def __init__(self) -> None:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        self.session = ort.InferenceSession(
            hf_hub_download(MODEL_REPO, MODEL_FILE), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def embed(self, audio: np.ndarray) -> np.ndarray | None:
        """A unit-length voiceprint for 16 kHz mono float32 audio (None if it's under half a second)."""
        if len(audio) < SAMPLE_RATE // 2:
            return None
        import torch
        import torchaudio

        waveform = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))[None] * 32768
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform, num_mel_bins=80, frame_length=25, frame_shift=10, dither=0.0,
            sample_frequency=SAMPLE_RATE, window_type="hamming", use_energy=False,
        )
        fbank = fbank - fbank.mean(dim=0, keepdim=True)  # per-utterance mean normalization
        embedding = self.session.run(None, {self.input_name: fbank.numpy()[None].astype(np.float32)})[0][0]
        return embedding / (np.linalg.norm(embedding) + 1e-9)


@dataclass
class SpeakerAssignment:
    labels: list[int]  # speaker number (0, 1, ...) per input sentence, numbered by first appearance
    count: int


def _cosine_distance_matrix(vectors: np.ndarray) -> np.ndarray:
    return 1.0 - vectors @ vectors.T


def cluster_embeddings(
    embeddings: np.ndarray, durations: list[float], num_speakers: int | None = None,
    threshold: float = DEFAULT_DISTANCE_THRESHOLD,
) -> list[int]:
    """Groups unit-length voiceprints into speakers. With `num_speakers` unset the
    count is decided by `threshold`, and groups with too little speech are folded
    into their nearest neighbour."""
    from sklearn.cluster import AgglomerativeClustering

    n = len(embeddings)
    if n == 0:
        return []
    if n == 1 or num_speakers == 1:
        return [0] * n
    clustering = AgglomerativeClustering(
        n_clusters=num_speakers, distance_threshold=None if num_speakers else threshold,
        metric="cosine", linkage="average",
    )
    labels = clustering.fit_predict(embeddings)

    if not num_speakers:
        durations_arr = np.asarray(durations)
        while True:
            groups = sorted(set(labels))
            if len(groups) < 2:
                break
            speech = {g: durations_arr[labels == g].sum() for g in groups}
            tiny = min(groups, key=lambda g: speech[g])
            if speech[tiny] >= MIN_SPEAKER_SECONDS:
                break
            centroids = {g: embeddings[labels == g].mean(axis=0) for g in groups}
            others = [g for g in groups if g != tiny]
            nearest = max(others, key=lambda g: float(centroids[tiny] @ centroids[g]
                                                     / (np.linalg.norm(centroids[tiny]) * np.linalg.norm(centroids[g]) + 1e-9)))
            labels = np.where(labels == tiny, nearest, labels)
    return relabel_by_first_appearance([int(x) for x in labels])


def relabel_by_first_appearance(labels: list[int]) -> list[int]:
    order: dict[int, int] = {}
    for label in labels:
        order.setdefault(label, len(order))
    return [order[label] for label in labels]


def assign_speakers(
    units, audio: np.ndarray, encoder: SpeakerEncoder, num_speakers: int | None = None,
    threshold: float = DEFAULT_DISTANCE_THRESHOLD,
) -> SpeakerAssignment:
    """Speaker number for each unit (objects with .start/.end, seconds into `audio`)."""
    embeddings: list[np.ndarray | None] = []
    for unit in units:
        clip = audio[int(unit.start * SAMPLE_RATE): int(unit.end * SAMPLE_RATE)]
        reliable = (unit.end - unit.start) >= MIN_EMBED_SECONDS
        embeddings.append(encoder.embed(clip) if reliable else None)

    reliable_idx = [i for i, e in enumerate(embeddings) if e is not None]
    if not reliable_idx:
        return SpeakerAssignment([0] * len(units), 1)
    matrix = np.stack([embeddings[i] for i in reliable_idx])
    durations = [units[i].end - units[i].start for i in reliable_idx]
    clustered = cluster_embeddings(matrix, durations, num_speakers, threshold)

    labels: list[int | None] = [None] * len(units)
    for i, label in zip(reliable_idx, clustered):
        labels[i] = label
    centroids = {g: matrix[[k for k, l in enumerate(clustered) if l == g]].mean(axis=0) for g in set(clustered)}

    for i, unit in enumerate(units):
        if labels[i] is not None:
            continue
        clip = audio[int(unit.start * SAMPLE_RATE): int(unit.end * SAMPLE_RATE)]
        short = encoder.embed(clip)
        if short is not None:  # a short sentence: closest known voice
            labels[i] = max(centroids, key=lambda g: float(short @ centroids[g]))
        else:  # too short to fingerprint at all: whoever was speaking just before
            labels[i] = labels[i - 1] if i > 0 and labels[i - 1] is not None else clustered[0]
    final = relabel_by_first_appearance([int(x) for x in labels])
    return SpeakerAssignment(final, max(final) + 1)
