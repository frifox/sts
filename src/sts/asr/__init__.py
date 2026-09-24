"""Speech recognition: Sber's GigaAM-v3 (see sts.asr.gigaam_asr), driven as
sentence-locking live captions by sts.transcript."""

from dataclasses import dataclass

from sts.config import ASRConfig


@dataclass
class TimedWord:
    """One recognized word (with any punctuation attached, e.g. "плода,")
    and when it was spoken, in seconds from the start of the audio given."""

    text: str
    start: float
    end: float


def create_asr(cfg: ASRConfig):
    try:
        from sts.asr.gigaam_asr import GigaAMASR
    except ImportError as exc:
        raise SystemExit(
            f"Speech recognition (GigaAM) isn't installed in this environment ({exc}).\n"
            f"  Its dependencies come with the project itself:  pip install -e ."
        ) from exc
    return GigaAMASR(cfg)
