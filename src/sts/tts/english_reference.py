"""Whether a cloned Qwen3 voice speaks through an English reference, and where
that reference is cached - see Qwen3TTSSynthesizer._english_reference for what
it is. Import-light (no model): the pipeline checks it to make a missing
reference in a helper process while the current voice keeps speaking."""

import hashlib
from pathlib import Path

from sts.tts.accent import ACCENT_DONORS

# What a voice cloned from a non-English reference first says in English: about
# 11 s, plain statements and a list, ending on a period.
ENGLISH_BRIDGE_TEXT = (
    "Good morning, brothers and sisters. Today we will read together from the Gospel, "
    "and we will think about what these words mean for our lives. Let us listen carefully, and let us pray."
)
# What an accent donor (sts.tts.accent, TTSConfig.qwen3_accent_donor) says for it.
DONOR_TEXT = (
    "Welcome, everyone. It's good to see all of you here this morning. "
    "We're going to open the Bible together, and I want us to listen closely to what it says."
)


def transcript_is_english(text: str | None) -> bool:
    """True for a Latin-script transcript; False for Cyrillic/other scripts or no transcript."""
    if not text:
        return False
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and sum(c.isascii() for c in letters) / len(letters) > 0.8


def clone_conditioning(mode: str, ref_text: str | None) -> str:
    """How a reference clip conditions the voice: "icl" (the clip + its
    transcript), "bridge" (an English clip in the voice + its English text)
    or "xvector" (the speaker embedding alone)."""
    if ref_text is None or mode == "xvector":
        return "xvector"
    if mode == "icl" or transcript_is_english(ref_text):
        return "icl"
    return "bridge"


def accent_donor(cfg) -> str:
    return cfg.qwen3_accent_donor if cfg.qwen3_accent_donor in ACCENT_DONORS else ""


def cached_reference_path(cfg, reference_wav_path) -> Path:
    """Where the English reference for this clip (and the configured model and
    accent donor) is kept, whether or not it has been made yet."""
    donor = accent_donor(cfg)
    source = Path(reference_wav_path).read_bytes()
    recipe = ENGLISH_BRIDGE_TEXT + (donor + DONOR_TEXT if donor else "") + cfg.qwen3_model_name
    key = hashlib.sha1(source + recipe.encode()).hexdigest()[:16]
    return Path(cfg.voice_library_dir) / ".english-references" / f"{key}-v3.wav"


def uses_english_reference(cfg, ref_text: str | None) -> bool:
    """Whether the TTS backend clones this voice from its English reference:
    Qwen3 per its clone mode (clone_conditioning), Drama 3 whenever the
    transcript isn't English (it needs a transcript, and the English came out
    Russian-paced from a Russian one)."""
    if cfg.backend == "qwen3":
        return clone_conditioning(cfg.qwen3_clone_mode, ref_text) == "bridge"
    if cfg.backend == "drama3":
        return not transcript_is_english(ref_text)
    return False


def needs_english_reference(cfg, reference_wav_path, ref_text: str | None) -> bool:
    """Cloning this clip will first have to make its English reference (tens of
    seconds of Qwen3 synthesis)."""
    return uses_english_reference(cfg, ref_text) and not cached_reference_path(cfg, reference_wav_path).is_file()
