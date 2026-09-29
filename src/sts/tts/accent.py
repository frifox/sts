"""The voices whose accent an English reference for a non-English cloned voice
can take - see Qwen3TTSSynthesizer._english_reference: "Default", the voice
bundled with the app (an English recording and its transcript), and the
American preset voices of Qwen3-TTS's CustomVoice model. Import-light: the
pipeline and the caption page list them without loading any model."""

BUNDLED_DONOR = "Default"  # src/sts/bundled_voices/default
ACCENT_DONORS = (BUNDLED_DONOR, "Aiden", "Ryan")
