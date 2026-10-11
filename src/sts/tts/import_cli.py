"""`sts-import-voice`: registers a LoRA adapter trained elsewhere (e.g. on a
rented GPU via `training/qwen3_lora/`) into the local voice library, so it
shows up in the caption page's speaker dropdown without retraining.
"""

import argparse

from sts.paths import default_voice_library_dir
from sts.tts.voice_library import VoiceLibrary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="Display name for the dropdown, e.g. 'Pastor Ivan'")
    parser.add_argument("speaker_name", help="Internal speaker name used at training time (--speaker_name)")
    parser.add_argument("adapter_dir", help="Path to the checkpoint-epoch-N directory to import")
    parser.add_argument("--lora-scale", type=float, default=0.3, help="LoRA scale to apply at inference")
    parser.add_argument("--voice-library-dir", default=default_voice_library_dir(),
                        help=f"Voice library directory (default: {default_voice_library_dir()})")
    args = parser.parse_args()

    library = VoiceLibrary(args.voice_library_dir)
    entry = library.save_lora_adapter(
        args.name, args.adapter_dir, speaker_name=args.speaker_name, lora_scale=args.lora_scale
    )
    print(f"Imported '{entry.name}' as speaker_id={entry.speaker_id!r} into {args.voice_library_dir}")


if __name__ == "__main__":
    main()
