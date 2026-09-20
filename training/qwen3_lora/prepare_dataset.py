"""Turns one recorded-speaker audio file into a Qwen3-TTS LoRA training set:
VAD-segmented into utterances, transcribed with faster-whisper, resampled to
the 24kHz mono the Qwen3-TTS codec pipeline requires, and written out as
train/val JSONL in the official finetuning schema (`audio`, `text`,
`ref_audio` per line - see ../../README.md's data format section).

Run on the training droplet after setup_droplet.sh, before train.sh:

    python prepare_dataset.py sermon.mp3 --speaker-name pastor_ivan --output-dir ./data

Silero VAD only supports 8kHz/16kHz analysis (not 24kHz), so speech is
detected on a 16kHz copy and the resulting timestamps are scaled up to slice
the actual training clips from full-quality 24kHz audio.
"""

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import soundfile as sf
import torch
from faster_whisper import WhisperModel

SAMPLE_RATE = 24_000  # Qwen3-TTS codec pipeline requires exactly this
VAD_SAMPLE_RATE = 16_000  # what Silero VAD actually supports


def convert_to_wav(input_path: Path, output_path: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(input_path), "-ar", str(SAMPLE_RATE), "-ac", "1", str(output_path)],
        check=True,
        capture_output=True,
    )


def find_speech_segments(master_wav_24k: Path, min_seconds: float, max_seconds: float):
    model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
    get_speech_timestamps, _, read_audio, *_ = utils

    audio_16k = read_audio(str(master_wav_24k), sampling_rate=VAD_SAMPLE_RATE)
    timestamps_16k = get_speech_timestamps(
        audio_16k,
        model,
        sampling_rate=VAD_SAMPLE_RATE,
        min_speech_duration_ms=int(min_seconds * 1000),
        max_speech_duration_s=max_seconds,
        speech_pad_ms=200,
    )

    audio_24k, sr = sf.read(master_wav_24k, dtype="float32")
    if sr != SAMPLE_RATE:
        sys.exit(f"expected {master_wav_24k} at {SAMPLE_RATE}Hz, got {sr}Hz")

    scale = SAMPLE_RATE / VAD_SAMPLE_RATE
    timestamps_24k = [
        {"start": int(t["start"] * scale), "end": int(t["end"] * scale)} for t in timestamps_16k
    ]
    return audio_24k, timestamps_24k


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_audio", help="Recorded speaker file (mp3, wav, ... - anything ffmpeg reads)")
    parser.add_argument(
        "--speaker-name", required=True, help="Internal speaker name for training (passed as --speaker_name to train.sh)"
    )
    parser.add_argument("--output-dir", default="./data")
    parser.add_argument("--language", default="ru", help="Whisper language code for transcription")
    parser.add_argument("--whisper-model", default="large-v3")
    parser.add_argument("--min-clip-seconds", type=float, default=4.0)
    parser.add_argument("--max-clip-seconds", type=float, default=20.0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    audio_dir = output_dir / "clips"
    audio_dir.mkdir(parents=True, exist_ok=True)

    master_wav = output_dir / "master_24k.wav"
    print(f"Converting {args.input_audio} -> {master_wav} (24kHz mono)...")
    convert_to_wav(Path(args.input_audio), master_wav)

    print("Running VAD segmentation...")
    audio, timestamps = find_speech_segments(master_wav, args.min_clip_seconds, args.max_clip_seconds)
    if not timestamps:
        sys.exit("No speech detected - check the input recording.")
    print(f"Found {len(timestamps)} speech segments.")

    print(f"Transcribing with faster-whisper ({args.whisper_model})...")
    whisper = WhisperModel(
        args.whisper_model, device=args.device, compute_type="float16" if args.device == "cuda" else "int8"
    )

    rows = []
    for i, ts in enumerate(timestamps):
        clip = audio[ts["start"] : ts["end"]]
        duration = len(clip) / SAMPLE_RATE
        clip_path = audio_dir / f"utt{i:04d}.wav"
        sf.write(clip_path, clip, SAMPLE_RATE)

        segments, _ = whisper.transcribe(str(clip_path), language=args.language)
        text = " ".join(s.text.strip() for s in segments).strip()
        if not text:
            clip_path.unlink()
            continue
        rows.append({"path": str(clip_path.resolve()), "text": text, "duration": duration})

    if not rows:
        sys.exit("No transcribable speech segments found.")
    print(f"Kept {len(rows)} transcribed clips ({sum(r['duration'] for r in rows) / 60:.1f} min total).")

    # Official docs strongly recommend the SAME ref_audio for every training
    # row - pick the clip closest to 10s as a clean, representative reference.
    ref_row = min(rows, key=lambda r: abs(r["duration"] - 10.0))
    ref_audio = ref_row["path"]
    print(f"Using {ref_audio} ({ref_row['duration']:.1f}s) as the shared reference clip for all rows.")

    random.Random(0).shuffle(rows)
    val_count = int(len(rows) * args.val_fraction) if len(rows) >= 10 else 0
    val_rows, train_rows = rows[:val_count], rows[val_count:]

    def write_jsonl(path: Path, split_rows: list[dict]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for r in split_rows:
                f.write(
                    json.dumps({"audio": r["path"], "text": r["text"], "ref_audio": ref_audio}, ensure_ascii=False)
                    + "\n"
                )

    write_jsonl(output_dir / "train_raw.jsonl", train_rows)
    if val_rows:
        write_jsonl(output_dir / "val_raw.jsonl", val_rows)

    print(f"\nWrote {len(train_rows)} train / {len(val_rows)} val rows to {output_dir}")
    print(f"Speaker name for training: {args.speaker_name}")
    print(f"\nNext: DATASET_DIR={output_dir} SPEAKER_NAME={args.speaker_name} bash train.sh")


if __name__ == "__main__":
    main()
