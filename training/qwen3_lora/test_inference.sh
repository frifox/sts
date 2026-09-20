#!/usr/bin/env bash
# Quick sanity check of a trained checkpoint before downloading it.
# The companion repo's own advice: LoRA scale 1.0 is almost always
# over-steered - sweep 0.2/0.3/0.35/0.5 and listen before committing to one.
#
#   CHECKPOINT_DIR=./data/adapter/checkpoint-epoch-9 SPEAKER_NAME=pastor_ivan \
#     bash test_inference.sh
set -euo pipefail

WORK_DIR="${WORK_DIR:-$HOME/qwen3-lora}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:?set CHECKPOINT_DIR, e.g. ./data/adapter/checkpoint-epoch-9}"
SPEAKER_NAME="${SPEAKER_NAME:?set SPEAKER_NAME - must match what train.sh was run with}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-TTS-12Hz-1.7B-Base}"
TEXT="${TEXT:-This is a test of the trained voice.}"
LORA_SCALE="${LORA_SCALE:-0.3}"
OUT_WAV="${OUT_WAV:-test_output.wav}"

source "$WORK_DIR/venv/bin/activate"

QWEN_DIR="$WORK_DIR/Qwen3-TTS" \
BASE_MODEL="$BASE_MODEL" \
ADAPTER_DIR="$CHECKPOINT_DIR" \
SPEAKER_NAME="$SPEAKER_NAME" \
LORA_SCALE="$LORA_SCALE" \
TEXT="$TEXT" \
OUT_WAV="$OUT_WAV" \
bash "$WORK_DIR/qwen3-tts-lora-finetuning/scripts/run_lora_infer.sh"

echo "Wrote $OUT_WAV. Download and listen, then try other LORA_SCALE values (0.2-0.5) before picking one."
