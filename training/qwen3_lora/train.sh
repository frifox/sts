#!/usr/bin/env bash
# Extracts audio codes (official upstream step) then LoRA fine-tunes
# (community-patched trainer - see README.md for why the patch, not the
# vanilla upstream script, is used).
#
# Run after prepare_dataset.py has written train_raw.jsonl (+ val_raw.jsonl)
# into DATASET_DIR:
#
#   DATASET_DIR=./data SPEAKER_NAME=pastor_ivan bash train.sh
set -euo pipefail

WORK_DIR="${WORK_DIR:-$HOME/qwen3-lora}"
DATASET_DIR="${DATASET_DIR:?set DATASET_DIR to the --output-dir passed to prepare_dataset.py}"
SPEAKER_NAME="${SPEAKER_NAME:?set SPEAKER_NAME - must match the --speaker-name given to prepare_dataset.py}"
OUTPUT_DIR="${OUTPUT_DIR:-$DATASET_DIR/adapter}"

INIT_MODEL_PATH="${INIT_MODEL_PATH:-Qwen/Qwen3-TTS-12Hz-1.7B-Base}"
EPOCHS="${EPOCHS:-10}"                    # companion repo's guidance: single-speaker overfits past ~epoch 10
LR="${LR:-2e-6}"                          # NOT the upstream default (2e-5) - that produces noise/no-EOS output
BATCH_SIZE="${BATCH_SIZE:-8}"             # H100/H200 has headroom for LoRA; lower if you hit OOM
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"

source "$WORK_DIR/venv/bin/activate"
QWEN_DIR="$WORK_DIR/Qwen3-TTS"
TOKENIZER_MODEL_PATH="Qwen/Qwen3-TTS-Tokenizer-12Hz"

extract_codes() {
  local raw_jsonl="$1" out_jsonl="$2"
  python "$QWEN_DIR/finetuning/prepare_data.py" \
    --device cuda:0 \
    --tokenizer_model_path "$TOKENIZER_MODEL_PATH" \
    --input_jsonl "$raw_jsonl" \
    --output_jsonl "$out_jsonl"
}

echo "== extracting audio codes (train) =="
extract_codes "$DATASET_DIR/train_raw.jsonl" "$DATASET_DIR/train_with_codes.jsonl"

VAL_JSONL=""
if [ -f "$DATASET_DIR/val_raw.jsonl" ]; then
  echo "== extracting audio codes (val) =="
  extract_codes "$DATASET_DIR/val_raw.jsonl" "$DATASET_DIR/val_with_codes.jsonl"
  VAL_JSONL="$DATASET_DIR/val_with_codes.jsonl"
fi

echo "== LoRA fine-tuning =="
QWEN_DIR="$QWEN_DIR" \
INIT_MODEL_PATH="$INIT_MODEL_PATH" \
OUTPUT_DIR="$OUTPUT_DIR" \
TRAIN_JSONL="$DATASET_DIR/train_with_codes.jsonl" \
VAL_JSONL="$VAL_JSONL" \
LR="$LR" \
EPOCHS="$EPOCHS" \
BATCH_SIZE="$BATCH_SIZE" \
GRAD_ACCUM_STEPS="$GRAD_ACCUM_STEPS" \
SPEAKER_NAME="$SPEAKER_NAME" \
bash "$WORK_DIR/qwen3-tts-lora-finetuning/scripts/run_lora_train.sh"

echo
echo "Done. Adapter checkpoints:"
ls -d "$OUTPUT_DIR"/checkpoint-epoch-* 2>/dev/null || echo "  (none found - check the training log above for errors)"
echo
echo "Next: sanity-check a checkpoint with test_inference.sh, then package_adapter.sh to download it."
