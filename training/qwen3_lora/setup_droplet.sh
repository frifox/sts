#!/usr/bin/env bash
# One-time environment setup for LoRA fine-tuning Qwen3-TTS on a fresh
# DigitalOcean GPU droplet (H100/H200). Run this once per droplet, then use
# prepare_dataset.py and train.sh.
#
# Pins the Qwen3-TTS checkout to the commit the LoRA companion patch
# (github.com/cheeweijie/qwen3-tts-lora-finetuning) is tested against, so
# `git apply` has a matching base to patch.
set -euo pipefail

WORK_DIR="${WORK_DIR:-$HOME/qwen3-lora}"
QWEN_COMMIT="0c6a7cbb6c8421a46332f8c2434c7825c4c855ef"

mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

echo "== system packages =="
sudo apt-get update -qq
sudo apt-get install -y -qq ffmpeg sox libsox-fmt-all git python3-venv python3-pip

echo "== python venv =="
python3 -m venv venv
source venv/bin/activate
pip install -U pip

echo "== python packages =="
pip install -U torch torchaudio
pip install -U qwen-tts peft accelerate safetensors soundfile librosa faster-whisper
pip install -U flash-attn --no-build-isolation || echo "flash-attn build failed - will fall back to eager attention (slower, still correct)"

echo "== Qwen3-TTS (pinned commit) =="
if [ ! -d Qwen3-TTS ]; then
  git clone https://github.com/QwenLM/Qwen3-TTS.git
fi
(cd Qwen3-TTS && git fetch --depth 1 origin "$QWEN_COMMIT" && git checkout "$QWEN_COMMIT")

echo "== LoRA companion patch (community, Apache-2.0, unofficial) =="
if [ ! -d qwen3-tts-lora-finetuning ]; then
  git clone https://github.com/cheeweijie/qwen3-tts-lora-finetuning.git
fi
QWEN_DIR="$WORK_DIR/Qwen3-TTS" bash qwen3-tts-lora-finetuning/scripts/apply_patches.sh

echo
echo "Setup complete. Verify the GPU is visible:"
echo "  nvidia-smi"
echo "Then activate the environment for future shells:"
echo "  source $WORK_DIR/venv/bin/activate"
