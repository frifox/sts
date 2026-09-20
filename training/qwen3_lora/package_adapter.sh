#!/usr/bin/env bash
# Packages a chosen checkpoint directory for download off the droplet.
#
#   bash package_adapter.sh ./data/adapter/checkpoint-epoch-9 pastor-ivan
#   scp root@<droplet-ip>:~/qwen3-lora/pastor-ivan-qwen3-lora.tar.gz .
#
# On the machine that will run the live pipeline:
#   tar -xzf pastor-ivan-qwen3-lora.tar.gz
#   sts-import-voice "Pastor Ivan" pastor_ivan ./checkpoint-epoch-9 \
#     --voice-library-dir /path/to/sts/voices
set -euo pipefail

CHECKPOINT_DIR="${1:?Usage: package_adapter.sh <checkpoint_dir> <speaker_id>}"
SPEAKER_ID="${2:?Usage: package_adapter.sh <checkpoint_dir> <speaker_id>}"

OUT="${SPEAKER_ID}-qwen3-lora.tar.gz"
tar -czf "$OUT" -C "$(dirname "$CHECKPOINT_DIR")" "$(basename "$CHECKPOINT_DIR")"

echo "Packaged $OUT ($(du -h "$OUT" | cut -f1))"
echo "Download with: scp \$(whoami)@\$(curl -s ifconfig.me):$(pwd)/$OUT ."
