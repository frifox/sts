# Qwen3-TTS LoRA training (rented GPU droplet)

Trains a real, reusable per-speaker LoRA adapter for Qwen3-TTS from a
recorded speech file - the "true LoRA training" path, as opposed to the live
pipeline's zero-shot clone caching (`sts.tts.voice_library`, trained from a
short 18s live sample). This needs far more data and compute than a live
session gives you, hence: record separately, train on a rented GPU, then
import the result back into the live pipeline's voice library.

**This is not officially supported by the Qwen team.** Qwen3-TTS's own
`finetuning/` only does full fine-tuning (`sft_12hz.py`, all parameters, no
LoRA). LoRA support here comes from a third-party community patch,
[cheeweijie/qwen3-tts-lora-finetuning](https://github.com/cheeweijie/qwen3-tts-lora-finetuning)
(Apache-2.0), pinned to a specific upstream Qwen3-TTS commit so the patch
applies cleanly. That patch also fixes a real bug in the *official* trainer
(a double label-shift that makes speech progressively speed up each epoch),
so even the "vanilla" full-fine-tune path is better off patched. Read that
repo's README before a real training run - it documents several other
model-inherent pitfalls (wrong default learning rate, mandatory 24kHz audio,
LoRA scale near 1.0 being over-steered, occasional EOS/hang failures) that
these scripts already account for, but are worth understanding.

## Why train here and run elsewhere

Full-precision LoRA training needs enough VRAM for the base model **and**
backprop/optimizer state - an H100 (80GB) or H200 (141GB) makes that trivial
and fast. Using the *trained* adapter for inference is a different, far
cheaper workload (a forward pass through the same base model, same as
zero-shot cloning already requires) - a consumer GPU (RTX 3090/4090-class)
handles that fine. So: train here, then copy the small adapter directory
over to wherever the live pipeline actually runs.

## Workflow

1. **Create the droplet.** A DigitalOcean GPU Droplet (H100 or H200),
   Ubuntu with the NVIDIA/CUDA AI/ML image. Upload your recorded speaker
   audio (mp3 or anything ffmpeg reads) to it, e.g.:
   ```bash
   scp sermon.mp3 root@<droplet-ip>:~/
   ```

2. **One-time setup:**
   ```bash
   git clone <this repo> && cd sts/training/qwen3_lora
   bash setup_droplet.sh
   source ~/qwen3-lora/venv/bin/activate
   ```

3. **Prepare the dataset** - VAD-segments the recording into utterances,
   transcribes each with faster-whisper, and writes the training JSONL:
   ```bash
   python prepare_dataset.py ~/sermon.mp3 \
     --speaker-name pastor_ivan \
     --output-dir ~/qwen3-lora/data \
     --language ru
   ```
   Aim for a recording with **at least 10-15 minutes** of clean, single-speaker
   audio (the community repo's own real run used a similarly sized corpus);
   more/cleaner audio generally means a better adapter. This is the "10-30
   min" fine-tuning data requirement mentioned earlier in this project's
   history - well beyond the 18s zero-shot sample the live "Train voice"
   button captures.

4. **Train:**
   ```bash
   DATASET_DIR=~/qwen3-lora/data SPEAKER_NAME=pastor_ivan bash train.sh
   ```
   Writes checkpoints to `~/qwen3-lora/data/adapter/checkpoint-epoch-N/`.
   Defaults (`EPOCHS=10`, `LR=2e-6`) come from the companion repo's own
   findings, not the (buggier/higher-LR) upstream defaults. Expect this to
   take well under an hour on an H100/H200 for a single-speaker LoRA run,
   though actual time scales with dataset size and epoch count.

5. **Sanity-check before downloading anything:**
   ```bash
   CHECKPOINT_DIR=~/qwen3-lora/data/adapter/checkpoint-epoch-9 \
     SPEAKER_NAME=pastor_ivan bash test_inference.sh
   ```
   Listen to `test_output.wav`. Try a couple of `LORA_SCALE` values (0.2 to
   0.5) - 1.0 is documented to be over-steered - and pick the epoch/scale
   that sounds best before packaging.

6. **Package and download:**
   ```bash
   bash package_adapter.sh ~/qwen3-lora/data/adapter/checkpoint-epoch-9 pastor-ivan
   # on your local machine:
   scp root@<droplet-ip>:~/qwen3-lora/qwen3-tts-lora-finetuning/pastor-ivan-qwen3-lora.tar.gz .
   ```

7. **Destroy the droplet.** Nothing left to do on it - GPU droplets bill by
   the hour, so tear it down once you have the adapter.

8. **Import into the live pipeline**, on whatever machine actually runs
   `sts --tts-backend qwen3`:
   ```bash
   tar -xzf pastor-ivan-qwen3-lora.tar.gz
   sts-import-voice "Pastor Ivan" pastor_ivan ./checkpoint-epoch-9 \
     --lora-scale 0.3 --voice-library-dir /path/to/voices
   ```
   "Pastor Ivan" then shows up in the caption page's speaker dropdown like
   any other trained voice, right alongside zero-shot clones - live
   selection is a fresh model load (a few seconds), not a re-run of any of
   the steps above.

## Data format reference

`prepare_dataset.py` writes the schema Qwen3-TTS's official
`finetuning/prepare_data.py` expects - one JSON object per line:

```json
{"audio": "/path/to/utt0001.wav", "text": "...transcript...", "ref_audio": "/path/to/shared_ref.wav"}
```

`ref_audio` is deliberately the *same* clip for every row (the official
docs' own recommendation, for speaker-embedding consistency across the
dataset) - `prepare_dataset.py` picks the ~10s segment closest to that
target duration as the shared reference automatically.
