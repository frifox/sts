#!/usr/bin/env python3
"""Compares Russian -> English translation models on one sermon transcript.

    .venv-qwen/bin/python translate.py                  # every model below
    .venv-qwen/bin/python translate.py --models nllb    # only labels containing "nllb"

The first --minutes of --video are transcribed with GigaAM (the live
pipeline's ASR), split into sentences the way the live translator splits
them, and each sentence is translated by every model, one model loaded at a
time. The result is one text file, each Russian sentence followed by every
model's English:

    [ru] ...
      [nllb 3.3b] ...
      [hy-mt1.5 7b] ...

Run it from .venv-qwen: HY-MT1.5 needs transformers >= 4.56, and .venv-xtts
is pinned below 4.50. TranslateGemma is gated - accept its license on
huggingface.co and `hf auth login` first, or it is reported as an error.

Models are compared raw: no glossary, no short-phrase table, no context from
earlier sentences, and greedy decoding everywhere, so reruns are
reproducible. Each model's translations are cached in --out-dir, so a rerun
only translates what's missing (--force redoes them).

All the sizes together are ~200 GB of downloads, kept in the Hugging Face
cache. --delete-downloads removes each model this script downloaded once
it's done (models that were already cached are never deleted).
"""

import argparse
import gc
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from sts.devices import detect_device
from sts.translate.translator import split_sentences

SAMPLE_RATE = 16_000

# (label, kind, repo id). Order is the order in the output file.
MODELS = [
    ("nllb 600m distilled", "nllb", "facebook/nllb-200-distilled-600M"),
    ("nllb 1.3b distilled", "nllb", "facebook/nllb-200-distilled-1.3B"),
    ("nllb 1.3b", "nllb", "facebook/nllb-200-1.3B"),
    ("nllb 3.3b", "nllb", "facebook/nllb-200-3.3B"),
    ("madlad 3b", "madlad", "google/madlad400-3b-mt"),
    ("madlad 7b", "madlad", "google/madlad400-7b-mt"),
    ("madlad 10b", "madlad", "google/madlad400-10b-mt"),
    ("hy-mt1.5 1.8b", "hymt", "tencent/HY-MT1.5-1.8B"),
    ("hy-mt1.5 7b", "hymt", "tencent/HY-MT1.5-7B"),
    ("translategemma 4b", "gemma", "google/translategemma-4b-it"),
    ("translategemma 12b", "gemma", "google/translategemma-12b-it"),
    ("translategemma 27b", "gemma", "google/translategemma-27b-it"),
]


def extract_audio(video: Path, minutes: float) -> np.ndarray:
    """The first `minutes` of the video's audio, 16 kHz mono float32."""
    raw = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-t", str(minutes * 60), "-i", str(video),
         "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"],
        check=True, capture_output=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def transcribe(video: Path, minutes: float, device: str, cache: Path) -> list[str]:
    if cache.exists():
        print(f"[asr] using cached transcript {cache}")
        return json.loads(cache.read_text())
    from sts.asr.gigaam_asr import GigaAMASR
    from sts.config import ASRConfig

    print(f"[asr] extracting the first {minutes:g} min of {video}")
    audio = extract_audio(video, minutes)
    print(f"[asr] transcribing {len(audio) / SAMPLE_RATE:.0f} s with GigaAM on {device}")
    asr = GigaAMASR(ASRConfig(device=device))
    sentences = split_sentences(asr.transcribe(audio))
    asr.close()
    cache.write_text(json.dumps(sentences, ensure_ascii=False, indent=1))
    return sentences


def max_new_tokens(input_tokens: int) -> int:
    return min(512, input_tokens * 3 + 20)


class NLLB:
    def __init__(self, repo: str, device: str):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(repo, src_lang="rus_Cyrl")
        # fp32 off cuda, like the live Translator.
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.model = AutoModelForSeq2SeqLM.from_pretrained(repo, torch_dtype=dtype).to(device).eval()
        self.target = self.tokenizer.convert_tokens_to_ids("eng_Latn")

    def translate(self, text: str) -> str:
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **inputs, forced_bos_token_id=self.target, do_sample=False,
            max_new_tokens=max_new_tokens(inputs["input_ids"].shape[1]),
        )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)


class MADLAD:
    def __init__(self, repo: str, device: str):
        from transformers import T5ForConditionalGeneration, T5Tokenizer

        self.device = device
        self.tokenizer = T5Tokenizer.from_pretrained(repo)
        # bf16 is what T5 was trained in; fp16 overflows.
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        self.model = T5ForConditionalGeneration.from_pretrained(repo, torch_dtype=dtype).to(device).eval()

    def translate(self, text: str) -> str:
        inputs = self.tokenizer(f"<2en> {text}", return_tensors="pt").to(self.device)
        out = self.model.generate(
            **inputs, do_sample=False, max_new_tokens=max_new_tokens(inputs["input_ids"].shape[1]),
        )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)


class HYMT:
    """Prompt from the model card (the non-Chinese source form). The card
    recommends sampling (top_k 20, top_p 0.6, temperature 0.7); greedy here
    for reproducible comparisons, keeping its repetition_penalty."""

    PROMPT = "Translate the following segment into English, without additional explanation.\n\n{text}"

    def __init__(self, repo: str, device: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(repo)
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=dtype).to(device).eval()

    def translate(self, text: str) -> str:
        messages = [{"role": "user", "content": self.PROMPT.format(text=text)}]
        ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self.device)
        out = self.model.generate(
            ids, do_sample=False, repetition_penalty=1.05, max_new_tokens=max_new_tokens(ids.shape[1] // 2),
        )
        return self.tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


class TranslateGemma:
    def __init__(self, repo: str, device: str):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.device = device
        self.dtype = torch.float32 if device == "cpu" else torch.bfloat16
        self.processor = AutoProcessor.from_pretrained(repo)
        self.model = AutoModelForImageTextToText.from_pretrained(repo, torch_dtype=self.dtype).to(device).eval()

    def translate(self, text: str) -> str:
        messages = [{"role": "user", "content": [
            {"type": "text", "source_lang_code": "ru", "target_lang_code": "en", "text": text},
        ]}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(self.device, dtype=self.dtype)
        input_len = inputs["input_ids"].shape[1]
        out = self.model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens(input_len // 2))
        return self.processor.decode(out[0][input_len:], skip_special_tokens=True).strip()


KINDS = {"nllb": NLLB, "madlad": MADLAD, "hymt": HYMT, "gemma": TranslateGemma}


def hub_cache_dir(repo: str) -> Path:
    from huggingface_hub.constants import HF_HUB_CACHE

    return Path(HF_HUB_CACHE) / f"models--{repo.replace('/', '--')}"


def free_memory(device: str) -> None:
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


def run_model(label: str, kind: str, repo: str, sentences: list[str], device: str, delete_downloads: bool) -> dict:
    was_cached = hub_cache_dir(repo).exists()
    print(f"\n[{label}] {repo}{'' if was_cached else ' (downloading)'}")
    model = None
    try:
        t0 = time.monotonic()
        model = KINDS[kind](repo, device)
        load_s = time.monotonic() - t0
        translations, t0 = [], time.monotonic()
        with torch.inference_mode():
            for i, sentence in enumerate(sentences, 1):
                translations.append(model.translate(sentence))
                print(f"  {i}/{len(sentences)} {translations[-1][:100]}")
        return {
            "repo": repo, "load_s": load_s,
            "per_sentence_s": (time.monotonic() - t0) / max(1, len(sentences)),
            "translations": translations,
        }
    except Exception as exc:
        print(f"  failed: {exc!r}")
        return {"repo": repo, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        del model
        free_memory(device)
        if not was_cached and delete_downloads and hub_cache_dir(repo).exists():
            shutil.rmtree(hub_cache_dir(repo))
            print(f"  removed the downloaded weights ({hub_cache_dir(repo)})")


def write_report(path: Path, sentences: list[str], results: dict[str, dict], video: Path, minutes: float) -> None:
    width = max(len(label) for label in results)
    lines = [f"# {video.name}, first {minutes:g} min: {len(sentences)} sentences", "#"]
    for label, result in results.items():
        if "error" in result:
            lines.append(f"# {label:<{width}}  ERROR {result['error'].splitlines()[0][:200]}")
        else:
            lines.append(f"# {label:<{width}}  load {result['load_s']:6.1f} s   "
                         f"{result['per_sentence_s']:5.2f} s/sentence   {result['repo']}")
    lines.append("")
    for i, sentence in enumerate(sentences):
        lines.append(f"[ru] {sentence}")
        for label, result in results.items():
            if "error" not in result:
                lines.append(f"  [{label}] {result['translations'][i]}")
        lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--video", type=Path, default=Path("videos/k_neyman.mkv"))
    parser.add_argument("--minutes", type=float, default=5)
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"], default=None,
                        help="Default: auto-detect (cuda, then mps, then cpu)")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Only models whose label contains one of these (e.g. nllb 'hy-mt' '7b')")
    parser.add_argument("--out-dir", type=Path, default=Path("benchmark_results/translate"))
    parser.add_argument("--force", action="store_true", help="Re-translate models that already have cached results")
    parser.add_argument("--delete-downloads", action="store_true",
                        help="Delete each model's weights after it runs, if this script downloaded them "
                        "(all sizes together: ~200 GB)")
    args = parser.parse_args()

    import transformers

    if tuple(int(p) for p in transformers.__version__.split(".")[:2]) < (4, 56):
        sys.exit(f"transformers {transformers.__version__} is too old for HY-MT1.5 (needs >= 4.56): "
                 "run this with .venv-qwen/bin/python")

    device = args.device or detect_device()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.video.stem}_{args.minutes:g}min"
    sentences = transcribe(args.video, args.minutes, device, args.out_dir / f"{stem}_transcript.json")
    print(f"[asr] {len(sentences)} sentences")

    selected = [m for m in MODELS if not args.models or any(key in m[0] for key in args.models)]
    results_path = args.out_dir / f"{stem}_results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    for label, kind, repo in selected:
        cached = results.get(label)
        if cached and "error" not in cached and len(cached["translations"]) == len(sentences) and not args.force:
            print(f"\n[{label}] cached")
            continue
        results[label] = run_model(label, kind, repo, sentences, device, args.delete_downloads)
        results_path.write_text(json.dumps(results, ensure_ascii=False, indent=1))

    ordered = {label: results[label] for label, _, _ in MODELS if label in results}
    report = args.out_dir / f"{stem}.txt"
    write_report(report, sentences, ordered, args.video, args.minutes)
    print(f"\nwrote {report}")


if __name__ == "__main__":
    main()
