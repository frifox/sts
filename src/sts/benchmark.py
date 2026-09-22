"""Benchmarks each pipeline stage - ASR (faster-whisper), translation (NLLB),
and TTS (both the xtts and qwen3 backends, whichever are installed) - using
the exact same sts model configs and device-selection logic (sts.devices)
that `sts` itself runs, so a number from this script reflects what the live
pipeline would actually do, not a synthetic benchmark of the frameworks in
isolation.

Point of the whole script: run it on different machines/OSes (an old GPU vs
a new one, a Mac's MPS backend, CPU-only) and diff the JSON it writes to see
where one is actually faster - this is what decided the GPU upgrade options
in the church-service deployment this was built for.

Usage:
  sts-benchmark                       # auto-detect device (cuda > mps > cpu), benchmark everything installed
  sts-benchmark --device cpu          # force a device
  sts-benchmark --asr-device cpu      # override just ASR (e.g. to match a VRAM-constrained deployment)
  sts-benchmark --repeats 5           # timed repetitions per stage (default 3, plus 1 untimed warmup call)
  sts-benchmark --skip-tts            # ASR + translation only
  sts-benchmark --tts-backend qwen3   # only benchmark one TTS backend instead of both
  sts-benchmark --out results.json
"""

import argparse
import json
import platform
import statistics
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

from sts.config import ASRConfig, TranslationConfig, TTSConfig
from sts.devices import detect_device, resolve_devices

BUNDLED_DIR = Path(__file__).parent / "bundled_voices" / "default"
REFERENCE_WAV = BUNDLED_DIR / "reference.wav"
REFERENCE_TEXT = (
    "Brothers and sisters, we gather today to remember God's grace, "
    "and to rest in the peace that He gives to all who trust in Him."
)

# Real sermon-style sentences (from an actual translated service), fixed so
# every run does the same amount of work regardless of machine.
TRANSLATE_SENTENCES = [
    "Братья и сестры, сегодня мы собрались, чтобы вспомнить благодать Божью.",
    "Если пребудете во Мне, и слова Мои в вас пребудут, то чего ни пожелаете, просите, и будет вам.",
    "Духовное возрастание - это то, что необходимо каждому христианину для принесения плода.",
    "Господь радуется даже маленьким шагам и маленьким успехам в нашей жизни.",
    "Мы должны регулярно питаться словом Божьим, как хлебом для нашей духовной жизни.",
]

TTS_TEXT = (
    "In the beginning God created the heavens and the earth, and His Spirit "
    "was moving over the surface of the waters, bringing order out of chaos."
)


@dataclass
class StageResult:
    stage: str
    device: str
    ok: bool
    detail: str = ""
    model_load_s: float | None = None
    warmup_s: float | None = None
    runs_s: list[float] = field(default_factory=list)

    @property
    def mean_s(self) -> float | None:
        return statistics.mean(self.runs_s) if self.runs_s else None

    @property
    def median_s(self) -> float | None:
        return statistics.median(self.runs_s) if self.runs_s else None

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "device": self.device,
            "ok": self.ok,
            "detail": self.detail,
            "model_load_s": self.model_load_s,
            "warmup_s": self.warmup_s,
            "runs_s": self.runs_s,
            "mean_s": self.mean_s,
            "median_s": self.median_s,
        }


def wav_duration_s(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


def rtf(processing_s: float, audio_s: float) -> float:
    """Real-time factor: processing time / audio duration. Below 1.0 is
    faster than real-time (good); above 1.0 means the stage falls behind a
    live speaker by that multiple."""
    return processing_s / audio_s


def format_rtf(value: float) -> str:
    return f"{value:.2f}x " + ("(faster than real-time)" if value < 1.0 else "(slower than real-time)")


def system_info() -> dict:
    info: dict = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor() or platform.machine(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
            major, minor = torch.cuda.get_device_capability(0)
            info["cuda_compute_capability"] = f"{major}.{minor}"
            info["cuda_total_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
        info["mps_available"] = torch.backends.mps.is_available()
    except Exception as exc:  # torch itself, or a property lookup, failing shouldn't kill the benchmark
        info["torch_error"] = repr(exc)
    return info


def benchmark_asr(device: str, model_size: str, repeats: int) -> StageResult:
    from sts.asr.whisper_asr import WhisperASR

    cfg = ASRConfig(model_size=model_size, device=device, compute_type="float16" if device == "cuda" else "int8")
    t0 = time.perf_counter()
    try:
        asr = WhisperASR(cfg)
    except Exception as exc:
        return StageResult("asr", device, False, detail=repr(exc))
    model_load_s = time.perf_counter() - t0

    audio_s = wav_duration_s(REFERENCE_WAV)

    t0 = time.perf_counter()
    asr.transcribe(str(REFERENCE_WAV))  # warmup: excludes first-call CUDA/CTranslate2 init from the timed runs
    warmup_s = time.perf_counter() - t0

    runs_s = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        asr.transcribe(str(REFERENCE_WAV))
        runs_s.append(time.perf_counter() - t0)

    mean_rtf = rtf(statistics.mean(runs_s), audio_s)
    return StageResult(
        "asr", device, True,
        detail=f"whisper {model_size}, {audio_s:.1f}s clip, RTF {format_rtf(mean_rtf)}",
        model_load_s=model_load_s, warmup_s=warmup_s, runs_s=runs_s,
    )


def benchmark_translation(device: str, repeats: int) -> StageResult:
    from sts.translate.translator import Translator

    cfg = TranslationConfig(device=device)
    t0 = time.perf_counter()
    try:
        translator = Translator(cfg)
    except Exception as exc:
        return StageResult("translate", device, False, detail=repr(exc))
    model_load_s = time.perf_counter() - t0

    def run_batch() -> None:
        for sentence in TRANSLATE_SENTENCES:
            translator.translate(sentence)

    t0 = time.perf_counter()
    run_batch()
    warmup_s = time.perf_counter() - t0

    runs_s = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        run_batch()
        runs_s.append(time.perf_counter() - t0)

    ms_per_sentence = statistics.mean(runs_s) / len(TRANSLATE_SENTENCES) * 1000
    return StageResult(
        "translate", device, True,
        detail=f"{len(TRANSLATE_SENTENCES)} sentences/batch, {ms_per_sentence:.0f} ms/sentence",
        model_load_s=model_load_s, warmup_s=warmup_s, runs_s=runs_s,
    )


def benchmark_tts(backend: str, device: str, repeats: int) -> StageResult:
    from sts.tts import create_synthesizer

    cfg = TTSConfig(backend=backend, device=device)
    cfg.use_backend_voices()
    # Matches the actual deployed config: only the cloned voice used for
    # synthesis needs loading here, not the CustomVoice preset model too -
    # loading both on an 8GB-class card alongside NLLB (already resident from
    # the translation benchmark) reliably OOMs, the same failure this flag
    # was added to sts itself to avoid.
    cfg.qwen3_preload_presets = False

    t0 = time.perf_counter()
    try:
        synth = create_synthesizer(cfg)
        synth.clone_voice(str(REFERENCE_WAV), ref_text=REFERENCE_TEXT)
    except SystemExit as exc:
        # create_synthesizer raises SystemExit (not ImportError) with a
        # formatted message when a backend's extra isn't installed - by
        # design, so `sts` itself prints a clean CLI error instead of a
        # traceback. SystemExit isn't an Exception, so it must be caught
        # separately or it kills this whole script instead of just skipping
        # the one backend.
        return StageResult(f"tts:{backend}", device, False, detail=f"not installed: {exc}")
    except Exception as exc:
        return StageResult(f"tts:{backend}", device, False, detail=repr(exc))
    model_load_s = time.perf_counter() - t0

    def run_once() -> tuple[float, float]:
        t0 = time.perf_counter()
        audio, sample_rate = synth.synthesize(TTS_TEXT)
        elapsed = time.perf_counter() - t0
        return elapsed, len(audio) / sample_rate

    warmup_s, _ = run_once()

    runs_s = []
    out_durations_s = []
    for _ in range(repeats):
        elapsed, out_s = run_once()
        runs_s.append(elapsed)
        out_durations_s.append(out_s)

    avg_out_s = statistics.mean(out_durations_s)
    mean_rtf = rtf(statistics.mean(runs_s), avg_out_s) if avg_out_s else None
    detail = f"{avg_out_s:.1f}s generated/call"
    if mean_rtf is not None:
        detail += f", RTF {format_rtf(mean_rtf)}"
    return StageResult(
        f"tts:{backend}", device, True, detail=detail,
        model_load_s=model_load_s, warmup_s=warmup_s, runs_s=runs_s,
    )


def first_line(text: str) -> str:
    return text.splitlines()[0] if text else ""


def is_not_installed(r: StageResult) -> bool:
    """True for a skipped stage whose only reason is a missing TTS backend
    (see benchmark_tts's SystemExit/ModuleNotFoundError handling) - as
    opposed to a genuine error worth elaborating on in the Details section."""
    return not r.ok and r.detail.startswith("not installed")


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    """Renders a simple aligned ASCII table - no extra dependency (tabulate/
    rich), just stdlib string formatting, matching the rest of the project."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row[:-1]):  # last column is free text, sized by its own header only
            widths[i] = max(widths[i], len(cell))

    def format_row(cells: list[str]) -> str:
        # The last column is left unpadded: it can be a long free-text message
        # (e.g. an install hint), and padding every row to its width would
        # leave every other row trailing in whitespace for no visible reason.
        padded = [cell.ljust(widths[i]) for i, cell in enumerate(cells[:-1])]
        return "  ".join([*padded, cells[-1]])

    lines = [format_row(headers), "  ".join("-" * w for w in widths)]
    lines.extend(format_row(row) for row in rows)
    return "\n".join(lines)


def result_row(r: StageResult) -> list[str]:
    if not r.ok:
        return [r.stage, r.device, "SKIP", "-", "-", "-", "-", first_line(r.detail)]
    return [
        r.stage, r.device, "OK",
        f"{r.model_load_s:.2f}", f"{r.warmup_s:.2f}",
        f"{r.mean_s:.2f}", f"{r.median_s:.2f}",
        first_line(r.detail),
    ]


def print_report(info: dict, results: list[StageResult]) -> None:
    print("=" * 72)
    print("System")
    print("=" * 72)
    for key, value in info.items():
        print(f"  {key}: {value}")

    print()
    print("=" * 72)
    print("Results")
    print("=" * 72)
    headers = ["Stage", "Device", "Status", "Load(s)", "Warmup(s)", "Mean(s)", "Median(s)", "Detail"]
    print(format_table(headers, [result_row(r) for r in results]))

    # Anything that failed, or whose detail didn't fit on one line, gets its
    # full message printed separately rather than cramming it into - or
    # silently truncating it out of - the table. A missing-backend skip is
    # the one exception: running this script via an explicit .venv-xtts or
    # .venv-qwen python already tells the caller which backend to expect, so
    # the install-hint paragraph would just be noise here.
    extra = [r for r in results if (not r.ok or "\n" in r.detail) and not is_not_installed(r)]
    if extra:
        print()
        print("Details:")
        for r in extra:
            print(f"  [{r.stage}] {r.detail}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark sts's ASR/translation/TTS stages on this machine")
    parser.add_argument("--device", choices=["cuda", "cpu", "mps"], default=None,
                        help="Compute device for translation/TTS, and ASR unless --asr-device overrides it. "
                        "Default: auto-detect (cuda, then mps, then cpu)")
    parser.add_argument("--asr-device", default=None, help="Override just the ASR device")
    parser.add_argument("--whisper-model", default="large-v3", help="faster-whisper model size (default: large-v3, matches sts's default)")
    parser.add_argument("--repeats", type=int, default=3, help="Timed repetitions per stage, after 1 untimed warmup call (default: 3)")
    parser.add_argument("--skip-asr", action="store_true")
    parser.add_argument("--skip-translate", action="store_true")
    parser.add_argument("--skip-tts", action="store_true")
    parser.add_argument("--tts-backend", choices=["xtts", "qwen3"], default=None,
                        help="Only benchmark this TTS backend (default: try both, skipping whichever isn't installed)")
    parser.add_argument("--out", default=None, help="Path for the JSON results file (default: benchmark_results/<hostname>_<timestamp>.json)")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    asr_device, translation_device, tts_device = resolve_devices(args.device or detect_device(), args.asr_device)
    info = system_info()
    info["asr_device"] = asr_device
    info["translation_device"] = translation_device
    info["tts_device"] = tts_device

    results: list[StageResult] = []

    if not args.skip_asr:
        print(f"Benchmarking ASR (whisper {args.whisper_model} on {asr_device})...", flush=True)
        results.append(benchmark_asr(asr_device, args.whisper_model, args.repeats))

    if not args.skip_translate:
        print(f"Benchmarking translation (NLLB on {translation_device})...", flush=True)
        results.append(benchmark_translation(translation_device, args.repeats))

    if not args.skip_tts:
        backends = [args.tts_backend] if args.tts_backend else ["xtts", "qwen3"]
        for backend in backends:
            print(f"Benchmarking TTS ({backend} on {tts_device})...", flush=True)
            results.append(benchmark_tts(backend, tts_device, args.repeats))

    print_report(info, results)

    out_path = Path(args.out) if args.out else None
    if out_path is None:
        out_dir = Path("benchmark_results")
        out_dir.mkdir(exist_ok=True)
        host = platform.node().split(".")[0] or "unknown-host"
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out_path = out_dir / f"{host}_{stamp}.json"
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {"system": info, "results": [r.to_dict() for r in results]}
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
