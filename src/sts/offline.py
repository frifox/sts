"""Translates a whole video: Russian speech in, the same video out with an
English voice track in place of the original audio.

    python translate.py --input talk.mp4 [--output talk-english.mp4] [--voice NAME]

Pipeline: ffmpeg extracts the audio -> Whisper transcribes it with timestamps
-> sentences are grouped -> NLLB translates -> the TTS backend speaks each one
in a cloned voice -> each clip is placed on the original timeline (sped up a
little where English runs longer than the Russian it replaces) -> ffmpeg swaps
the audio track, copying the video untouched.

The pure planning logic (grouping, choosing a reference clip, timeline
placement) lives at the top so it can be tested without any model or ffmpeg.
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sts.engine_args import add_engine_arguments, build_engine_config, installed_backends
from sts.log import vlog

ASR_SAMPLE_RATE = 16_000
SENTENCE_END = (".", "!", "?", "…")


# -- pure planning helpers -----------------------------------------------


@dataclass
class Unit:
    """One or more consecutive segments that form a sentence-sized chunk to translate and speak."""

    start: float
    end: float
    text: str
    avg_logprob: float
    translation: str = ""


def default_output_path(input_path: Path) -> Path:
    """`talk.mp4` -> `talk-english.mp4`, next to the input."""
    return input_path.with_name(f"{input_path.stem}-english{input_path.suffix}")


def group_units(segments, max_gap: float = 1.0, max_len: float = 20.0) -> list[Unit]:
    """Joins Whisper segments into sentence-sized units. Whisper often cuts
    mid-sentence at a pause; translating those halves separately gives bad
    English, so segments keep merging until the text ends a sentence, there is
    a long pause, or the unit gets too long."""
    units: list[Unit] = []
    current: list = []

    def flush() -> None:
        if not current:
            return
        text = " ".join(seg.text.strip() for seg in current).strip()
        logprob = sum(seg.avg_logprob for seg in current) / len(current)
        units.append(Unit(current[0].start, current[-1].end, text, logprob))
        current.clear()

    for seg in segments:
        if current:
            gap = seg.start - current[-1].end
            too_long = seg.end - current[0].start > max_len
            if gap > max_gap or too_long:
                flush()
        current.append(seg)
        if seg.text.rstrip().endswith(SENTENCE_END):
            flush()
    flush()
    return units


def pick_reference_window(
    units: list[Unit], max_seconds: float = 20.0, min_logprob: float = -0.6, max_gap: float = 2.0
) -> tuple[int, int] | None:
    """Indices (first, last) of the run of consecutive confident units that
    covers the most speech within `max_seconds`: the best clip to clone the
    speaker's voice from. Units more than `max_gap` seconds apart aren't joined
    (the clip would contain silence or someone else). Returns the longest
    available run even if it is short; None when no unit is usable."""
    best: tuple[float, float, int, int] | None = None
    for i, first in enumerate(units):
        if first.avg_logprob < min_logprob:
            continue
        for j in range(i, len(units)):
            unit = units[j]
            span = unit.end - first.start
            if span > max_seconds or unit.avg_logprob < min_logprob:
                break
            if j > i and unit.start - units[j - 1].end > max_gap:
                break
            score = (span, sum(u.avg_logprob for u in units[i : j + 1]) / (j - i + 1))
            if best is None or score > (best[0], best[1]):
                best = (span, score[1], i, j)
    if best is None:
        return None
    return best[2], best[3]


@dataclass
class Placement:
    start: float  # where the clip begins on the output timeline (seconds)
    rate: float  # playback speed-up applied to it (1.0 = untouched)


def plan_timeline(
    original_starts: list[float], durations: list[float], max_speedup: float = 1.3, min_gap: float = 0.15
) -> list[Placement]:
    """Places each translated clip at the time its original was spoken.

    English often runs longer than the Russian it replaces. When a clip would
    run into the next one, it is sped up (at most `max_speedup`). If it still
    overflows, later clips are pushed back, and they catch up again as soon as
    the original leaves a long enough pause."""
    placements: list[Placement] = []
    cursor = 0.0
    for i, (start_at, duration) in enumerate(zip(original_starts, durations)):
        start = max(start_at, cursor)
        rate = 1.0
        if i + 1 < len(original_starts):
            available = original_starts[i + 1] - start - min_gap
            if available <= 0:
                rate = max_speedup  # already running behind: catch up as far as allowed
            elif duration > available:
                rate = min(duration / available, max_speedup)
        placements.append(Placement(start, rate))
        cursor = start + duration / rate + min_gap
    return placements


# -- ffmpeg -----------------------------------------------------------------


def _require(tool: str) -> str:
    path = shutil.which(tool)
    if not path:
        raise SystemExit(f"{tool} not found. Install ffmpeg first (macOS: brew install ffmpeg; Debian: sudo apt install ffmpeg).")
    return path


def probe(input_path: Path) -> dict:
    out = subprocess.run(
        [_require("ffprobe"), "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(input_path)],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        raise SystemExit(f"ffprobe couldn't read {input_path}: {out.stderr.strip()}")
    return json.loads(out.stdout)


def extract_audio(input_path: Path, track: int) -> np.ndarray:
    """The chosen audio stream as mono 16 kHz float32, decoded by ffmpeg."""
    out = subprocess.run(
        [_require("ffmpeg"), "-v", "error", "-i", str(input_path), "-map", f"0:a:{track}", "-vn",
         "-ac", "1", "-ar", str(ASR_SAMPLE_RATE), "-f", "f32le", "pipe:1"],
        capture_output=True,
    )
    if out.returncode != 0:
        raise SystemExit(f"ffmpeg couldn't extract audio track {track}: {out.stderr.decode(errors='replace').strip()}")
    return np.frombuffer(out.stdout, dtype=np.float32)


def write_wav(path: Path | str, audio: np.ndarray, sample_rate: int) -> None:
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())


def mux(input_path: Path, english_wav: Path, output_path: Path, keep_original: bool, original_track: int) -> None:
    """Copies the video stream untouched and swaps in the English audio (optionally
    keeping the original as a second, non-default track)."""
    cmd = [_require("ffmpeg"), "-y", "-v", "error", "-i", str(input_path), "-i", str(english_wav),
           "-map", "0:v:0?", "-map", "1:a:0"]
    if keep_original:
        cmd += ["-map", f"0:a:{original_track}"]
    cmd += ["-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-metadata:s:a:0", "language=eng", "-metadata:s:a:0", "title=English (translated)"]
    if keep_original:
        cmd += ["-metadata:s:a:1", "language=rus", "-metadata:s:a:1", "title=Original"]
    cmd += ["-disposition:a:0", "default"]
    if output_path.suffix.lower() in (".mp4", ".mov", ".m4v"):
        cmd += ["-movflags", "+faststart"]
    cmd.append(str(output_path))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"ffmpeg failed to write {output_path}:\n{result.stderr.strip()}")


# -- the run ----------------------------------------------------------------


def _fmt(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}" if seconds >= 3600 else f"{seconds // 60}:{seconds % 60:02d}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Translate a video's Russian speech into English and replace its audio with a synthesized "
        "English voice (video stream is copied untouched)."
    )
    add_engine_arguments(parser)
    parser.add_argument("--input", required=True, type=Path, help="Video to translate (mp4, mkv, mov, ...)")
    parser.add_argument("--output", type=Path, default=None, help="Where to write the result (default: <input>-english.<ext>)")
    parser.add_argument(
        "--voice",
        default=None,
        help="Saved voice to speak with (name or id from the voice library, e.g. one trained in the live app). "
        "Default: clone the speaker from the video itself.",
    )
    parser.add_argument(
        "--no-clone",
        action="store_true",
        help="Don't clone anyone; use the built-in default voice (see --default-voice)",
    )
    parser.add_argument("--save-voice", default=None, metavar="NAME",
                        help="Save the voice cloned from the video into the voice library under this name")
    parser.add_argument("--max-speedup", type=float, default=1.3,
                        help="Fastest an English clip may be sped up to fit the time of the Russian it replaces (default 1.3)")
    parser.add_argument("--keep-original", action="store_true", help="Also keep the original audio as a second track")
    parser.add_argument("--audio-track", type=int, default=0, help="Which audio stream to translate (default 0, the first)")
    parser.add_argument("--save-transcript", action="store_true",
                        help="Also write <output>.json with the Russian text, English text and timings")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    input_path: Path = args.input
    if not input_path.is_file():
        raise SystemExit(f"Input not found: {input_path}")
    output_path: Path = args.output or default_output_path(input_path)
    if output_path.resolve() == input_path.resolve():
        raise SystemExit("--output must differ from --input")

    info = probe(input_path)
    audio_streams = [s for s in info["streams"] if s.get("codec_type") == "audio"]
    if not audio_streams:
        raise SystemExit(f"{input_path} has no audio track to translate")
    if args.audio_track >= len(audio_streams):
        raise SystemExit(f"--audio-track {args.audio_track} out of range: the file has {len(audio_streams)} audio track(s)")
    video_seconds = float(info["format"].get("duration") or 0)

    cfg, (asr_device, translation_device, tts_device) = build_engine_config(args, prefer_backend="qwen3")

    # Fail before loading any model if the chosen TTS backend isn't installed here
    # (the xtts and qwen3 extras live in separate virtualenvs).
    installed = installed_backends()
    if cfg.tts.backend not in installed:
        available = f"This environment has the {installed[0]} backend, so use --tts-backend {installed[0]}, or run " if installed else "Run "
        raise SystemExit(
            f"The '{cfg.tts.backend}' TTS backend isn't installed in this Python environment.\n"
            f"  {available}this from the environment that has it (pip install -e '.[{cfg.tts.backend}]')."
        )

    # The preset-voice model is only needed when no voice is being cloned.
    cfg.tts.qwen3_preload_presets = bool(args.no_clone and not args.voice)

    # Heavy imports happen only now, so `--help` and bad arguments fail fast.
    from sts.asr.whisper_asr import WhisperASR
    from sts.translate.translator import Translator
    from sts.tts import create_synthesizer
    from sts.tts.voice_library import BUNDLED_VOICES_DIR, VoiceLibrary

    started = time.monotonic()
    print(f"Loading models (whisper={cfg.asr.model_size} on {asr_device}, translation on {translation_device}, "
          f"tts={cfg.tts.backend} on {tts_device})…", flush=True)
    asr = WhisperASR(cfg.asr)
    translator = Translator(cfg.translation)
    tts = create_synthesizer(cfg.tts)
    library = VoiceLibrary(cfg.tts.voice_library_dir, bundled_dir=BUNDLED_VOICES_DIR)

    print(f"Extracting audio from {input_path.name}…", flush=True)
    audio = extract_audio(input_path, args.audio_track)
    total_seconds = len(audio) / ASR_SAMPLE_RATE
    if total_seconds < 1:
        raise SystemExit("The audio track is empty")

    print(f"Transcribing {_fmt(total_seconds)} of audio…", flush=True)
    last_report = [0.0]

    def report(done: float, total: float) -> None:
        if done - last_report[0] >= max(30.0, total / 20):
            last_report[0] = done
            print(f"  transcribed {_fmt(done)} / {_fmt(total)}  ({done / total:.0%})", flush=True)

    segments = asr.transcribe_segments(audio, on_progress=report)
    units = group_units(segments)
    if not units:
        raise SystemExit("No speech was found in the audio track")
    print(f"Found {len(units)} sentences of speech.", flush=True)

    # -- voice ----------------------------------------------------------
    if args.voice:
        wanted = args.voice.strip().lower()
        entry = next((e for e in library.list() if wanted in (e.name.lower(), e.speaker_id.lower())), None)
        if entry is None:
            names = ", ".join(e.name for e in library.list()) or "(library is empty)"
            raise SystemExit(f"No saved voice named {args.voice!r} in {cfg.tts.voice_library_dir}/. Available: {names}")
        tts.clone_from_entry(entry)
        print(f"Voice: saved voice '{entry.name}'.", flush=True)
    elif not args.no_clone:
        window = pick_reference_window(units)
        if window is None:
            print("Couldn't find a clean stretch of speech to clone from; using the default voice.", flush=True)
        else:
            first, last = window
            start, end = units[first].start, units[last].end
            clip = audio[int(start * ASR_SAMPLE_RATE): int(end * ASR_SAMPLE_RATE)]
            ref_text = " ".join(u.text for u in units[first: last + 1])
            reference_path = Path(tempfile.mkdtemp(prefix="sts_ref_")) / "reference.wav"
            write_wav(reference_path, clip, ASR_SAMPLE_RATE)
            tts.clone_voice(str(reference_path), ref_text)
            print(f"Voice: cloned from the speaker at {_fmt(start)}–{_fmt(end)} ({end - start:.0f}s).", flush=True)
            if end - start < 6:
                print("  Warning: that reference is short, so the cloned voice may be a weak match.", flush=True)
            if args.save_voice:
                entry = library.save(args.save_voice, reference_path, ref_text)
                print(f"  Saved as voice '{entry.name}' in {cfg.tts.voice_library_dir}/.", flush=True)
    else:
        print(f"Voice: built-in default ({cfg.tts.default_voice}).", flush=True)

    # -- translate ------------------------------------------------------
    print("Translating…", flush=True)
    for unit in units:
        unit.translation = translator.translate(unit.text)
    units = [u for u in units if u.translation.strip()]

    # -- synthesize -------------------------------------------------------
    print("Speaking the translation…", flush=True)
    import librosa

    clips: list[np.ndarray | None] = []
    sample_rate = None
    synth_started = time.monotonic()
    for index, unit in enumerate(units, start=1):
        try:
            clip, rate = tts.synthesize(unit.translation, pause_s=cfg.audio.pause_within_s)
            if sample_rate is None:
                sample_rate = rate
            elif rate != sample_rate:
                clip = librosa.resample(clip, orig_sr=rate, target_sr=sample_rate)
            clips.append(clip)
        except Exception as exc:  # one bad sentence leaves a gap rather than aborting an hour-long job
            print(f"  Warning: couldn't speak sentence {index} at {_fmt(unit.start)} ({exc!r}); leaving it silent.", flush=True)
            clips.append(None)
        if index % 10 == 0 or index == len(units):
            elapsed = time.monotonic() - synth_started
            eta = elapsed / index * (len(units) - index)
            print(f"  spoken {index}/{len(units)}  (about {_fmt(eta)} left)", flush=True)
    if sample_rate is None:
        raise SystemExit("Speech synthesis produced no audio")

    # -- assemble the timeline ------------------------------------------
    usable = [(u, c) for u, c in zip(units, clips) if c is not None and c.size]
    placements = plan_timeline([u.start for u, _ in usable], [len(c) / sample_rate for _, c in usable], args.max_speedup)
    end_time = max([total_seconds, video_seconds] + [p.start + len(c) / sample_rate / p.rate + 0.5
                                                     for p, (_, c) in zip(placements, usable)])
    track = np.zeros(int(end_time * sample_rate) + 1, dtype=np.float32)
    for placement, (_, clip) in zip(placements, usable):
        if placement.rate > 1.001:
            clip = librosa.effects.time_stretch(clip, rate=placement.rate)
        begin = int(placement.start * sample_rate)
        track[begin: begin + len(clip)] += clip[: len(track) - begin]
    peak = float(np.abs(track).max())
    if peak > 0.98:
        track *= 0.98 / peak  # avoid clipping where clips overlap or run hot

    sped_up = sum(p.rate > 1.001 for p in placements)
    pushed = sum(p.start > u.start + 1.0 for p, (u, _) in zip(placements, usable))
    print(f"Timeline: {sped_up} of {len(usable)} sentences sped up to fit; "
          f"{pushed} started more than 1 s late.", flush=True)

    # -- write ------------------------------------------------------------
    with tempfile.TemporaryDirectory(prefix="sts_translate_") as work:
        english_wav = Path(work) / "english.wav"
        write_wav(english_wav, track, sample_rate)
        print(f"Writing {output_path}…", flush=True)
        mux(input_path, english_wav, output_path, args.keep_original, args.audio_track)

    if args.save_transcript:
        transcript = output_path.with_suffix(".json")
        transcript.write_text(json.dumps(
            [{"start": round(u.start, 2), "end": round(u.end, 2), "ru": u.text, "en": u.translation} for u in units],
            ensure_ascii=False, indent=2))
        print(f"Wrote {transcript}", flush=True)

    print(f"Done in {_fmt(time.monotonic() - started)}: {output_path}", flush=True)


if __name__ == "__main__":
    main()
