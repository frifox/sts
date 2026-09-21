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
import statistics
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
    speaker: int = 0  # who says it (0 unless --multiple identified several speakers)
    clarity: float | None = None  # how clear the original recording of it is (sts.audioquality), if measured


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
    best: tuple[tuple, int, int] | None = None
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
            run = units[i : j + 1]
            clarity = sum(u.clarity if u.clarity is not None else 0.0 for u in run) / len(run)
            # Long enough (10 s) is what matters; among runs that long, the clearest recording wins
            # (a cloned voice copies a muffled recording's muffled sound), then the best transcribed.
            score = (min(span, 10.0), clarity, sum(u.avg_logprob for u in run) / len(run), span)
            if best is None or score > best[0]:
                best = (score, i, j)
    if best is None:
        return None
    return best[1], best[2]


def pick_reference_units(
    units: list[Unit], indices: list[int], target_seconds: float = 14.0, max_seconds: float = 20.0,
    min_logprob: float = -0.6, max_unit_seconds: float = 12.0,
) -> list[int]:
    """Which of one speaker's sentences (`indices` into `units`) to build their
    voice-cloning reference from, returned in time order.

    The cloned voice copies the recording conditions of its reference (a muffled,
    distant recording gives a muffled, distant voice), so this prefers the
    clearest recordings among the confidently transcribed ones. Very long units
    come last: a 25 s "sentence" is usually several lines, maybe several people,
    and gives a blurry voiceprint. Collection stops at `target_seconds`. A
    speaker's sentences are rarely back to back, so unlike the single-speaker
    case this can join sentences from different places in the video."""
    def rank(i: int):
        length = units[i].end - units[i].start
        clarity = units[i].clarity if units[i].clarity is not None else -99.0
        return (units[i].avg_logprob >= min_logprob, length <= max_unit_seconds, clarity, length)

    chosen: list[int] = []
    total = 0.0
    for i in sorted(indices, key=rank, reverse=True):
        length = units[i].end - units[i].start
        if chosen and total + length > max_seconds:
            continue  # would overshoot; a shorter one may still fit
        chosen.append(i)
        total += length
        if total >= target_seconds:
            break
    return sorted(chosen)


def find_dull_takes(clarities: dict[int, float], margin_db: float = 4.0, min_takes: int = 3) -> tuple[float, list[int]]:
    """(bar, indices of the takes clearly duller than the rest), worst first.

    `clarities` maps a sentence index to the clarity (dB) of its synthesized clip,
    for one voice. Judged only against that voice's own median: clarity depends on
    the voice and on the sentence, so there is no absolute bar, and with fewer
    than `min_takes` takes there is nothing to compare against."""
    if len(clarities) < min_takes:
        return float("-inf"), []
    bar = statistics.median(clarities.values()) - margin_db
    return bar, sorted((i for i, c in clarities.items() if c < bar), key=lambda i: clarities[i])


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
                        help="Save the voice cloned from the video into the voice library under this name "
                        "(with --multiple: NAME 1, NAME 2, ...)")
    parser.add_argument("--multiple", action="store_true",
                        help="The video has several speakers: work out who says what and give each person their "
                        "own voice, cloned from their own speech")
    parser.add_argument("--speakers", type=int, default=None, metavar="N",
                        help="Number of speakers, when you know it (implies --multiple). Much more reliable than "
                        "letting the script guess")
    parser.add_argument("--speaker-threshold", type=float, default=None,
                        help="How different two voices must be to count as different people when guessing the "
                        "number of speakers (cosine distance; lower = splits more readily; default 0.32)")
    parser.add_argument("--retakes", type=int, default=2, metavar="N",
                        help="Sentences that sound duller (more muffled/distant) than the rest of that speaker's "
                        "are re-recorded up to N more times, keeping the clearest (default 2; 0 = off)")
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
    multiple = args.multiple or args.speakers is not None
    if multiple and (args.voice or args.no_clone):
        raise SystemExit("--multiple gives each speaker their own cloned voice, so it can't be combined with "
                         "--voice or --no-clone (they choose one voice for everyone).")
    if args.speakers is not None and args.speakers < 1:
        raise SystemExit("--speakers must be at least 1")
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

    # Dull takes are handled by the retake pass below (it can compare a sentence with all of that
    # speaker's others, including ones spoken later); the synthesizer's own in-line retries stay as configured.
    if args.qwen_quality_retries is None:
        cfg.tts.qwen3_quality_retries = 0

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
    from sts.audioquality import clarity_db

    for unit in units:
        unit.clarity = clarity_db(audio[int(unit.start * ASR_SAMPLE_RATE): int(unit.end * ASR_SAMPLE_RATE)], ASR_SAMPLE_RATE)

    # -- voice ----------------------------------------------------------
    speaker_voices: dict[int, tuple[Path, str] | None] = {}  # speaker -> (reference wav, its text); None = built-in voice
    if multiple:
        from sts.diarize import DEFAULT_DISTANCE_THRESHOLD, SpeakerEncoder, assign_speakers

        print("Working out who is speaking…", flush=True)
        assignment = assign_speakers(
            units, audio, SpeakerEncoder(), num_speakers=args.speakers,
            threshold=args.speaker_threshold or DEFAULT_DISTANCE_THRESHOLD,
        )
        for unit, label in zip(units, assignment.labels):
            unit.speaker = label
        guessed = "" if args.speakers else "  (a guess: pass --speakers N if that's wrong)"
        print(f"Found {assignment.count} speaker(s){guessed}:", flush=True)
        work = Path(tempfile.mkdtemp(prefix="sts_speakers_"))
        for speaker in range(assignment.count):
            mine = [i for i, u in enumerate(units) if u.speaker == speaker]
            picked = pick_reference_units(units, mine)
            speech = sum(units[i].end - units[i].start for i in mine)
            pad = int(0.1 * ASR_SAMPLE_RATE)
            gap = np.zeros(int(0.25 * ASR_SAMPLE_RATE), dtype=np.float32)
            pieces = []
            for i in picked:
                begin = max(0, int(units[i].start * ASR_SAMPLE_RATE) - pad)
                pieces += [audio[begin: int(units[i].end * ASR_SAMPLE_RATE) + pad], gap]
            clip = np.concatenate(pieces[:-1]) if pieces else np.zeros(0, dtype=np.float32)
            reference_seconds = len(clip) / ASR_SAMPLE_RATE
            first, last = units[mine[0]].start, units[mine[-1]].end
            line = (f"  Speaker {speaker + 1}: {len(mine)} sentence(s), {_fmt(first)}–{_fmt(last)}, "
                    f"{speech:.0f}s of speech; voice reference {reference_seconds:.0f}s")
            if reference_seconds < 2.0:
                speaker_voices[speaker] = None
                print(line + "  -> too little speech to clone, using the built-in voice", flush=True)
                continue
            if reference_seconds < 6.0:
                line += "  (short: the cloned voice may be a weak match)"
            print(line, flush=True)
            reference_path = work / f"speaker{speaker + 1}.wav"
            ref_text = " ".join(units[i].text for i in picked)
            write_wav(reference_path, clip, ASR_SAMPLE_RATE)
            speaker_voices[speaker] = (reference_path, ref_text)
            if args.save_voice:
                entry = library.save(f"{args.save_voice} {speaker + 1}", reference_path, ref_text)
                print(f"    Saved as voice '{entry.name}' in {cfg.tts.voice_library_dir}/.", flush=True)
    elif args.voice:
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

    clips: list[np.ndarray | None] = [None] * len(units)
    sample_rate = None
    synth_started = time.monotonic()
    # One speaker at a time, so each voice is set up once; the timeline below doesn't care about the order.
    order = sorted(range(len(units)), key=lambda i: (units[i].speaker, i))
    active_speaker = None

    def use_voice_of(speaker: int) -> None:
        nonlocal active_speaker
        if multiple and speaker != active_speaker:
            active_speaker = speaker
            voice = speaker_voices.get(speaker)
            if voice is None:
                tts.reset_voice()
            else:
                tts.clone_voice(str(voice[0]), voice[1])

    for done, i in enumerate(order, start=1):
        unit = units[i]
        use_voice_of(unit.speaker)
        try:
            clip, rate = tts.synthesize(unit.translation, pause_s=cfg.audio.pause_within_s)
            if sample_rate is None:
                sample_rate = rate
            elif rate != sample_rate:
                clip = librosa.resample(clip, orig_sr=rate, target_sr=sample_rate)
            clips[i] = clip
        except Exception as exc:  # one bad sentence leaves a gap rather than aborting an hour-long job
            print(f"  Warning: couldn't speak the sentence at {_fmt(unit.start)} ({exc!r}); leaving it silent.", flush=True)
        if done % 10 == 0 or done == len(order):
            elapsed = time.monotonic() - synth_started
            eta = elapsed / done * (len(order) - done)
            print(f"  spoken {done}/{len(order)}  (about {_fmt(eta)} left)", flush=True)
    if sample_rate is None:
        raise SystemExit("Speech synthesis produced no audio")

    # -- retake sentences that sound duller than the rest ----------------------
    # A cloned voice copies its reference's recording conditions, and some takes come out muffled
    # or distant-sounding. Compare each sentence with that speaker's own median and re-record the outliers.
    if args.retakes > 0:
        from sts.audioquality import clarity_db

        retaken = improved = 0
        for speaker in sorted({u.speaker for u in units}):
            mine = [i for i in range(len(units)) if units[i].speaker == speaker and clips[i] is not None
                    and len(clips[i]) > sample_rate]  # over a second: shorter clips can't be judged
            clarity = {i: clarity_db(clips[i], sample_rate) for i in mine}
            bar, dull = find_dull_takes(clarity)
            if not dull:
                continue
            use_voice_of(speaker)
            for i in dull:
                best, best_clarity, previous = clips[i], clarity[i], None
                for _ in range(args.retakes):
                    clip, rate = tts.synthesize(units[i].translation, pause_s=cfg.audio.pause_within_s)
                    if rate != sample_rate:
                        clip = librosa.resample(clip, orig_sr=rate, target_sr=sample_rate)
                    took = clarity_db(clip, sample_rate)
                    if took > best_clarity:
                        best, best_clarity = clip, took
                    if best_clarity >= bar or (previous is not None and abs(took - previous) < 1.5):
                        break  # clear enough now, or retakes come out equally dull: more tries won't help
                    previous = took
                retaken += 1
                improved += best_clarity > clarity[i]
                clips[i] = best
        if retaken:
            print(f"Re-recorded {retaken} sentence(s) that sounded duller than the rest; {improved} came out clearer.", flush=True)

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
            [{"start": round(u.start, 2), "end": round(u.end, 2), **({"speaker": u.speaker + 1} if multiple else {}),
              "ru": u.text, "en": u.translation} for u in units],
            ensure_ascii=False, indent=2))
        print(f"Wrote {transcript}", flush=True)

    print(f"Done in {_fmt(time.monotonic() - started)}: {output_path}", flush=True)


if __name__ == "__main__":
    main()
