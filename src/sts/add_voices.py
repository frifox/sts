"""`sts-add-voices`: turns video/audio files dropped into the voice library's
`new/` folder (on macOS ~/Library/Application Support/STS/voices/new/, the
library STS.app uses too - see sts.paths) into voices in the library - the batch equivalent of pressing Clone in the caption
page, with no training (zero-shot cloning, like Clone).

For each file (any format ffmpeg reads: mp4, mov, mkv, mp3, m4a, wav, ...):

1. decode it to 16 kHz mono;
2. pick the best ~18 s of speech (as long as a Clone capture): the stretch
   with the most continuous speech (Silero VAD), steady loudness and no
   clipping, skipping the first minute of a long recording (intros, music);
3. transcribe it (GigaAM), as Clone does;
4. save it to the library under the file's name ("Neyman Konstantin.mp4" ->
   "Neyman Konstantin"), where it shows in the caption page's Voice list;
5. for Qwen3-TTS, make the voice's English reference now - otherwise done on
   its first use (seconds) - with the base voice set in the caption page;
6. move the file to `new/done/`.

A file whose voice is already in the library is skipped (--replace redoes
it). One speaker per file: a second voice in the picked stretch (an
interviewer, a translator) would be cloned along with it - trim such files
first, or point at the right part with --start/--end.
"""

import argparse
import gc
import shutil
import subprocess
import sys
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sts.paths import default_voice_library_dir

SAMPLE_RATE = 16_000
MEDIA_SUFFIXES = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma",
                  ".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpg", ".mpeg", ".ts"}
MAX_GAP_S = 1.2  # a longer pause ends a stretch: the clip should sound like one continuous passage
SKIP_INTRO_S = 60.0  # of a long recording (see pick_clip)
MIN_SECONDS = 6.0  # less speech than this in the best stretch: the file is skipped


@dataclass
class Clip:
    start: float
    end: float
    speech_s: float

    @property
    def seconds(self) -> float:
        return self.end - self.start


def decode(path: Path, start: float | None = None, end: float | None = None) -> np.ndarray:
    """The file's audio as 16 kHz mono float32 (ffmpeg)."""
    ffmpeg = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
    command = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error"]
    if start is not None:
        command += ["-ss", str(start)]
    if end is not None:
        command += ["-to", str(end)]
    command += ["-i", str(path), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"]
    result = subprocess.run(command, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode(errors="replace").strip() or "ffmpeg failed")
    return np.frombuffer(result.stdout, dtype=np.float32).copy()


_vad = None


def speech_segments(audio: np.ndarray) -> list[tuple[float, float]]:
    """(start, end) seconds of each stretch of speech (Silero VAD)."""
    global _vad
    import torch

    if _vad is None:
        _vad = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True, verbose=False)
    model, (get_speech_timestamps, *_) = _vad
    stamps = get_speech_timestamps(torch.from_numpy(audio), model, sampling_rate=SAMPLE_RATE,
                                   min_silence_duration_ms=250, min_speech_duration_ms=250)
    return [(s["start"] / SAMPLE_RATE, s["end"] / SAMPLE_RATE) for s in stamps]


def level_db(audio: np.ndarray, start: float, end: float) -> float:
    part = audio[int(start * SAMPLE_RATE): int(end * SAMPLE_RATE)]
    return float(20 * np.log10(np.sqrt(np.mean(part ** 2)) + 1e-9)) if len(part) else -120.0


def pick_clip(segments: list[tuple[float, float]], audio: np.ndarray, target_s: float) -> Clip | None:
    """The best stretch of about `target_s` seconds of speech: consecutive
    speech segments with no pause longer than MAX_GAP_S between them, scored
    by how much of it is speech, how close it comes to `target_s` of it, how
    steady its loudness is (a change of mic or speaker shows as a jump), and
    how little is clipped. The first
    SKIP_INTRO_S of a recording long enough to spare it is left out."""
    duration = len(audio) / SAMPLE_RATE
    if duration > SKIP_INTRO_S + 4 * target_s:
        segments = [seg for seg in segments if seg[0] >= SKIP_INTRO_S]
    best: tuple[float, Clip] | None = None
    fallback: Clip | None = None
    for i in range(len(segments)):
        speech = 0.0
        j = i
        while j < len(segments) and speech < target_s:
            if j > i and segments[j][0] - segments[j - 1][1] > MAX_GAP_S:
                break
            speech += segments[j][1] - segments[j][0]
            j += 1
        clip = Clip(segments[i][0], segments[j - 1][1], speech)
        if clip.seconds > target_s * 1.4:  # ended in one long segment: cut it at a quiet moment
            clip = Clip(clip.start, quiet_point(audio, clip.start + target_s, clip.start + target_s * 1.4),
                        speech - (clip.seconds - target_s))
        if speech < target_s * 0.9:
            if speech >= MIN_SECONDS and (fallback is None or speech > fallback.speech_s):
                fallback = clip
            continue
        levels = [level_db(audio, s, e) for s, e in segments[i:j]]
        part = audio[int(clip.start * SAMPLE_RATE): int(clip.end * SAMPLE_RATE)]
        clipped = float(np.mean(np.abs(part) > 0.98)) if len(part) else 1.0
        shortfall = max(0.0, target_s - clip.speech_s)  # less than asked for
        score = clip.speech_s / clip.seconds - 0.05 * shortfall - 0.03 * float(np.std(levels)) - 50 * clipped
        if best is None or score > best[0]:
            best = (score, clip)
    return best[1] if best else fallback


def quiet_point(audio: np.ndarray, earliest: float, latest: float) -> float:
    """The quietest 100 ms between two times: where to cut mid-speech."""
    frame = SAMPLE_RATE // 10
    first, last = int(earliest * SAMPLE_RATE), min(len(audio), int(latest * SAMPLE_RATE))
    if last - first < frame:
        return latest
    energies = [np.mean(audio[k: k + frame] ** 2) for k in range(first, last - frame + 1, frame // 2)]
    return (first + int(np.argmin(energies)) * (frame // 2) + frame // 2) / SAMPLE_RATE


def write_wav(path: Path, audio: np.ndarray) -> None:
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(pcm16.tobytes())


def voice_name(path: Path) -> str:
    """The voice's name from its file's: "goldinov-venya.mp3" -> "Goldinov Venya"
    (a name typed all in lower case is capitalized; otherwise kept as written)."""
    name = " ".join(path.stem.replace("_", " ").replace("-", " ").split())
    return name.title() if name == name.lower() else name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--voice-library-dir", default=default_voice_library_dir(),
                        help=f"Voice library directory (default: {default_voice_library_dir()})")
    parser.add_argument("--input-dir", default=None, help="Where the files are (default: <voice library>/new)")
    parser.add_argument("--seconds", type=float, default=18.0, help="How much speech to clone from (default 18, as Clone)")
    parser.add_argument("--start", type=float, default=None, help="Only use the file from this many seconds in")
    parser.add_argument("--end", type=float, default=None, help="Only use the file up to this many seconds in")
    parser.add_argument("--replace", action="store_true", help="Redo voices already in the library")
    parser.add_argument("--no-english-reference", action="store_true",
                        help="Don't make Qwen3's English references now (they're made on first use)")
    parser.add_argument("--config-db", default="sts.db", help="The app's settings (the base voice for cloning)")
    parser.add_argument("--device", default=None, help="Device for speech recognition and TTS (default: auto)")
    args = parser.parse_args()

    from sts.config import ASRConfig
    from sts.devices import detect_device
    from sts.tts.voice_library import VoiceLibrary, _slugify

    library = VoiceLibrary(args.voice_library_dir)
    inbox = Path(args.input_dir) if args.input_dir else library.root / "new"
    inbox.mkdir(parents=True, exist_ok=True)
    done = inbox / "done"
    files = sorted(p for p in inbox.iterdir() if p.is_file() and p.suffix.lower() in MEDIA_SUFFIXES)
    if not files:
        print(f"Nothing to add: put video or audio files in {inbox}/ (one speaker each; the file name is the voice's name).")
        return
    device = args.device or detect_device()

    todo = []
    for path in files:
        name = voice_name(path)
        speaker_id = _slugify(name)
        if speaker_id in ("new", "done") or speaker_id.startswith("."):
            print(f"- {path.name}: pick another file name (\"{name}\" is reserved)")
            continue
        if library.get(speaker_id) is not None and not args.replace:
            print(f"- {path.name}: \"{name}\" is already a voice (--replace to redo it); skipped")
            continue
        todo.append((path, name))
    if not todo:
        return

    asr = None
    added = []
    for path, name in todo:
        print(f"\n{path.name} -> \"{name}\"")
        try:
            audio = decode(path, args.start, args.end)
        except Exception as exc:
            print(f"  couldn't read it: {exc}")
            continue
        print(f"  {len(audio) / SAMPLE_RATE / 60:.1f} min of audio")
        clip = pick_clip(speech_segments(audio), audio, args.seconds)
        if clip is None:
            print(f"  not enough continuous speech (needs at least {MIN_SECONDS:.0f} s); skipped")
            continue
        offset = args.start or 0.0
        print(f"  using {_clock(offset + clip.start)}-{_clock(offset + clip.end)} "
              f"({clip.speech_s:.1f} s of speech in {clip.seconds:.1f} s)")
        pad = int(0.1 * SAMPLE_RATE)
        part = audio[max(0, int(clip.start * SAMPLE_RATE) - pad): int(clip.end * SAMPLE_RATE) + pad]
        if asr is None:
            from sts.asr.gigaam_asr import GigaAMASR

            asr_cfg = ASRConfig()
            asr_cfg.device = device
            asr = GigaAMASR(asr_cfg)
        ref_text = asr.transcribe(part) or None
        print(f"  transcript: {ref_text or '(none - the voice will use its speaker embedding only)'}")
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "reference.wav"
            write_wav(wav, part)
            entry = library.save(name, wav, ref_text)
        done.mkdir(exist_ok=True)
        shutil.move(str(path), str(done / path.name))
        added.append(entry)
        print(f"  saved as voice \"{entry.name}\" ({library.root / entry.speaker_id})")

    if asr is not None:
        asr.close()
        del asr
        gc.collect()
    if added and not args.no_english_reference:
        make_english_references(added, args, device)
    print(f"\n{len(added)} voice(s) added" + (f"; files moved to {done}/" if added else ""))


def make_english_references(entries, args, device: str) -> None:
    """Qwen3-TTS: the English reference of each voice that clones through one
    (a non-English transcript - see Qwen3TTSSynthesizer._english_reference)."""
    from sts.config import TTSConfig
    from sts.engine_args import installed_backends
    from sts.persistent_config import PersistentConfig
    from sts.tts.qwen3 import clone_conditioning

    if "qwen3" not in installed_backends():
        return
    cfg = TTSConfig(backend="qwen3", device=device)
    cfg.use_backend_voices()
    cfg.qwen3_preload_presets = False
    cfg.voice_library_dir = args.voice_library_dir
    if args.config_db and Path(args.config_db).exists():
        settings = PersistentConfig(args.config_db)
        saved = settings.get("qwen3_accent_donor")
        if saved is not None:
            cfg.qwen3_accent_donor = saved
        settings.close()
    needing = [e for e in entries if clone_conditioning(cfg.qwen3_clone_mode, e.ref_text) == "bridge"]
    if not needing:
        return
    from sts.tts.qwen3 import Qwen3TTSSynthesizer

    print(f"\nMaking English references (base voice: {cfg.qwen3_accent_donor or 'none'})...")
    tts = Qwen3TTSSynthesizer(cfg)
    for entry in needing:
        tts._english_reference(str(entry.wav_path))
        print(f"  {entry.name}: done")


def _clock(seconds: float) -> str:
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes}:{seconds:02d}"


if __name__ == "__main__":
    sys.exit(main())
