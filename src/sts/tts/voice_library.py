"""Persistent library of per-speaker voice references, so a voice captured
in one session can be picked from a dropdown and reused in a later one.

Each entry is a directory under `root/<speaker_id>/` holding the raw
inputs a cloning call needs (`reference.wav` + `meta.json` with the
speaker's name and, for backends that use one, a transcript of the
reference clip) rather than a backend's in-memory prompt/embedding object.
Recomputing that object from the raw inputs takes seconds, and doing it
this way means an entry stays readable and reusable across process
restarts and TTS backend/library version changes - a pickled internal
object would not be.
"""

import json
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return slug or "speaker"


@dataclass(frozen=True)
class VoiceEntry:
    speaker_id: str
    name: str
    ref_text: str | None
    wav_path: Path
    created_at: float


class VoiceLibrary:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, name: str, wav_path: str | Path, ref_text: str | None = None) -> VoiceEntry:
        speaker_id = _slugify(name)
        entry_dir = self.root / speaker_id
        entry_dir.mkdir(parents=True, exist_ok=True)

        dest_wav = entry_dir / "reference.wav"
        dest_wav.write_bytes(Path(wav_path).read_bytes())

        created_at = time.time()
        meta = {"name": name, "ref_text": ref_text, "created_at": created_at}
        (entry_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

        return VoiceEntry(speaker_id, name, ref_text, dest_wav, created_at)

    def list(self) -> list[VoiceEntry]:
        entries = []
        for entry_dir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            meta_path = entry_dir / "meta.json"
            wav_path = entry_dir / "reference.wav"
            if not (meta_path.exists() and wav_path.exists()):
                continue
            meta = json.loads(meta_path.read_text())
            entries.append(
                VoiceEntry(
                    speaker_id=entry_dir.name,
                    name=meta["name"],
                    ref_text=meta.get("ref_text"),
                    wav_path=wav_path,
                    created_at=meta.get("created_at", 0.0),
                )
            )
        return sorted(entries, key=lambda e: e.created_at)

    def get(self, speaker_id: str) -> VoiceEntry | None:
        entry_dir = self.root / speaker_id
        meta_path = entry_dir / "meta.json"
        wav_path = entry_dir / "reference.wav"
        if not (meta_path.exists() and wav_path.exists()):
            return None
        meta = json.loads(meta_path.read_text())
        return VoiceEntry(
            speaker_id=speaker_id,
            name=meta["name"],
            ref_text=meta.get("ref_text"),
            wav_path=wav_path,
            created_at=meta.get("created_at", 0.0),
        )

    def delete(self, speaker_id: str) -> None:
        entry_dir = self.root / speaker_id
        if entry_dir.exists():
            shutil.rmtree(entry_dir)
