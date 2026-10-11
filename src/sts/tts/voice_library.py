"""Persistent library of per-speaker voices, so a voice captured/trained in
one session can be picked from a dropdown and reused in a later one.

Two kinds of entry:

- `"zeroshot"` - a reference clip (+ optional transcript) for zero-shot
  cloning, captured live via the "Train voice" button. Recomputing the
  clone prompt from these raw inputs takes seconds.
- `"lora"` - a real trained LoRA adapter (see `training/qwen3_lora/`),
  imported from elsewhere (a rented GPU used to train it) via
  `save_lora_adapter`. Stores the adapter files, the internal speaker name
  baked into them at training time, and the LoRA scale to apply at
  inference (Qwen3-TTS LoRA fine-tunes are reportedly over-steered at the
  library default of 1.0 - 0.3 is a more typical usable value; see
  `training/qwen3_lora/README.md`).

Either way, entries hold raw inputs/files rather than a backend's in-memory
prompt/adapter object, so they stay readable and reusable across process
restarts and TTS backend/library version changes - a pickled internal
object would not be.
"""

import json
import re
import shutil
import time
from dataclasses import dataclass, replace
from pathlib import Path


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return slug or "speaker"


@dataclass(frozen=True)
class VoiceEntry:
    speaker_id: str
    name: str
    created_at: float
    kind: str = "zeroshot"  # "zeroshot" or "lora"
    # zeroshot fields
    ref_text: str | None = None
    wav_path: Path | None = None
    # lora fields
    adapter_dir: Path | None = None
    speaker_name: str | None = None
    lora_scale: float = 0.3
    # Shipped with the app (read-only, can't be deleted); a voice of the same id saved by the user wins.
    builtin: bool = False
    # Made by the caption page's Design dialog (invented, not cloned from a speaker).
    designed: bool = False


# Voices that ship with the package (src/sts/bundled_voices/<id>/{reference.wav,meta.json}).
BUNDLED_VOICES_DIR = Path(__file__).resolve().parent.parent / "bundled_voices"


class VoiceLibrary:
    def __init__(self, root: Path, bundled_dir: Path | None = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.bundled_dir = Path(bundled_dir) if bundled_dir else None

    def _bundled(self) -> dict[str, VoiceEntry]:
        """The shipped voices by id, read-only. Loaded from their metadata like any other entry."""
        entries: dict[str, VoiceEntry] = {}
        if self.bundled_dir and self.bundled_dir.is_dir():
            for entry_dir in sorted(p for p in self.bundled_dir.iterdir() if p.is_dir()):
                entry = self._load_entry(entry_dir)
                if entry is not None:
                    entries[entry.speaker_id] = replace(entry, builtin=True)
        return entries

    def save(self, name: str, wav_path: str | Path, ref_text: str | None = None, designed: bool = False) -> VoiceEntry:
        """Registers a zero-shot clone entry from a captured reference clip
        (or, `designed`, one the Design dialog invented)."""
        speaker_id = _slugify(name)
        entry_dir = self.root / speaker_id
        entry_dir.mkdir(parents=True, exist_ok=True)

        dest_wav = entry_dir / "reference.wav"
        dest_wav.write_bytes(Path(wav_path).read_bytes())

        created_at = time.time()
        meta = {"kind": "zeroshot", "name": name, "ref_text": ref_text, "created_at": created_at}
        if designed:
            meta["designed"] = True
        (entry_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

        return VoiceEntry(
            speaker_id=speaker_id, name=name, created_at=created_at, kind="zeroshot",
            ref_text=ref_text, wav_path=dest_wav, designed=designed,
        )

    def save_lora_adapter(
        self,
        name: str,
        adapter_source_dir: str | Path,
        speaker_name: str,
        lora_scale: float = 0.3,
    ) -> VoiceEntry:
        """Imports a trained LoRA adapter directory (e.g. downloaded from a
        rented GPU used with `training/qwen3_lora/`). `speaker_name` is the
        internal name it was trained under (`--speaker_name` at training
        time) - required to select it via `generate_custom_voice`."""
        speaker_id = _slugify(name)
        entry_dir = self.root / speaker_id
        if entry_dir.exists():
            shutil.rmtree(entry_dir)
        entry_dir.mkdir(parents=True)

        dest_adapter = entry_dir / "adapter"
        shutil.copytree(adapter_source_dir, dest_adapter)

        created_at = time.time()
        meta = {
            "kind": "lora",
            "name": name,
            "speaker_name": speaker_name,
            "lora_scale": lora_scale,
            "created_at": created_at,
        }
        (entry_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

        return VoiceEntry(
            speaker_id=speaker_id, name=name, created_at=created_at, kind="lora",
            adapter_dir=dest_adapter, speaker_name=speaker_name, lora_scale=lora_scale,
        )

    def _load_entry(self, entry_dir: Path) -> VoiceEntry | None:
        meta_path = entry_dir / "meta.json"
        if not meta_path.exists():
            return None
        meta = json.loads(meta_path.read_text())
        kind = meta.get("kind", "zeroshot")

        if kind == "lora":
            adapter_dir = entry_dir / "adapter"
            if not adapter_dir.exists():
                return None
            return VoiceEntry(
                speaker_id=entry_dir.name,
                name=meta["name"],
                created_at=meta.get("created_at", 0.0),
                kind="lora",
                adapter_dir=adapter_dir,
                speaker_name=meta.get("speaker_name"),
                lora_scale=meta.get("lora_scale", 0.3),
            )

        wav_path = entry_dir / "reference.wav"
        if not wav_path.exists():
            return None
        return VoiceEntry(
            speaker_id=entry_dir.name,
            name=meta["name"],
            created_at=meta.get("created_at", 0.0),
            kind="zeroshot",
            ref_text=meta.get("ref_text"),
            wav_path=wav_path,
            designed=bool(meta.get("designed")),
        )

    def list(self) -> list[VoiceEntry]:
        """Shipped voices first, then the user's by age. A saved voice with a shipped voice's id replaces it."""
        mine = []
        for entry_dir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            entry = self._load_entry(entry_dir)
            if entry is not None:
                mine.append(entry)
        taken = {e.speaker_id for e in mine}
        shipped = [e for e in self._bundled().values() if e.speaker_id not in taken]
        return shipped + sorted(mine, key=lambda e: e.created_at)

    def _entry_dir(self, speaker_id: str) -> Path | None:
        """The entry's directory, only if `speaker_id` is a plain name of a
        direct child of the library root. Ids arrive from the browser, so
        anything like "../x" or an absolute path must never resolve."""
        if not speaker_id or Path(speaker_id).name != speaker_id or speaker_id in (".", ".."):
            return None
        entry_dir = self.root / speaker_id
        if entry_dir.is_symlink() or entry_dir.resolve().parent != self.root.resolve():
            return None
        return entry_dir if entry_dir.is_dir() else None

    def get(self, speaker_id: str) -> VoiceEntry | None:
        entry_dir = self._entry_dir(speaker_id)
        if entry_dir:
            entry = self._load_entry(entry_dir)
            if entry is not None:
                return entry
        return self._bundled().get(speaker_id)  # a plain dict lookup: the id never becomes a path

    def delete(self, speaker_id: str) -> bool:
        """Removes a voice from disk. Returns False if there was no such voice."""
        entry_dir = self._entry_dir(speaker_id)
        if entry_dir is None:
            return False
        shutil.rmtree(entry_dir)
        return True
