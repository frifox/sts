"""Which version of each model sts runs, and newer ones on the hub.

A model loaded by name alone ("main") is downloaded again whenever its
publisher uploads a new version - unannounced, at the next start (Drama 3's
re-upload was ~27 GB). So once a model has loaded, the commit it loaded is
kept (sts.db, MODELS_KEY) and loaded from then on, from the cache, and a
newer version only comes when the operator asks for it: the caption page
checks the hub now and then (`check_updates`) and shows an Update button in
the dropdown of the stage the model belongs to (Caption, Translate, Voice).
"""

import json
import re
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

MODELS_KEY = "model_revisions"  # {repo: commit} in the persistent settings
COMMIT = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class TrackedModel:
    key: str  # "asr", "translation", "qwen3", "qwen3-presets", "drama3"
    stage: str  # the caption page dropdown it belongs in: caption | translate | voice
    label: str
    repo: str
    branch: str  # what's followed for newer versions
    section: str  # the config section (cfg.asr / .translation / .tts) ...
    field: str  # ... and its field holding the revision to load

    def revision(self, cfg) -> str:
        return getattr(getattr(cfg, self.section), self.field)

    def with_revision(self, section_cfg, revision: str):
        """A copy of the config section (cfg.asr, ...) loading `revision`."""
        return replace(section_cfg, **{self.field: revision})


def tracked_models(cfg, in_use_only: bool = True) -> list[TrackedModel]:
    """The hub models sts loads (in use: for the current voice model only).
    Not tracked: XTTS (not from the hub) and Qwen3-TTS on the PyTorch engine
    (qwen_tts loads part of it from the latest version anyway)."""
    from sts.asr.gigaam_asr import MODEL_NAME as ASR_MODEL

    models = [
        TrackedModel("asr", "caption", "GigaAM-v3 (speech recognition)", ASR_MODEL, "e2e_rnnt",
                     "asr", "model_revision"),
        TrackedModel("translation", "translate", cfg.translation.model_name.split("/")[-1],
                     cfg.translation.model_name, "main", "translation", "model_revision"),
    ]
    tts = cfg.tts
    if tts.backend in ("qwen3", "drama3") or not in_use_only:
        if not in_use_only or tts.backend == "qwen3":
            from sts.tts.qwen3_engine import resolve_engine

            if resolve_engine(tts.qwen3_engine) == "mlx":
                from sts.tts.qwen3_mlx import mlx_model_name

                models += [
                    TrackedModel("qwen3", "voice", "Qwen3-TTS", mlx_model_name(tts.qwen3_model_name), "main",
                                 "tts", "qwen3_model_revision"),
                    TrackedModel("qwen3-presets", "voice", "Qwen3-TTS preset voices",
                                 mlx_model_name(tts.qwen3_custom_model_name), "main",
                                 "tts", "qwen3_custom_model_revision"),
                ]
        if not in_use_only or tts.backend == "drama3":
            models.append(TrackedModel("drama3", "voice", "Drama 3", tts.drama3_model_name, "main",
                                       "tts", "drama3_model_revision"))
    return models


def saved_revisions(persistent_config) -> dict[str, str]:
    if persistent_config is None:
        return {}
    try:
        saved = json.loads(persistent_config.get(MODELS_KEY) or "{}")
    except (ValueError, TypeError):
        return {}
    return {repo: rev for repo, rev in saved.items() if isinstance(rev, str) and COMMIT.match(rev)}


def save_revision(persistent_config, repo: str, revision: str) -> None:
    if persistent_config is None:
        return
    saved = saved_revisions(persistent_config)
    saved[repo] = revision
    persistent_config.set(MODELS_KEY, json.dumps(saved, sort_keys=True))


def apply_saved_revisions(cfg, persistent_config) -> None:
    """Every tracked model loads the version kept for it, if one is."""
    saved = saved_revisions(persistent_config)
    for model in tracked_models(cfg, in_use_only=False):
        if model.repo in saved:
            setattr(getattr(cfg, model.section), model.field, saved[model.repo])


def cached_commit(repo: str, revision: str) -> str | None:
    """The commit `revision` (a branch, or a commit already) is in the local hub cache."""
    if COMMIT.match(revision):
        return revision
    from sts.downloads import hub_cache_dir

    ref = hub_cache_dir() / f"models--{repo.replace('/', '--')}" / "refs" / revision
    try:
        commit = ref.read_text().strip()
    except OSError:
        return None
    return commit if COMMIT.match(commit) else None


def keep_loaded_revisions(cfg, persistent_config) -> None:
    """After loading: the commit each model in use loaded is kept, so it's what
    loads next time (and not whatever its branch has moved to by then)."""
    saved = saved_revisions(persistent_config)
    for model in tracked_models(cfg):
        if model.repo in saved:
            continue
        commit = cached_commit(model.repo, model.revision(cfg))
        if commit:
            save_revision(persistent_config, model.repo, commit)
            setattr(getattr(cfg, model.section), model.field, commit)


def check_updates(cfg) -> list[dict]:
    """The models in use with a newer version on the hub: {key, stage, label,
    repo, revision, date}. Network errors: nothing found."""
    from huggingface_hub import HfApi

    api = HfApi()
    updates = []
    for model in tracked_models(cfg):
        current = cached_commit(model.repo, model.revision(cfg))
        try:
            info = api.model_info(model.repo, revision=model.branch, timeout=15)
        except Exception:  # noqa: BLE001 - offline, rate-limited, ...: try again next time
            continue
        if info.sha and current and info.sha != current:
            when = info.last_modified
            updates.append({
                "key": model.key, "stage": model.stage, "label": model.label, "repo": model.repo,
                "revision": info.sha,
                "date": when.strftime("%Y-%m-%d") if isinstance(when, datetime) else None,
            })
    return updates
