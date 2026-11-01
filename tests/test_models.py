"""Models load the version they loaded last time; newer ones are found, not fetched."""

from datetime import datetime
from types import SimpleNamespace

from sts import models
from sts.config import PipelineConfig

OLD, NEW = "a" * 40, "b" * 40


class Settings(dict):
    def set(self, key, value):
        self[key] = value


def cache_with_ref(tmp_path, monkeypatch, repo, branch, commit):
    monkeypatch.setattr("sts.downloads.hub_cache_dir", lambda: tmp_path)
    ref = tmp_path / f"models--{repo.replace('/', '--')}" / "refs" / branch
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_text(commit)


def test_the_loaded_commit_is_kept_and_loaded_next_time(tmp_path, monkeypatch):
    cfg = PipelineConfig()
    cfg.tts.backend = "xtts"  # (no hub voice model to track)
    cache_with_ref(tmp_path, monkeypatch, cfg.translation.model_name, "main", OLD)
    settings = Settings()
    models.keep_loaded_revisions(cfg, settings)
    assert models.saved_revisions(settings)[cfg.translation.model_name] == OLD
    later = PipelineConfig()
    models.apply_saved_revisions(later, settings)
    assert later.translation.model_revision == OLD


def test_a_newer_commit_on_the_branch_is_an_update(tmp_path, monkeypatch):
    cfg = PipelineConfig()
    cfg.tts.backend = "xtts"
    cfg.translation.model_revision = OLD
    monkeypatch.setattr("sts.downloads.hub_cache_dir", lambda: tmp_path)

    class Api:
        def model_info(self, repo, revision, timeout):
            sha = NEW if repo == cfg.translation.model_name else None
            return SimpleNamespace(sha=sha, last_modified=datetime(2026, 10, 31))

    monkeypatch.setattr("huggingface_hub.HfApi", Api)
    updates = models.check_updates(cfg)
    assert [(u["key"], u["stage"], u["revision"], u["date"]) for u in updates] == [
        ("translation", "translate", NEW, "2026-10-31")
    ]


def test_offline_finds_nothing(monkeypatch):
    cfg = PipelineConfig()
    cfg.tts.backend = "xtts"

    class Api:
        def model_info(self, *args, **kwargs):
            raise OSError("offline")

    monkeypatch.setattr("huggingface_hub.HfApi", Api)
    assert models.check_updates(cfg) == []


def test_only_commits_are_kept():
    settings = Settings(model_revisions='{"x/y": "main", "x/z": "%s"}' % OLD)
    assert models.saved_revisions(settings) == {"x/z": OLD}
