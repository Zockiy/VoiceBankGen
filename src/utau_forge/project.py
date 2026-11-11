"""Project initialization and metadata access."""

from __future__ import annotations

from pathlib import Path

from .config import Config, read_config, write_config
from .languages import language, profile
from .models import Project
from .storage import read_json, write_json


def initialize(directory: Path, language_id: str = "ja", profile_id: str = "ja-cv", speaker_id: str = "speaker_001") -> Project:
    lang = language(language_id)
    prof = profile(profile_id)
    if prof.language_id != lang.language_id:
        raise ValueError("profile target language does not match selected language")
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"project directory is not empty: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    project = Project(speaker_id=speaker_id, language=language_id, source_languages=[language_id], target_languages=[language_id], export_profiles=[profile_id])
    config = Config(language=language_id, source_languages=[language_id], target_languages=[language_id], export_profiles=[profile_id],inventory=lang.inventory["version"])
    write_json(directory / "project.json", project)
    write_config(directory / "config.yaml", config)
    write_json(directory / "models.lock.json", {"schema_version": 1, "models": [], "status": "unverified"})
    for relative in ("sources", "audio/canonical", "audio/analysis", "alignment", "derived", "runs", "exports"):
        (directory / relative).mkdir(parents=True, exist_ok=True)
    return project


def load(directory: Path) -> tuple[Project, Config]:
    return read_json(directory / "project.json", Project), read_config(directory / "config.yaml")
