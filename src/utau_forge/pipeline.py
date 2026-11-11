"""Restartable command pipeline. Each stage records only committed output signatures."""

from __future__ import annotations

import json
from pathlib import Path

from .align import align
from .analyze import analyze_project
from .build import build, select
from .models import SourceAsset, Utterance
from .project import load
from .segment import segment_source
from .storage import json_hash, read_jsonl, sha256_file, write_json
from .transcribe import transcribe


_STAGES = ("segment", "transcribe", "align", "analyze", "select", "build")


def run_pipeline(project_dir: Path, until: str = "build") -> dict:
    if until not in _STAGES:
        raise ValueError(f"--until must be one of {', '.join(_STAGES)}")
    _, config = load(project_dir)
    assets = read_jsonl(project_dir / "sources" / "manifest.jsonl", SourceAsset)
    if not assets:
        raise ValueError("import an audio source before run")
    state_path = project_dir / "runs" / "pipeline.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"schema_version": 1, "stages": {}}
    result: dict = {"until": until, "stages": {}}
    for asset in assets:
        segment_source(project_dir, asset, config)
    result["stages"]["segment"] = {"utterances": len(read_jsonl(project_dir / "utterances.jsonl", Utterance))}
    if until == "segment":
        return result
    result["stages"]["transcribe"] = {"transcribed": transcribe(project_dir, config)}
    if until == "transcribe":
        return result
    alignment = align(project_dir, config)
    result["stages"]["align"] = alignment
    if until == "align":
        return result
    phones = project_dir / "phones.jsonl"
    analyze_key = json_hash({"phones": sha256_file(phones), "features": config.features.model_dump(), "sustain": config.sustain.model_dump(), "profile": config.export_profiles, "algorithm": "analyze-2"})
    candidates = project_dir / "candidates.jsonl"
    if state["stages"].get("analyze", {}).get("key") == analyze_key and candidates.exists():
        result["stages"]["analyze"] = {"cached": True}
    else:
        result["stages"]["analyze"] = analyze_project(project_dir, config)
        state["stages"]["analyze"] = {"key": analyze_key}
        write_json(state_path, state)
    if until == "analyze":
        return result
    select_key = json_hash({"candidates": sha256_file(candidates), "config": config.selection.model_dump(), "targets": config.target_languages, "algorithm": "select-2"})
    selection = project_dir / "selection.json"
    if state["stages"].get("select", {}).get("key") == select_key and selection.exists():
        result["stages"]["select"] = {"cached": True}
    else:
        result["stages"]["select"] = select(project_dir, config)
        state["stages"]["select"] = {"key": select_key}
        write_json(state_path, state)
    if until == "select":
        return result
    build_key = json_hash({"selection": sha256_file(selection), "candidates": sha256_file(candidates), "pitch": config.pitch.model_dump(), "level": config.level.model_dump(), "export": config.export.model_dump(), "profiles": config.export_profiles, "algorithm": "build-4-romanized-filenames"})
    old = state["stages"].get("build", {})
    if old.get("key") == build_key and old.get("path") and Path(old["path"]).is_dir():
        result["stages"]["build"] = {"cached": True, "path": old["path"], "build_id": old["build_id"]}
    else:
        made = build(project_dir, config, config.pitch.mode, config.pitch.target_note, "draft")
        result["stages"]["build"] = made
        state["stages"]["build"] = {"key": build_key, "path": made["path"], "build_id": made["build_id"]}
        write_json(state_path, state)
    return result
