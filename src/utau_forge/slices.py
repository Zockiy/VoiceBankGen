"""Unmodified-pitch source slice export with explicit provenance."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4
import csv
import math

import soundfile as sf

from .audio import read_region
from .custom_units import default_stem, validate_unit
from .languages import language, profile
from .models import Candidate, SourceAsset
from .project import load
from .storage import read_jsonl, write_json


def slice_sort_key(candidate: Candidate) -> tuple:
    """Prefer measured stable vowels, then use full slice length as fallback."""
    stable = candidate.metrics.stable_vowel_ms
    measured = stable is not None and math.isfinite(stable) and stable > 0 and "NO_SUSTAIN_REGION" not in candidate.quality_flags
    duration = (candidate.span.end - candidate.span.start) / 44.1
    return (not measured, -(stable if measured else duration), -duration,
            candidate.source_id, candidate.span.start, candidate.candidate_id)


def export_slices(project_dir: Path, all_candidates: bool = False, top_k: int = 3, padding_ms: int = 0, *, custom_units=(), selected_candidate_ids=(), emit=lambda **event: None) -> dict:
    if top_k < 1 or padding_ms < 0:
        raise ValueError("top_k must be positive and padding_ms nonnegative")
    candidates = read_jsonl(project_dir / "candidates.jsonl", Candidate)
    config = load(project_dir)[1]
    export_profile = profile(config.export_profiles[0])
    # Source slices include every original unit in the shared project, even
    # when the active bank is CV. CV and VV can share a phone key (e.g. n+a),
    # so retain the role when resolving a filename.
    naming_profiles={code:profile(code+"-cvvc") for code in {candidate.target_language or candidate.source_language for candidate in candidates}}
    units = {(code,unit.unit_type if unit.unit_type in {"VC", "VV"} else "CV", unit.unit_key,alias): unit
             for code,naming_profile in naming_profiles.items() for unit in naming_profile.required_units(language(code).inventory, "core") for alias in naming_profile.aliases(unit)}
    custom_stems = {unit["alias"]: validate_unit(unit["label"], unit["filename_stem"])[1] for unit in custom_units}
    sources = {item.source_id: item for item in read_jsonl(project_dir / "sources" / "manifest.jsonl", SourceAsset)}
    selected_ids = set(selected_candidate_ids)
    groups: dict[tuple, list[Candidate]] = {}
    for candidate in candidates:
        if candidate.provenance_kind.value == "original" or (not all_candidates and candidate.candidate_id in selected_ids):
            groups.setdefault((candidate.target_language or candidate.source_language,candidate.alias), []).append(candidate)
    selected: list[Candidate] = []
    for items in groups.values():
        items.sort(key=lambda item: (item.candidate_id not in selected_ids, *slice_sort_key(item)))
        selected.extend(items if all_candidates else items[:top_k])
    pad = round(padding_ms * 44.1)
    planned = []
    for candidate in selected:
        source = sources[candidate.source_id]
        derived = candidate.phonetic_context.get("derived_audio_path")
        audio_path = project_dir / (derived or source.canonical_path)
        if not audio_path.resolve().is_relative_to(project_dir.resolve()):
            raise ValueError("切片音频越出工程目录")
        total = sf.info(audio_path).frames if derived else source.canonical_samples
        if derived and (sf.info(audio_path).samplerate != 44100 or sf.info(audio_path).channels != 1):
            raise ValueError("派生切片必须为 44.1kHz 单声道")
        start = max(0, candidate.span.start - pad)
        end = min(total, candidate.span.end + pad)
        if not (0 <= candidate.span.start < candidate.span.end <= total):
            raise ValueError(f"候选 {candidate.candidate_id} 的切片范围超出原音频")
        if candidate.unit_role == "breath":
            stem = "br"
        elif candidate.unit_role == "custom":
            stem = custom_stems.get(candidate.alias) or default_stem(candidate.alias)
        else:
            role = candidate.unit_role if candidate.unit_role in {"VC", "VV"} else "CV"
            code = candidate.target_language or candidate.source_language
            unit = units.get((code,role, candidate.unit_key,candidate.alias))
            if unit is None:
                raise ValueError(f"候选 {candidate.candidate_id} 没有可用的切片命名定义：{role} / {candidate.unit_key}")
            stem = naming_profiles[code].filename_stem(unit)
        planned.append((candidate, source, start, end, stem, audio_path, derived))
    root = project_dir / "exports" / ("slices_" + uuid4().hex[:12])
    root.mkdir(parents=True)
    manifest = []
    alias_counts: dict[str, int] = {}
    for index, (candidate, source, start, end, stem, audio_path, derived) in enumerate(planned):
        emit(stage="导出原始切片", current=index, total=len(planned), reason=stem)
        alias_counts[stem] = alias_counts.get(stem, 0) + 1
        filename = f"{stem}_{alias_counts[stem]:03d}.wav"
        audio = sf.read(audio_path, start=start, stop=end, dtype="float32")[0] if derived else read_region(project_dir, source, start, end)
        sf.write(root / filename, audio, 44100, subtype="PCM_16")
        manifest.append({"filename": filename, "candidate_id": candidate.candidate_id, "alias": candidate.alias, "source_language":candidate.source_language,"unit_key": candidate.unit_key, "unit_role": candidate.unit_role, "source_id": source.source_id, "source_start": start, "source_end": end, "unit_start_in_file": candidate.span.start - start, "unit_end_in_file": candidate.span.end - start, "padding_before_samples": candidate.span.start - start, "padding_after_samples": end - candidate.span.end, "boundary_source": "refined" if candidate.landmarks else "aligned", "pitch_mode": "preserve", "provenance_kind": candidate.provenance_kind.value})
        manifest[-1].update(selected=candidate.candidate_id in selected_ids, audio_path=derived or source.canonical_path,
                            stable_vowel_ms=candidate.metrics.stable_vowel_ms,
                            ranking_basis="stable_vowel_ms" if not slice_sort_key(candidate)[0] else "slice_duration_ms")
    write_json(root / "manifest.json", {"schema_version": 1, "sample_rate": 44100, "slices": manifest})
    if manifest:
        with (root/"sources.csv").open("w",encoding="utf-8-sig",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(manifest[0]))
            writer.writeheader()
            writer.writerows(manifest)
    emit(stage="导出原始切片", current=len(planned), total=len(planned))
    return {"path": str(root.resolve()), "slices": len(manifest)}
