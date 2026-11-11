"""Deterministic candidate selection, draft compilation, and mechanical validation."""

from __future__ import annotations

import csv
import io
import math
import json
from decimal import Decimal
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import soundfile as sf

from .audio import read_region
from .config import Config
from .languages import language, profile
from .models import Candidate, ExportEntry, Landmarks, ProvenanceKind, SourceAsset
from .oto import encode_oto, parse_oto
from .storage import atomic_bytes, json_hash, read_jsonl, sha256_file, write_json, write_jsonl


def select(project_dir: Path, config: Config) -> dict[str, int]:
    candidates = read_jsonl(project_dir / "candidates.jsonl", Candidate)
    grouped: dict[str, list[Candidate]] = {}
    for candidate in candidates:
        if candidate.source_language not in config.target_languages:
            continue  # A cross-language mapper must explicitly approve other-language material.
        grouped.setdefault(candidate.alias, []).append(candidate)
    chosen: dict[str, str] = {}
    ranks: dict[str, list[str]] = {}
    selection_path = project_dir / "selection.json"
    previous = json.loads(selection_path.read_text(encoding="utf-8")) if selection_path.exists() else {}
    manual = previous.get("manual_chosen", {})
    stale_manual: list[str] = []
    for alias, items in grouped.items():
        usable = [c for c in items if c.landmarks is not None and c.review_status != "rejected" and (not c.quality_flags or (c.review_status == "accepted" and set(c.quality_flags) <= {"COARTICULATION_UNCERTAIN"}))]
        usable.sort(key=lambda c: (c.review_status != "accepted", -(c.quality_score or 0), c.source_id, c.span.start))
        if usable:
            valid_ids = {item.candidate_id for item in usable}
            chosen[alias] = manual.get(alias) if manual.get(alias) in valid_ids else usable[0].candidate_id
        if alias in manual and manual[alias] not in {item.candidate_id for item in usable}:
            stale_manual.append(alias)
        ranks[alias] = [c.candidate_id for c in usable[:config.selection.top_k]]
    write_json(selection_path, {"schema_version": 1, "chosen": chosen, "manual_chosen": manual, "stale_manual_aliases": sorted(stale_manual), "top_candidates": ranks})
    return {"aliases_with_candidates": len(grouped), "selected": len(chosen)}


def _note_to_hz(note: str) -> float:
    import re
    match = re.fullmatch(r"([A-G])([#b]?)([0-8])", note)
    if not match:
        raise ValueError(f"invalid target note: {note}")
    semitones = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    midi = 12 * (int(match.group(3)) + 1) + semitones[match.group(1)] + {"#": 1, "b": -1, "": 0}[match.group(2)]
    return 440 * 2 ** ((midi - 69) / 12)


def _target_pitch(candidates: list[Candidate], note: str) -> float:
    if note != "auto":
        return _note_to_hz(note)
    eligible = [(c.metrics.f0_median_hz, c.metrics.stable_vowel_ms) for c in candidates if c.metrics.f0_median_hz and c.metrics.stable_vowel_ms]
    if not eligible:
        raise ValueError("no measured F0; set --target-note or use --pitch-mode preserve")
    eligible.sort()
    total = sum(weight for _, weight in eligible)
    cumulative = 0.0
    median = eligible[-1][0]
    for hz, weight in eligible:
        cumulative += weight
        if cumulative >= total / 2:
            median = hz
            break
    midi = round(69 + 12 * math.log2(median / 440))
    return 440 * 2 ** ((midi - 69) / 12)


def _flatten(audio: np.ndarray, target_hz: float, config: Config) -> tuple[np.ndarray, dict]:
    try:
        import pyworld
    except ImportError as exc:
        raise RuntimeError("PyWORLD missing; install with uv sync --extra pitch") from exc
    fs = 44100
    waveform = audio.astype("float64")
    original_f0, times = pyworld.harvest(waveform, fs, f0_floor=config.features.f0_min_hz, f0_ceil=config.features.f0_max_hz, frame_period=config.features.hop_ms)
    original_f0 = pyworld.stonemask(waveform, original_f0, times, fs)
    if not np.any(original_f0 > 0):
        raise ValueError("no voiced frames for pitch flattening")
    spectrum = pyworld.cheaptrick(waveform, original_f0, times, fs)
    aperiodicity = pyworld.d4c(waveform, original_f0, times, fs)
    modified = np.where(original_f0 > 0, target_hz, 0.0)
    rendered = pyworld.synthesize(modified, spectrum, aperiodicity, fs, frame_period=config.features.hop_ms)
    if len(rendered) < len(audio):
        rendered = np.pad(rendered, (0, len(audio) - len(rendered)))
    else:
        rendered = rendered[:len(audio)]
    median_source = float(np.median(original_f0[original_f0 > 0]))
    return rendered.astype("float32"), {"target_hz": target_hz, "original_f0_median_hz": median_source, "shift_semitones": 12 * math.log2(target_hz / median_source), "resynthesis_count": 1}


def _remeasure_f0(audio: np.ndarray, marks: Landmarks, target_hz: float, config: Config) -> dict:
    import pyworld
    f0, times = pyworld.harvest(audio.astype("float64"), 44100, f0_floor=config.features.f0_min_hz, f0_ceil=config.features.f0_max_hz, frame_period=config.features.hop_ms)
    f0 = pyworld.stonemask(audio.astype("float64"), f0, times, 44100)
    stable = (times * 44100 >= marks.fixed_end) & (times * 44100 < marks.end) & (f0 > 0)
    if not np.any(stable):
        return {"measured_f0_median_hz": None, "f0_error_cents": None, "f0_warning": "UNMEASURABLE_F0"}
    median = float(np.median(f0[stable]))
    error = 1200 * math.log2(median / target_hz)
    return {"measured_f0_median_hz": median, "f0_error_cents": error, "f0_warning": "F0_OUT_OF_TOLERANCE" if abs(error) > 20 else None}


def _level(audio: np.ndarray, marks: Landmarks, config: Config) -> tuple[np.ndarray, dict]:
    region = audio[marks.fixed_end:marks.end]
    if region.size == 0:
        # A manual fixed-end at the used endpoint has no stretchable tail.
        # Preserve its level rather than inventing vowel evidence for it.
        return audio.copy(), {"mode": "preserve: no sustain tail", "gain_db": 0}
    rms = float(np.sqrt(np.mean(region.astype("float64") ** 2)))
    if rms < 1e-6:
        raise ValueError("sustain region too quiet")
    target = 10 ** (config.level.target_vowel_rms_dbfs / 20)
    gain_db = float(np.clip(20 * math.log10(target / rms), -config.level.max_gain_db, config.level.max_gain_db))
    gain = 10 ** (gain_db / 20)
    peak_limit = 10 ** (config.level.max_peak_dbfs / 20)
    peak = float(np.max(np.abs(audio)))
    if peak * gain > peak_limit:
        gain = peak_limit / peak
        gain_db = 20 * math.log10(gain)
    output = (audio * gain).astype("float32")
    rng = np.random.default_rng(12345)
    dither = (rng.random(len(output)) - rng.random(len(output))) / 32768
    output = np.clip(output + dither, -peak_limit, peak_limit).astype("float32")
    return output, {"gain_db": gain_db, "sample_peak_dbfs": 20 * math.log10(max(float(np.max(np.abs(output))), 1e-8))}


def _csv_bytes(rows: list[dict[str, str]], fieldnames: list[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8-sig")


def _bank_digest(bank: Path) -> str:
    return json_hash({path.relative_to(bank).as_posix(): sha256_file(path) for path in sorted(bank.rglob("*")) if path.is_file()})


def build(project_dir: Path, config: Config, pitch_mode: str, target_note: str, release_mode: str,
          *, include_unreviewed_splices: bool = False, singer_name: str | None = None) -> dict:
    if pitch_mode not in {"preserve", "flatten"} or release_mode not in {"draft", "strict"}:
        raise ValueError("invalid pitch or release mode")
    if include_unreviewed_splices and release_mode != "draft":
        raise ValueError("unreviewed splices are only available in draft builds")
    display_name = singer_name or project_dir.name
    if not display_name.strip() or any(char in display_name for char in "\r\n=\x00"):
        raise ValueError("invalid singer name")
    selection_path = project_dir / "selection.json"
    if not selection_path.exists():
        raise ValueError("run select before build")
    chosen = dict(json.loads(selection_path.read_text(encoding="utf-8"))["chosen"])
    candidates = {c.candidate_id: c for c in read_jsonl(project_dir / "candidates.jsonl", Candidate)}
    if include_unreviewed_splices:
        pending_splices: dict[str, Candidate] = {}
        for candidate in candidates.values():
            if (candidate.alias in chosen or candidate.provenance_kind != ProvenanceKind.cv_spliced
                    or candidate.review_status != "unreviewed" or candidate.landmarks is None
                    or candidate.source_language not in config.target_languages
                    or set(candidate.quality_flags) != {"REPAIR_REVIEW_REQUIRED", "COARTICULATION_UNCERTAIN"}):
                continue
            current = pending_splices.get(candidate.alias)
            if current is None or (candidate.quality_score or 0, candidate.candidate_id) > (current.quality_score or 0, current.candidate_id):
                pending_splices[candidate.alias] = candidate
        chosen.update({alias: candidate.candidate_id for alias, candidate in pending_splices.items()})
    assets = {s.source_id: s for s in read_jsonl(project_dir / "sources" / "manifest.jsonl", SourceAsset)}
    selected = [candidates[identifier] for identifier in chosen.values()]
    target_hz = _target_pitch(selected, target_note) if pitch_mode == "flatten" and selected else None
    build_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    root = project_dir / "exports" / build_id
    bank = root / "voicebank"
    reports = root / "reports"
    bank.mkdir(parents=True)
    reports.mkdir()
    prof = profile(config.export_profiles[0])
    inventory = language(prof.language_id).inventory
    required = prof.required_units(inventory, "core")
    entries: list[ExportEntry] = []
    oto_lines: list[str] = []
    failed: list[dict[str, str]] = []
    transformations: list[dict] = []
    for candidate in selected:
        if candidate.landmarks is None or candidate.source_language != prof.language_id:
            failed.append({"alias": candidate.alias, "reason": "UNSUPPORTED_SOURCE_OR_TIMING"})
            continue
        source = assets[candidate.source_id]
        marks = candidate.landmarks
        try:
            derived_path = candidate.phonetic_context.get("derived_audio_path")
            if derived_path:
                derived_file = (project_dir / derived_path).resolve(strict=True)
                if not derived_file.is_relative_to(project_dir.resolve()):
                    raise ValueError("derived audio path escapes project")
                derived, derived_rate = sf.read(derived_file, dtype="float32")
                if derived_rate != 44100 or derived.ndim != 1:
                    raise ValueError("invalid derived audio format")
                raw = derived[marks.start:marks.end]
            else:
                raw = read_region(project_dir, source, marks.start, marks.end)
            local = Landmarks(start=0, overlap=marks.overlap - marks.start, preutter=marks.preutter - marks.start, fixed_end=marks.fixed_end - marks.start, end=marks.end - marks.start)
            local.validate_oto(len(raw))
            processed, pitch_info = _flatten(raw, target_hz, config) if target_hz else (raw, {"mode": "preserve", "resynthesis_count": 0})
            if target_hz:
                pitch_info.update(_remeasure_f0(processed, local, target_hz, config))
                if abs(pitch_info["shift_semitones"]) > config.pitch.warn_shift_semitones:
                    pitch_info["shift_warning"] = "LARGE_PITCH_SHIFT"
            processed, level_info = _level(processed, local, config)
            unit = next((u for u in required if u.unit_key == candidate.unit_key), None)
            if unit is None:
                raise ValueError("candidate unit is absent from the target inventory")
            filename = f"{prof.filename_stem(unit)}.wav"
            if (bank / filename).exists():
                raise ValueError(f"romanized filename collision: {filename}")
            sf.write(bank / filename, processed, 44100, subtype="PCM_16")
            for alias in prof.aliases(unit):
                oto_lines.append(prof.compile_timing(local, len(processed), filename, alias))
            mapping = ({"derived_audio_path": derived_path, "source_range": candidate.phonetic_context.get("source_range"), "donor_range": candidate.phonetic_context.get("donor_range"), "join": candidate.phonetic_context.get("join"), "parent_candidate_ids": candidate.parent_candidate_ids, "padding_samples": 0} if derived_path else {"source_id": source.source_id, "source_start": marks.start, "source_end": marks.end, "padding_samples": 0})
            entries.append(ExportEntry(filename=filename, target_language=prof.language_id, target_alias=candidate.alias, source_candidate_ids=[candidate.candidate_id, *candidate.parent_candidate_ids], landmarks=local, output_samples=len(processed), source_to_output=mapping, provenance_kind=candidate.provenance_kind))
            transformations.append({"filename": filename, "candidate_id": candidate.candidate_id, "pitch": pitch_info, "level": level_info})
        except (ValueError, RuntimeError) as exc:
            failed.append({"alias": candidate.alias, "reason": str(exc)})
    atomic_bytes(bank / "oto.ini", encode_oto(oto_lines, prof.oto_encoding))
    atomic_bytes(bank / "character.txt", f"name={display_name}\r\n".encode(prof.oto_encoding, errors="strict"))
    write_jsonl(reports / "provenance.jsonl", entries)
    present = {entry.target_alias for entry in entries}
    missing = [{"unit_key": unit.unit_key, "alias": prof.aliases(unit)[0], "reason": "NO_APPROVED_SUSTAIN_CANDIDATE"} for unit in required if prof.aliases(unit)[0] not in present]
    atomic_bytes(reports / "missing_units.csv", _csv_bytes(missing, ["unit_key", "alias", "reason"]))
    review = [{"alias": entry.target_alias, "candidate_id": entry.source_candidate_ids[0], "reason": "UNREVIEWED"} for entry in entries if candidates[entry.source_candidate_ids[0]].review_status == "unreviewed"]
    atomic_bytes(reports / "needs_review.csv", _csv_bytes(review, ["alias", "candidate_id", "reason"]))
    original_count = sum(e.provenance_kind == "original" for e in entries)
    extended_count = sum(e.provenance_kind == "vowel_extended" for e in entries)
    splice_count = sum(e.provenance_kind == "cv_spliced" for e in entries)
    confirmed_count = sum(candidates[e.source_candidate_ids[0]].review_status == "accepted" for e in entries)
    denominator = max(len(required), 1)
    write_json(reports / "coverage.json", {"required_units": len(required), "exported_units": len(entries), "original": original_count, "vowel_extended": extended_count, "cv_spliced": splice_count, "human_confirmed": confirmed_count, "missing": len(missing), "total_coverage_pct": 100 * len(entries) / denominator, "original_coverage_pct": 100 * original_count / denominator, "vowel_extended_coverage_pct": 100 * extended_count / denominator, "cv_spliced_coverage_pct": 100 * splice_count / denominator, "human_confirmed_pct": 100 * confirmed_count / max(len(entries), 1), "excluded_candidates": len(candidates) - len(entries), "alias_equivalent_lines": len(oto_lines) - len(entries)})
    write_json(reports / "quality.json", {"status": "mechanical_validation_pending", "failed_entries": failed, "pitch_mode": pitch_mode, "transformations": transformations})
    write_json(reports / "build_manifest.json", {"build_id": build_id, "profile": prof.profile_id, "profile_version": prof.version, "filename_style": "ja-romaji-v1", "release_mode": "draft", "render_status": "pending", "pitch_mode": pitch_mode, "target_note_requested": target_note if pitch_mode == "flatten" else None, "target_hz": target_hz, "include_unreviewed_splices": include_unreviewed_splices, "singer_name": display_name, "source_project": str(project_dir.resolve()), "bank_sha256": _bank_digest(bank), "selected_candidates_sha256": json_hash([candidate.model_dump(mode="json") for candidate in selected])})
    validation = validate_export(root)
    render_gate = validate_render_report(root, present)
    strict_ready = not missing and not failed and not review and validation["errors"] == 0 and render_gate["passed"]
    write_json(reports / "quality.json", {"status": "mechanically_valid" if validation["errors"] == 0 else "invalid", "failed_entries": failed, "pitch_mode": pitch_mode, "transformations": transformations, "validation": validation, "render_gate": render_gate})
    zip_path = root / "voicebank.zip"
    with ZipFile(zip_path, mode="x", compression=ZIP_DEFLATED) as archive:
        for path in sorted(bank.iterdir()):
            if path.is_file():
                archive.write(path, arcname=f"voicebank/{path.name}")
    return {"build_id": build_id, "path": str(root.resolve()), "zip_path": str(zip_path.resolve()), "pitch_mode": pitch_mode, "target_note": target_note if pitch_mode == "flatten" else None, "singer_name": display_name, "include_unreviewed_splices": include_unreviewed_splices, "exported": len(entries), "cv_spliced": splice_count, "missing": len(missing), "failed": len(failed), "needs_review": len(review), "strict_ready": strict_ready, "validation_errors": validation["errors"]}


def validate_render_report(root: Path, exported_aliases: set[str]) -> dict:
    path = root / "reports" / "render_report.json"
    if not path.exists():
        return {"passed": False, "reason": "RENDER_REPORT_MISSING"}
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("status") != "passed" or report.get("import_status") != "passed" or report.get("human_listening") != "passed":
            return {"passed": False, "reason": "RENDER_OR_LISTENING_NOT_PASSED"}
        if report.get("engine") != "OpenUtau" or not report.get("engine_version") or not report.get("renderer"):
            return {"passed": False, "reason": "ENGINE_IDENTITY_MISSING"}
        manifest = json.loads((root / "reports" / "build_manifest.json").read_text(encoding="utf-8"))
        if report.get("bank_sha256") != manifest.get("bank_sha256") or _bank_digest(root / "voicebank") != manifest.get("bank_sha256"):
            return {"passed": False, "reason": "BANK_CHANGED_SINCE_RENDER"}
        if not exported_aliases.issubset(set(report.get("tested_aliases", []))):
            return {"passed": False, "reason": "ALIASES_NOT_ALL_TESTED"}
        files = report.get("rendered_wav", [])
        if not files:
            return {"passed": False, "reason": "RENDERED_WAV_MISSING"}
        for relative in files:
            wav = (root / relative).resolve()
            if not wav.is_relative_to(root.resolve()) or not wav.is_file() or sf.info(wav).frames == 0:
                return {"passed": False, "reason": "RENDERED_WAV_INVALID"}
    except (ValueError, TypeError, OSError, UnicodeError) as exc:
        return {"passed": False, "reason": f"RENDER_REPORT_INVALID: {exc}"}
    return {"passed": True, "reason": None}


def finalize_export(root: Path) -> dict:
    reports = root / "reports"
    manifest = json.loads((reports / "build_manifest.json").read_text(encoding="utf-8"))
    project_dir = Path(manifest["source_project"])
    entries = read_jsonl(reports / "provenance.jsonl", ExportEntry)
    all_candidates = {candidate.candidate_id: candidate for candidate in read_jsonl(project_dir / "candidates.jsonl", Candidate)}
    selected_ids = [entry.source_candidate_ids[0] for entry in entries if entry.source_candidate_ids]
    missing_candidate_ids = sorted(set(selected_ids) - set(all_candidates))
    selected = [all_candidates[candidate_id] for candidate_id in selected_ids if candidate_id in all_candidates]
    current_hash = json_hash([candidate.model_dump(mode="json") for candidate in selected]) if not missing_candidate_ids else None
    coverage = json.loads((reports / "coverage.json").read_text(encoding="utf-8"))
    quality = json.loads((reports / "quality.json").read_text(encoding="utf-8"))
    validation = validate_export(root)
    aliases = {entry.target_alias for entry in entries}
    gate = validate_render_report(root, aliases)
    reasons = []
    if coverage["missing"]:
        reasons.append("MANDATORY_UNITS_MISSING")
    if quality["failed_entries"] or validation["errors"]:
        reasons.append("MECHANICAL_VALIDATION_FAILED")
    if any(transform["pitch"].get("f0_warning") for transform in quality["transformations"]):
        reasons.append("PITCH_VALIDATION_FAILED")
    if any(candidate.review_status != "accepted" for candidate in selected):
        reasons.append("SELECTED_CANDIDATES_NEED_REVIEW")
    if missing_candidate_ids or len(selected_ids) != len(entries) or current_hash != manifest.get("selected_candidates_sha256"):
        reasons.append("CANDIDATES_CHANGED_SINCE_BUILD")
    if not gate["passed"]:
        reasons.append(gate["reason"])
    result = {"strict_ready": not reasons, "reasons": reasons, "missing_candidate_ids": missing_candidate_ids,
              "validation": validation, "render_gate": gate, "build_id": manifest["build_id"]}
    write_json(reports / "release_check.json", result)
    if result["strict_ready"]:
        manifest["release_mode"] = "strict"
        manifest["render_status"] = "passed"
        write_json(reports / "build_manifest.json", manifest)
    return result


def validate_export(root: Path) -> dict[str, int | list[str]]:
    bank = root / "voicebank"
    manifest = json.loads((root / "reports" / "build_manifest.json").read_text(encoding="utf-8"))
    prof = profile(manifest["profile"])
    lines = (bank / "oto.ini").read_bytes().decode(prof.oto_encoding, errors="strict").splitlines()
    errors: list[str] = []
    checked = 0
    for line in lines:
        try:
            parsed = parse_oto(line)
            wav = bank / parsed.filename
            if not wav.resolve().is_relative_to(bank.resolve()):
                raise ValueError("oto WAV path escapes voicebank")
            info = sf.info(wav)
            if (info.samplerate, info.channels, info.subtype) != (44100, 1, "PCM_16"):
                raise ValueError("WAV must be 44.1k mono PCM16")
            duration_ms = info.frames * 1000 / 44100
            absolute = parsed.absolute_ms(Decimal(str(duration_ms)))
            if not (0 <= absolute["start"] < absolute["end"] <= Decimal(str(duration_ms)) + Decimal("0.002") and all(absolute["start"] <= absolute[field] <= absolute["end"] for field in ("overlap", "preutter", "fixed_end"))):
                raise ValueError("oto landmarks out of range")
            audio, _ = sf.read(wav, dtype="float32")
            if audio.size == 0 or not bool(np.isfinite(audio).all()):
                raise ValueError("empty or non-finite WAV")
            peak = float(np.max(np.abs(audio)))
            if peak > 10 ** (-0.98 / 20):
                raise ValueError("sample peak exceeds -1 dBFS tolerance")
            checked += 1
        except (OSError, ValueError, UnicodeError) as exc:
            errors.append(f"{line}: {exc}")
    return {"checked_oto_lines": checked, "errors": len(errors), "details": errors}
