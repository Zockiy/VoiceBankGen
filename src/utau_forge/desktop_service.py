"""Local previews, immutable processed audio and installable bank compilation."""

from __future__ import annotations

import json
import os
import shutil
from decimal import Decimal
from pathlib import Path
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import soundfile as sf
import yaml

from .audio import read_region
from .build import _flatten, _level, _target_pitch
from .config import Config, read_config
from .desktop_store import ProjectStore
from .languages import language, profile
from .models import Candidate, Landmarks, Metrics, SampleSpan, SourceAsset, Utterance
from .oto import compile_oto, encode_oto, parse_oto
from .storage import atomic_bytes, json_hash, sha256_file, write_json
from .bank_aliases import export_aliases, filename_stem, split_variant, units_by_alias, DEFAULT_ALIAS_FORMAT


def safe_path(root: Path, relative: str) -> Path:
    path = Path(relative)
    if not relative or path.is_absolute() or path.drive or ".." in path.parts:
        raise ValueError(f"不安全的相对路径：{relative}")
    result = (root / path).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError(f"文件越过音源目录：{relative}")
    return result


def load_manifest(root: Path) -> tuple[dict, dict]:
    marker = json.loads((root / "uforge.project-link.json").read_text(encoding="utf-8"))
    if marker.get("format") != "utau-forge-project-link" or marker.get("schema_version") != 1 or marker.get("app_schema_version", 0) > 3:
        raise ValueError("音源标记损坏或版本不受支持；请关联原工程或导入为恢复工程")
    manifest_path = safe_path(root, marker["manifest_file"])
    if sha256_file(manifest_path) != marker["manifest_sha256"]:
        raise ValueError("音源清单已改变；请先解决外部冲突")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("project_id") != marker["project_id"] or manifest.get("bank_id") != marker["bank_id"]:
        raise ValueError("音源标记与清单身份不一致")
    for relative in manifest["files"]:
        safe_path(root, relative)
    return marker, manifest


def candidate_audio(store: ProjectStore, candidate: Candidate) -> tuple[np.ndarray, int, int]:
    derived = candidate.phonetic_context.get("derived_audio_path")
    if derived:
        audio, sample_rate = sf.read(safe_path(store.directory, derived), dtype="float32")
        if sample_rate != 44100 or audio.ndim != 1:
            raise ValueError("派生候选必须为 44.1kHz 单声道")
        return audio, 0, len(audio)
    asset = next(item for item in store.records("assets", SourceAsset) if item.source_id == candidate.source_id)
    marks = candidate.automatic_landmarks or candidate.landmarks
    start = max(0, min(candidate.span.start, marks.start if marks else candidate.span.start) - 4410)
    end = min(asset.canonical_samples, max(candidate.span.end, marks.end if marks else candidate.span.end) + 4410)
    return read_region(store.directory, asset, start, end), start, end


def cached_preview(store: ProjectStore, candidate_id: str) -> dict | None:
    """Resolve an existing preview without reading the source audio or DSP."""
    directory = store.directory / "analyses" / "previews"
    path = directory / f"{candidate_id}.wav"
    if not path.is_file():
        return None
    metadata = directory / f"{candidate_id}.json"
    if metadata.is_file():
        result = json.loads(metadata.read_text(encoding="utf-8"))
        # The project may have been copied or moved since this cache was made.
        return {**result, "path": str(path), "spectrum": str(directory / f"{candidate_id}_spectrum.npz"), "candidate_id": candidate_id}
    candidate = store.candidate(candidate_id)
    marks = candidate.automatic_landmarks or candidate.landmarks
    start = 0 if candidate.phonetic_context.get("derived_audio_path") else max(0, min(candidate.span.start, marks.start if marks else candidate.span.start) - 4410)
    result = {"path": str(path), "start": start, "end": start + sf.info(path).frames,
              "candidate_id": candidate_id, "spectrum": str(directory / f"{candidate_id}_spectrum.npz")}
    write_json(metadata, result)
    return result


def preview_file(store: ProjectStore, candidate_id: str, spectrum: bool = False) -> dict:
    result = cached_preview(store, candidate_id)
    directory = store.directory / "analyses" / "previews"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{candidate_id}.wav"
    if result is None:
        candidate = store.candidate(candidate_id)
        audio, start, end = candidate_audio(store, candidate)
        sf.write(path, audio, 44100, subtype="PCM_16")
        result = {"path": str(path), "spectrum": str(directory / f"{candidate_id}_spectrum.npz"),
                  "start": start, "end": end, "candidate_id": candidate_id}
        write_json(directory / f"{candidate_id}.json", result)
    spectrum_path = Path(result["spectrum"])
    if spectrum and not spectrum_path.exists():
        from scipy.signal import spectrogram
        audio, _ = sf.read(path, dtype="float32")
        segment = min(1024, len(audio))
        frequencies,times,power=spectrogram(audio,fs=44100,nperseg=segment,noverlap=min(768,segment-1))
        mask=frequencies<=10000
        np.savez_compressed(spectrum_path,frequencies=frequencies[mask],times=times*1000,power=10*np.log10(np.maximum(power[mask],1e-12)))
    return result


def project_config(store: ProjectStore) -> Config:
    return read_config(store.directory / "config.yaml") if (store.directory / "config.yaml").exists() else Config()


def preview_interval(candidate: Candidate, preview: dict) -> tuple[int, int]:
    """Use oto when available, otherwise audition the complete cached slice."""
    if candidate.landmarks:
        return candidate.landmarks.start - preview["start"], candidate.landmarks.end - preview["start"]
    return 0, preview["end"] - preview["start"]


def force_automatic_timing(store: ProjectStore, candidate: Candidate, preview: dict) -> Landmarks:
    """Propose one candidate's automatic oto without the sustain acceptance gate.

    Returns a proposal only. The UI persists it as an undoable timing override;
    acoustic evidence, review state and formal selection remain separate.
    """
    if candidate.automatic_landmarks is not None:
        return candidate.automatic_landmarks.model_copy(deep=True)
    path = Path(preview["path"]).resolve()
    if not path.is_relative_to(store.directory.resolve()):
        raise ValueError("分析音频必须位于当前工程内")
    start = max(preview["start"], candidate.span.start)
    end = min(preview["end"], candidate.span.end)
    if end <= start:
        raise ValueError("当前切片没有可标注的音频")
    if candidate.unit_role in {"sustain_vowel", "nasal"} and candidate.phonetic_context.get("vowel_span"):
        from .analyze import _features, _longest_sustain
        audio, sample_rate = sf.read(path, dtype="float32")
        if sample_rate != 44100 or audio.ndim != 1:
            raise ValueError("自动标注需要 44.1kHz 单声道切片")
        config = project_config(store)
        span = SampleSpan(start=preview["start"], end=preview["end"])
        utterance = Utterance(source_id=candidate.source_id, speaker_id=candidate.speaker_id,
                              span=span, vad_span=span, corpus_offset_samples=0,
                              language_id=candidate.source_language)
        marks, _ = _longest_sustain(candidate, _features(audio, sample_rate, config), utterance, config, force=True)
        return marks
    boundary = max(start, min(end - 1, candidate.phonetic_context.get("transition_boundary", start)))
    return Landmarks(start=start, overlap=start, preutter=boundary,
                     fixed_end=min(end - 1, boundary + 441), end=end)


def force_missing_sustain_timings(store: ProjectStore, bank_id: str, emit=lambda **event: None) -> dict:
    """Propose one usable slice per missing alias, without publishing edits."""
    selected = store.selections(bank_id)
    targets = dict(store.targets(bank_id, include_variants=False))
    groups = {}
    allowed = {"NO_SUSTAIN_REGION", "COARTICULATION_UNCERTAIN", "ZH_TIMING_REVIEW"}
    for candidate in store.candidates(bank_id=bank_id):
        if (candidate.alias in targets and candidate.alias not in selected and candidate.landmarks is None
                and candidate.review_status != "rejected" and "NO_SUSTAIN_REGION" in candidate.quality_flags
                and not set(candidate.quality_flags) - allowed):
            groups.setdefault(candidate.alias, []).append(candidate)
    proposals, failures = [], []
    for index, (alias, candidates) in enumerate(groups.items()):
        emit(stage="批量强制标注缺音 oto", current=index, total=len(groups), reason=alias, device="cpu")
        candidates.sort(key=lambda c: (-(c.span.end-c.span.start), c.candidate_id))
        errors = []
        for candidate in candidates:
            try:
                marks = force_automatic_timing(store, candidate, preview_file(store, candidate.candidate_id))
                proposals.append({"alias": alias, "candidate_id": candidate.candidate_id, "marks": marks.model_dump()})
                break
            except Exception as error:
                errors.append(str(error))
        else:
            failures.append({"alias": alias, "errors": errors})
    emit(stage="批量强制标注缺音 oto", current=len(groups), total=len(groups), device="cpu")
    return {"proposals": proposals, "eligible": len(groups), "failures": failures}


def audition_file(store: ProjectStore, source: Path, start: int, end: int) -> Path:
    """Cache only the oto usage interval; playback cannot leak the context tail."""
    source = source.resolve()
    if not source.is_relative_to(store.directory.resolve()):
        raise ValueError("试听音频必须位于当前工程内")
    info = sf.info(source)
    if not 0 <= start < end <= info.frames:
        raise ValueError("试听区间必须位于 WAV 内，且使用起点小于使用终点")
    key = json_hash({"algorithm": "oto-audition-1", "source": sha256_file(source), "start": start, "end": end})
    directory = store.directory / "analyses" / "auditions"
    path = directory / (key + ".wav")
    if not path.is_file():
        directory.mkdir(parents=True, exist_ok=True)
        audio, sample_rate = sf.read(source, start=start, stop=end, dtype="int16")
        sf.write(path, audio, sample_rate, subtype="PCM_16")
    return path


def render_cache_key(candidate: Candidate, source_hash: str, target_hz: float | None, config: Config) -> str:
    automatic = candidate.automatic_landmarks or candidate.landmarks
    return json_hash({"algorithm": "desktop-render-2", "candidate": candidate.candidate_id, "source": source_hash,
                      "span": candidate.span.model_dump(), "automatic": automatic.model_dump(), "target_hz": target_hz,
                      "level": config.level.model_dump(), "features": config.features.model_dump()})


def render_candidate(store: ProjectStore, candidate: Candidate, bank: dict, target_hz: float | None) -> dict:
    if candidate.landmarks is None:
        raise ValueError("此候选尚无可用标线")
    export_profile = profile(bank["profile_id"])
    if (candidate.target_language or candidate.source_language)!=export_profile.language_id:
        raise ValueError("候选目标语言与当前音源版本不一致")
    unit = units_by_alias(bank["profile_id"]).get(candidate.alias)
    breath = candidate.unit_role == "breath"
    custom_units = store.custom_units(bank["id"])
    custom = any(item["alias"] == candidate.alias for item in custom_units)
    if unit is None and not breath and not custom:
        raise ValueError("候选不属于当前音源格式")
    if breath or custom:
        target_hz = None
    automatic = candidate.automatic_landmarks or candidate.landmarks
    sources = {item.source_id: item for item in store.records("assets", SourceAsset)}
    source_hash = sources[candidate.source_id].sha256
    derived = candidate.phonetic_context.get("derived_audio_path")
    if derived:
        source_hash = sha256_file(safe_path(store.directory, derived))
    config = project_config(store)
    cache_key = render_cache_key(candidate, source_hash, target_hz, config)
    cache_dir = store.directory / "processed"
    cache_dir.mkdir(exist_ok=True)
    cache_record = cache_dir / f"{cache_key}.json"
    if cache_record.exists():
        record = json.loads(cache_record.read_text(encoding="utf-8"))
        if sha256_file(safe_path(store.directory, record["path"])) != record["sha256"]:
            raise ValueError("处理缓存被外部修改；请更换预设或恢复缓存")
    else:
        audio, start, end = candidate_audio(store, candidate)
        local_automatic = Landmarks(**{field: getattr(automatic, field) - start for field in Landmarks.model_fields})
        local_automatic.validate_oto(len(audio))
        pitch_info = {"mode": "preserve", "resynthesis_count": 0}
        adopted=(candidate.phonetic_context.get("adopted_processed_preset")==store.preset_key(bank["id"])) or (
            target_hz is None and bool(derived) and bool(candidate.phonetic_context.get("imported_filename")))
        if target_hz is not None and not adopted:
            audio, pitch_info = _flatten(audio, target_hz, config)
        if breath:
            level_info = {"mode": "preserve breath", "gain_db": 0}
        elif custom or candidate.phonetic_context.get("manual_import"):
            level_info = {"mode": "preserve imported audio", "gain_db": 0}
        elif adopted:
            level_info={"mode":"adopted external WAV", "gain_db":0}
        else:
            level_marks=local_automatic
            if candidate.unit_role=="VC":
                # Level a VC from its preceding vowel, never from a single
                # terminal consonant sample or an artificial sustain region.
                level_marks=Landmarks(start=local_automatic.start,overlap=local_automatic.start,preutter=local_automatic.start,fixed_end=local_automatic.start,end=max(local_automatic.start+1,local_automatic.preutter))
            audio, level_info = _level(audio, level_marks, config)
        stem = filename_stem(bank["profile_id"], candidate.alias, custom_units)
        pending = cache_dir / f"{stem}_{cache_key[:16]}.wav"
        if not pending.exists():
            if adopted:
                shutil.copy2(safe_path(store.directory,derived),pending)
            else:
                sf.write(pending, audio, 44100, subtype="PCM_16")
        digest = sha256_file(pending)
        record = {"path": pending.relative_to(store.directory).as_posix(), "filename": f"{stem}_{candidate.candidate_id[:8]}_{digest[:12]}.wav",
                  "sha256": digest, "window_start": start, "window_end": end, "samples": len(audio), "pitch": pitch_info, "level": level_info}
        write_json(cache_record, record)
    local = Landmarks(**{field: getattr(candidate.landmarks, field) - record["window_start"] for field in Landmarks.model_fields})
    local.validate_oto(record["samples"])
    return {**record, "marks": local.model_dump(), "aliases": export_aliases(bank["profile_id"], candidate.alias, bank["preset"].get("alias_format", DEFAULT_ALIAS_FORMAT), custom_units)}


def bank_encoding(bank):
    encoding=profile(bank["profile_id"]).oto_encoding
    try: bank["name"].encode(encoding)
    except UnicodeEncodeError: encoding="utf-8"
    return encoding


def write_bank_metadata(store: ProjectStore, bank: dict, root: Path, entries: list[dict], missing: list[dict], *, wav_hashes: dict | None = None, processing_key: str | None = None) -> dict:
    manifest = {"schema_version": 1, "project_id": store.project_id, "bank_id": bank["id"], "profile_id": bank["profile_id"],
                "revision": store.revision, "entries": entries, "files": {}, "processing_key": processing_key,
                "alias_format": bank["preset"].get("alias_format", DEFAULT_ALIAS_FORMAT), "include_breaths": bool(bank["preset"].get("include_breaths")),
                "cv_base_bank_id":bank["preset"].get("cv_base_bank_id"),
                "custom_units": store.custom_units(bank["id"]), "oto_encoding":bank_encoding(bank)}
    names = {entry["filename"] for entry in entries} | {"oto.ini", "character.txt", "character.yaml", "missing-units.json", "README.txt", "presamp.ini"}
    for name in sorted(names):
        if (root / name).is_file():
            manifest["files"][name] = (wav_hashes or {}).get(name) or sha256_file(root / name)
    write_json(root / "uforge.manifest.json", manifest)
    marker = {"format": "utau-forge-project-link", "schema_version": 1, "project_id": store.project_id, "bank_id": bank["id"],
              "installation_id": None, "display_name": bank["name"], "profile_id": bank["profile_id"],
              "source_languages": store.meta("source_languages"), "target_languages": store.meta("target_languages"),
              "desired_revision": store.revision, "applied_revision": store.revision,
              "manifest_file": "uforge.manifest.json", "manifest_sha256": sha256_file(root / "uforge.manifest.json"), "app_schema_version": 3}
    write_json(root / "uforge.project-link.json", marker)
    return manifest


def compile_bank(store: ProjectStore, bank_id: str, generation: int, emit=lambda **event: None) -> dict:
    emit(stage="读取选音与处理缓存",current=0,total=0)
    bank = store.bank(bank_id)
    encoding=bank_encoding(bank)
    if bank["generation"] != generation:
        return {"stale": True}
    active_targets = dict(store.targets(bank_id))
    selections = {alias: entry for alias, entry in store.selections(bank_id).items() if alias in active_targets}
    captured_revision=store.revision
    candidates = {item.candidate_id: item for item in store.candidates(bank_id=bank_id, selected_only=True)}
    manual_timing_ids = store.manual_timing_ids(bank_id)
    selected = list({item["candidate_id"]: candidates[item["candidate_id"]] for item in selections.values()}.values())
    custom_units = store.custom_units(bank_id)
    custom_labels = {unit["alias"]: unit["label"] for unit in custom_units}
    def display_alias(alias):
        base, suffix = split_variant(alias)
        return custom_labels.get(base, base) + suffix
    voiced = [candidate for candidate in selected if candidate.unit_role not in {"breath", "custom", "VC", "VV"}]
    if bank["preset"].get("pitch_mode") == "flatten" and not bank["preset"].get("resolved_hz") and bank["preset"].get("target_note", "auto") == "auto":
        # Manual imports have no ASR/acoustic analysis. Measure their pitch only
        # if an automatic standard pitch is explicitly requested at build time.
        from .analyze import _features
        measured = []
        for candidate in voiced:
            if candidate.phonetic_context.get("manual_import") and not candidate.metrics.f0_median_hz:
                f0 = _features(candidate_audio(store, candidate)[0],44100,project_config(store))["f0"]
                hz = float(np.median(f0[f0 > 0])) if np.any(f0 > 0) else None
                # These temporary pitch weights are not persisted or presented
                # as measured sustain regions for manual audio.
                duration=(candidate.span.end-candidate.span.start)/44.1 if hz else None
                candidate = candidate.model_copy(update={"metrics": candidate.metrics.model_copy(update={"f0_median_hz": hz, "stable_vowel_ms": duration})})
            measured.append(candidate)
        voiced = measured
    target_hz = (bank["preset"].get("resolved_hz") or _target_pitch(voiced, bank["preset"].get("target_note", "auto"))) if bank["preset"].get("pitch_mode", "preserve") == "flatten" and voiced else None
    config = project_config(store)
    processing_key = json_hash({"algorithm": "desktop-render-2", "target_hz": target_hz,
                                "level": config.level.model_dump(), "features": config.features.model_dump(), "profile": bank["profile_id"]})
    previous_root = safe_path(store.directory, bank["build_path"]) if bank["build_path"] else None
    previous = load_manifest(previous_root)[1] if previous_root else {"entries": [], "files": {}}
    compatible = previous.get("processing_key") == processing_key
    previous_entries = {}
    for entry in previous["entries"]:
        previous_entries.setdefault(entry["candidate_id"], []).append(entry)
    if not compatible:
        if previous_root and not previous.get("processing_key"):
            # Old oto-only builds have no processing.json. Match each immutable
            # cache record directly, including source, DSP configuration and F0.
            sources = {item.source_id: item.sha256 for item in store.records("assets", SourceAsset)}
            for candidate in selected:
                retained = previous_entries.get(candidate.candidate_id)
                if not retained:
                    continue
                derived = candidate.phonetic_context.get("derived_audio_path")
                source_hash = sha256_file(safe_path(store.directory, derived)) if derived else sources[candidate.source_id]
                adopted = candidate.phonetic_context.get("adopted_processed_preset") == store.preset_key(bank_id)
                recovered = target_hz is None and candidate.phonetic_context.get("imported_filename")
                if derived and (adopted or recovered) and all(source_hash == previous["files"].get(entry["filename"]) for entry in retained):
                    continue
                record_path = store.directory / "processed" / (render_cache_key(candidate, source_hash, None if candidate.unit_role in {"breath","custom"} else target_hz, config) + ".json")
                record = json.loads(record_path.read_text(encoding="utf-8")) if record_path.exists() else {}
                if any(record.get("filename") != entry.get("base_filename", entry["filename"]) or record.get("sha256") != previous["files"].get(entry["filename"]) for entry in retained):
                    previous_entries.pop(candidate.candidate_id, None)
        else:
            previous_entries = {}
    complete_selections = set(selections) == {entry.get("selection_alias", entry["alias"]) for entry in previous["entries"]}
    if previous_root and complete_selections and previous.get("revision") == captured_revision and compatible and previous.get("alias_format") == bank["preset"].get("alias_format", DEFAULT_ALIAS_FORMAT) and previous.get("include_breaths") == bool(bank["preset"].get("include_breaths")):
        return {"path": str(previous_root), "rendered": 0, "updated_timings": 0, "unchanged": len(selected), "manifest": previous}
    build_root = store.directory / "builds" / bank_id / (str(store.revision) + "_" + uuid4().hex[:8])
    root = build_root / "voicebank"
    root.mkdir(parents=True)
    export_profile = profile(bank["profile_id"])
    entries: list[dict] = []
    failures: list[dict] = []
    transformations: list[dict] = []
    wav_hashes = {}
    rendered_count = 0
    timing_count = 0
    work_count = sum(candidate.candidate_id not in previous_entries for candidate in selected)
    emit(stage="处理有变化的音", current=0, total=work_count)
    for index, candidate in enumerate(selected):
        try:
            if candidate.review_status=="rejected":
                raise ValueError("候选已排除")
            # Manual oto replaces the missing timing decision, not the acoustic
            # evidence or the user's separate listening acceptance. Legacy
            # accepted manual candidates may predate the override table.
            manual_timing = candidate.landmarks is not None and (
                candidate.candidate_id in manual_timing_ids
                or (candidate.automatic_landmarks is None and candidate.review_status == "accepted")
            )
            timing_flags={"NO_SUSTAIN_REGION","DEVOICED_VOWEL"}
            allowed_flags={"COARTICULATION_UNCERTAIN","REPAIR_REVIEW_REQUIRED","CROSS_LANGUAGE_APPROXIMATE","ZH_TIMING_REVIEW"} | (timing_flags if manual_timing else set())
            blocked_flags = [flag for flag in candidate.quality_flags if flag not in allowed_flags]
            if blocked_flags:
                from .issue_labels import issue_text
                raise ValueError("候选有待处理问题："+issue_text(blocked_flags))
            if bank["preset"].get("release_mode")=="strict" and timing_flags.intersection(candidate.quality_flags) and candidate.review_status!="accepted":
                raise ValueError("严格导出需要先试听并点击「试听接受候选」确认手动 oto")
            if bank["preset"].get("release_mode")=="strict" and (candidate.provenance_kind.value!="original" or "ZH_TIMING_REVIEW" in candidate.quality_flags) and candidate.review_status!="accepted":
                raise ValueError("严格导出需要先试听接受补音候选")
            retained = previous_entries.get(candidate.candidate_id)
            if retained:
                old = retained[0]
                rendered = {**old, "filename": old.get("base_filename", old["filename"]),
                            "sha256": previous["files"][old["filename"]]}
                source_wav = safe_path(previous_root, old["filename"])
                marks = Landmarks(**{field: getattr(candidate.landmarks, field) - old["window_start"] for field in Landmarks.model_fields})
                marks.validate_oto(old["samples"])
                timing_count += int(old["marks"] != marks.model_dump())
            else:
                rendered = render_candidate(store, candidate, bank, target_hz)
                rendered_count += 1
                emit(stage="处理有变化的音", current=rendered_count, total=work_count)
                source_wav = safe_path(store.directory, rendered["path"])
                marks = Landmarks.model_validate(rendered["marks"])
                transformations.append({"candidate_id": candidate.candidate_id, "pitch": rendered["pitch"], "level": rendered["level"]})
            # Rebuild the alias roster independently of immutable audio. Adding
            # alternatives or changing notation never reprocesses existing WAVs.
            for selection_alias, selection in selections.items():
                if selection["candidate_id"] != candidate.candidate_id:
                    continue
                if split_variant(selection_alias)[0] != candidate.alias:
                    raise ValueError("备选音与候选发音不一致")
                suffix = split_variant(selection_alias)[1]
                output_filename = (filename_stem(bank["profile_id"], selection_alias, custom_units) + "_" + candidate.candidate_id[:8] + "_" + rendered["sha256"][:12] + ".wav") if suffix else rendered["filename"]
                target = root / output_filename
                if not target.exists():
                    try:
                        os.link(source_wav, target)
                    except OSError:
                        shutil.copy2(source_wav, target)
                wav_hashes[output_filename] = rendered["sha256"]
                for alias in export_aliases(bank["profile_id"], selection_alias, bank["preset"].get("alias_format", DEFAULT_ALIAS_FORMAT), custom_units):
                    line = export_profile.compile_timing(marks, rendered["samples"], output_filename, alias)
                    entries.append({"entry_id": selection["entry_id"] + ":" + alias, "alias": alias,
                                    "selection_alias": selection_alias, "base_filename": rendered["filename"],
                                    "candidate_id": candidate.candidate_id, "filename": output_filename,
                                    "line": line, "marks": marks.model_dump(), "samples": rendered["samples"], "window_start": rendered["window_start"],
                                    "unit_type": store.unit_type(candidate), "provenance_kind": candidate.provenance_kind.value, "parents": candidate.parent_candidate_ids,
                                    "review_status": candidate.review_status,"source_language":candidate.source_language,
                                    "target_language":candidate.target_language or candidate.source_language,
                                    "cross_language":candidate.phonetic_context.get("cross_language"),
                                    "timing_evidence":candidate.phonetic_context.get("timing_evidence")})
        except (ValueError, RuntimeError, OSError) as error:
            failures.append({"alias": display_alias(candidate.alias), "reason": str(error)})
    present = {entry["selection_alias"] for entry in entries}
    missing = [{"alias": display_alias(alias), "unit_type": unit_type} for alias, unit_type in store.targets(bank_id) if alias not in present]
    unknown = []
    if previous_root:
        owned = {(entry["alias"], entry["filename"]) for entry in previous["entries"]}
        for line in (previous_root / "oto.ini").read_bytes().decode(previous.get("oto_encoding",export_profile.oto_encoding)).splitlines():
            try:
                parsed = parse_oto(line)
                if (parsed.alias, parsed.filename) not in owned:
                    unknown.append(line)
            except ValueError:
                unknown.append(line)
    atomic_bytes(root / "oto.ini", encode_oto([entry["line"] for entry in entries] + unknown, encoding))
    atomic_bytes(root / "character.txt", ("name=" + bank["name"] + "\r\n").encode(encoding, errors="strict"))
    yaml_metadata = {"singer_type": "utau", "text_file_encoding": "utf-8" if encoding=="utf-8" else "shift_jis"}
    if bank["profile_id"] == "ja-cvvc" and bank["preset"].get("alias_format", DEFAULT_ALIAS_FORMAT) == "hiragana":
        yaml_metadata["default_phonemizer"] = "OpenUtau.Plugin.Builtin.JapaneseCVVCPhonemizer"
    if bank["profile_id"].startswith("zh-"):
        from .languages.zh import presamp_text
        atomic_bytes(root/"presamp.ini",presamp_text(language("zh").inventory).encode("utf-8"))
        if bank["profile_id"]=="zh-cvvc":
            yaml_metadata["default_phonemizer"]="OpenUtau.Plugin.Builtin.ChineseCVVCPhonemizer"
    atomic_bytes(root / "character.yaml", yaml.safe_dump(yaml_metadata).encode("utf-8"))
    write_json(root / "missing-units.json", {"missing": missing, "failures": failures})
    notation = bank["preset"].get("alias_format", DEFAULT_ALIAS_FORMAT)
    atomic_bytes(root / "README.txt", ("VoiceBankGen 草稿音源\r\n格式：" + bank["profile_id"] + "\r\noto 发音格式：" + notation + "\r\n缺音数：" + str(len(missing)) + "\r\n编号备选音可在乐谱中用 ア_1 / あ_1 / a_1 等别名指定；呼吸音使用 br / br_1。\r\n拼接音保留来源标记，请试听。OpenUtau 更新后请重新载入音源。\r\n" + ("罗马音或片假名请使用能直接保留别名的音素器；JA CVVC（legacy）请在音源设置中选择平假名。\r\n" if notation in {"romaji","katakana"} and bank["profile_id"].startswith("ja-") else ("请选择 JA CVVC phonemizer，VC/VV 缺项见 missing-units.json。\r\n" if bank["profile_id"] == "ja-cvvc" else ""))).encode("utf-8"))
    if bank["profile_id"].startswith("zh-"):
        atomic_bytes(root/"README.txt",("VoiceBankGen 普通话草稿音源\r\n拼音别名（ü 写 v），CVVC 请选择 OpenUtau ZH CVVC；CV 可用默认音素器直接使用拼音。\r\n编号备选例如 a_1，呼吸 br。复合韵母保留尾音，短音仅提供对齐初始标线，需试听校对。\r\n跨语种近似不能还原送气、卷舌、圆唇等差异；来源和损失说明见 uforge.manifest.json，缺音见 missing-units.json。\r\n").encode("utf-8"))
    manifest = write_bank_metadata(store, bank, root, entries, missing, wav_hashes=wav_hashes, processing_key=processing_key)
    write_json(build_root / "processing.json", {"target_hz": target_hz, "processing_key": processing_key, "transformations": transformations, "failures": failures})
    emit(stage="完成增量准备", current=rendered_count+timing_count, total=rendered_count+timing_count)
    with store.connection() as connection:
        current = connection.execute("SELECT generation FROM banks WHERE id=?", (bank_id,)).fetchone()[0]
        if current != generation or store.revision != captured_revision:
            return {"stale": True, "path": str(root)}
        if target_hz is not None:
            from .shared_banks import raw_bank
            owner=store.cv_base_id(bank_id)
            preset=raw_bank(store,owner)["preset"]
            preset["resolved_hz"]=target_hz
            connection.execute("UPDATE banks SET preset=? WHERE id=?",(json.dumps(preset),owner))
        connection.execute("UPDATE banks SET build_path=? WHERE id=?", (root.relative_to(store.directory).as_posix(), bank_id))
    return {"path": str(root), "rendered": rendered_count, "updated_timings": timing_count, "unchanged": len(selected)-rendered_count-timing_count,
            "exported": len({entry["candidate_id"] for entry in entries}), "missing": len(missing), "failures": failures,
            "language_coverage": {"native":len({e["selection_alias"] for e in entries if not e.get("cross_language") and e["provenance_kind"]=="original"}),"cross_language":len({e["selection_alias"] for e in entries if e.get("cross_language")}),"approximate":len({e["selection_alias"] for e in entries if (e.get("cross_language") or {}).get("grade")=="approximate"})},
            "coverage": {kind: len({entry["selection_alias"] for entry in entries if entry["unit_type"] == kind}) for kind in ("V", "CV", "VC", "VV", "呼吸", "自定义")}, "manifest": manifest}


def refresh_timings(store: ProjectStore, bank_id: str, generation: int | None = None) -> dict:
    # One incremental preparation path for timing, selection and preset edits.
    generation = store.bank(bank_id)["generation"] if generation is None else generation
    return compile_bank(store, bank_id, generation)


def export_zip(store: ProjectStore, bank_id: str, destination: Path) -> Path:
    bank = store.bank(bank_id)
    if not bank["build_path"]:
        raise ValueError("请先制作音源")
    root = safe_path(store.directory, bank["build_path"])
    if destination.exists():
        raise ValueError("ZIP 已存在；请选择新文件名以保留旧版本")
    marker, manifest = load_manifest(root)
    with ZipFile(destination, "x", compression=ZIP_DEFLATED) as archive:
        for name in [*manifest["files"], "uforge.manifest.json", "uforge.project-link.json"]:
            archive.write(safe_path(root, name), arcname=bank["name"] + "/" + name)
    return destination
