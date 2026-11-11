"""Conservative, provenance-rich vowel tail and experimental C+V repair."""

from __future__ import annotations

import math
import shutil
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import numpy as np
import soundfile as sf

from .analyze import _features, _longest_sustain, _score
from .audio import read_region
from .config import Config
from .languages import language, profile
from .models import Candidate, Metrics, ProvenanceKind, SampleSpan, SourceAsset, TargetUnit, Utterance
from .storage import read_jsonl, write_json, write_jsonl
from .storage import json_hash


def _derived_id(mode, source, donor, alias, config):
    settings = json_hash(config.repair.model_dump())
    return str(uuid5(NAMESPACE_URL, f"{mode}:{source.candidate_id}:{donor.candidate_id}:{alias}:{settings}"))


def _voiced_f0(audio: np.ndarray, config: Config, region_start: int, region_end: int) -> float | None:
    import pyworld
    values, times = pyworld.harvest(audio.astype("float64"), 44100, f0_floor=config.features.f0_min_hz, f0_ceil=config.features.f0_max_hz, frame_period=config.features.hop_ms)
    values = pyworld.stonemask(audio.astype("float64"), values, times, 44100)
    selected = (times * 44100 >= region_start) & (times * 44100 < region_end) & (values > 0)
    voiced = values[selected]
    if len(voiced) >= 3:
        return float(np.median(voiced))
    short = audio[region_start:region_end].astype("float64")
    if len(short) < round(0.03 * 44100) or float(np.sqrt(np.mean(short * short))) < 1e-4:
        return None
    short -= np.mean(short)
    nfft = 1 << (2 * len(short) - 1).bit_length()
    spectrum = np.fft.rfft(short * np.hanning(len(short)), n=nfft)
    correlation = np.fft.irfft(np.abs(spectrum) ** 2, n=nfft)
    low = max(1, 44100 // config.features.f0_max_hz)
    high = min(len(short) // 2, 44100 // config.features.f0_min_hz)
    if high <= low or correlation[0] <= 1e-10:
        return None
    lag = low + int(np.argmax(correlation[low:high + 1]))
    periodicity = correlation[lag] / correlation[0]
    return float(44100 / lag) if periodicity >= config.sustain.min_periodicity else None


def _join(source: np.ndarray, donor: np.ndarray, f0: float, config: Config) -> tuple[np.ndarray, dict]:
    fs = 44100
    crossfade = round((config.repair.crossfade_ms if config.repair.crossfade_ms is not None else np.clip(3000 / f0, 10, 40)) * fs / 1000)
    period = max(1, round(fs / f0))
    if len(source) < 2 * crossfade or len(donor) < crossfade + period + round(config.repair.min_donor_ms * fs / 1000) // 2:
        raise ValueError("INSUFFICIENT_JOIN_AUDIO")
    source_tail = source[-crossfade:].astype("float64")
    source_rms = float(np.sqrt(np.mean(source_tail * source_tail)))
    if source_rms < 1e-4:
        raise ValueError("NO_VOICED_SOURCE_VOWEL")
    best_shift = 0
    best_correlation = -2.0
    for shift in range(min(period + 1, len(donor) - crossfade)):
        part = donor[shift:shift + crossfade].astype("float64")
        denom = np.linalg.norm(source_tail) * np.linalg.norm(part)
        correlation = float(np.dot(source_tail, part) / denom) if denom else -1
        if correlation > best_correlation:
            best_correlation, best_shift = correlation, shift
    if best_correlation < 0.15:
        raise ValueError("PHASE_MATCH_FAILED")
    aligned = donor[best_shift:].astype("float64")
    donor_rms = float(np.sqrt(np.mean(aligned[:crossfade] ** 2)))
    gain_db = 20 * math.log10(source_rms / max(donor_rms, 1e-8))
    if abs(gain_db) > config.level.max_gain_db:
        raise ValueError("DONOR_LEVEL_MISMATCH")
    aligned *= 10 ** (gain_db / 20)
    weights = np.linspace(0, 1, crossfade, endpoint=True)
    blend = source_tail * (1 - weights) + aligned[:crossfade] * weights
    output = np.concatenate((source[:-crossfade], blend, aligned[crossfade:])).astype("float32")
    if not bool(np.isfinite(output).all()) or float(np.max(np.abs(output))) > 1.0:
        raise ValueError("DERIVED_AUDIO_CLIPS")
    return output, {"crossfade_samples": crossfade, "donor_shift_samples": best_shift, "correlation": best_correlation, "gain_db": gain_db, "join_source_samples": len(source) - crossfade}


def _source_join_f0(project_dir: Path, config: Config, source: Candidate, assets: dict[str, SourceAsset]) -> float | None:
    vowel = SampleSpan.model_validate(source.phonetic_context["vowel_span"])
    join_end = min(vowel.end, vowel.start + round(0.07 * 44100))
    context_start = max(0, source.span.start - round(0.1 * 44100))
    context_end = min(assets[source.source_id].canonical_samples, vowel.end + round(0.1 * 44100))
    context_audio = read_region(project_dir, assets[source.source_id], context_start, context_end)
    return _voiced_f0(context_audio, config, vowel.start - context_start, join_end - context_start)


def _derive(project_dir: Path, config: Config, source: Candidate, donor: Candidate, *, mode: str, alias: str, phones: list[str], unit_key: str, assets: dict[str, SourceAsset], source_f0: float | None = None) -> Candidate:
    if source.speaker_id != donor.speaker_id or source.source_language != donor.source_language:
        raise ValueError("DONOR_LANGUAGE_OR_SPEAKER_MISMATCH")
    if donor.landmarks is None or (donor.metrics.stable_vowel_ms or 0) < config.repair.min_donor_ms:
        raise ValueError("NO_STABLE_DONOR")
    vowel = SampleSpan.model_validate(source.phonetic_context["vowel_span"])
    if vowel.end - vowel.start < round(0.03 * 44100):
        raise ValueError("SOURCE_VOWEL_TOO_SHORT")
    join_end = min(vowel.end, vowel.start + round(0.07 * 44100))
    if join_end - vowel.start < round(0.03 * 44100):
        raise ValueError("NO_TARGET_VOWEL_ONSET")
    source_audio = read_region(project_dir, assets[source.source_id], source.span.start, join_end)
    f0_source = source_f0 or _source_join_f0(project_dir, config, source, assets)
    f0_donor = donor.metrics.f0_median_hz
    if f0_source is None or not f0_donor:
        raise ValueError("UNMEASURABLE_JOIN_F0")
    if abs(12 * math.log2(f0_source / f0_donor)) > 1.0:
        raise ValueError("DONOR_PITCH_MISMATCH")
    donor_audio = read_region(project_dir, assets[donor.source_id], donor.landmarks.fixed_end, min(donor.landmarks.end, donor.landmarks.fixed_end + round(0.5 * 44100)))
    output, join_record = _join(source_audio, donor_audio, f0_source, config)
    identifier = _derived_id(mode, source, donor, alias, config)
    derived_dir = project_dir / "derived"
    derived_dir.mkdir(exist_ok=True)
    path = derived_dir / f"{identifier}.wav"
    candidate = Candidate(candidate_id=identifier, speaker_id=source.speaker_id, source_id=source.source_id, utterance_id=source.utterance_id, alignment_revision=source.alignment_revision, source_language=source.source_language, unit_role="sustain_vowel", unit_key=unit_key, alias=alias, phones=phones, raw_phones=[*source.raw_phones, *donor.raw_phones], span=SampleSpan(start=0, end=len(output)), provenance_kind=ProvenanceKind.vowel_extended if mode == "vowel-tail" else ProvenanceKind.cv_spliced, parent_candidate_ids=[source.candidate_id, donor.candidate_id], phonetic_context={"vowel_span": {"start": vowel.start - source.span.start, "end": len(output)}, "derived_audio_path": path.relative_to(project_dir).as_posix(), "join": join_record, "source_range": [source.span.start, join_end], "donor_range": [donor.landmarks.fixed_end, min(donor.landmarks.end, donor.landmarks.fixed_end + len(donor_audio))]}, recording_session=source.recording_session)
    faux = Utterance(source_id=source.source_id, speaker_id=source.speaker_id, span=SampleSpan(start=0, end=len(output)), vad_span=SampleSpan(start=0, end=len(output)), corpus_offset_samples=0, language_id=source.source_language)
    features = _features(output, 44100, config)
    marks, data = _longest_sustain(candidate, features, faux, config)
    if marks is None:
        raise ValueError("DERIVED_NO_SUSTAIN_REGION")
    candidate.landmarks = marks
    candidate.automatic_landmarks = marks
    candidate.metrics = Metrics(**data)
    candidate.quality_flags = ["REPAIR_REVIEW_REQUIRED"] + (["COARTICULATION_UNCERTAIN"] if mode == "cv-splice" else [])
    _score(candidate, config)
    if not path.exists():
        sf.write(path, output, 44100, subtype="FLOAT")
    return candidate


def repair_project(project_dir: Path, config: Config, *, mode: str, experimental: bool = False, alias_filter: str | None = None) -> dict:
    if mode not in {"vowel-tail", "cv-splice"}:
        raise ValueError("invalid repair mode")
    if mode == "cv-splice" and not experimental:
        raise ValueError("cv-splice requires --experimental")
    candidates = read_jsonl(project_dir / "candidates.jsonl", Candidate)
    assets = {item.source_id: item for item in read_jsonl(project_dir / "sources" / "manifest.jsonl", SourceAsset)}
    donors = [item for item in candidates if item.provenance_kind == ProvenanceKind.original and item.landmarks is not None and (item.metrics.stable_vowel_ms or 0) >= config.repair.min_donor_ms and item.review_status != "rejected" and not item.quality_flags]
    present = {item.alias for item in candidates if item.landmarks is not None and
               (not item.quality_flags or (item.review_status == "accepted" and set(item.quality_flags) <= {"COARTICULATION_UNCERTAIN"}))}
    if mode == "cv-splice":
        present.update(item.alias for item in candidates if item.provenance_kind == ProvenanceKind.vowel_extended
                       and item.landmarks is not None and item.review_status != "rejected"
                       and set(item.quality_flags) <= {"REPAIR_REVIEW_REQUIRED"})
    generated: list[Candidate] = []
    reused_aliases: list[str] = []
    failures: list[dict[str, str]] = []
    targets: list[tuple[str, list[str], list[Candidate]]] = []
    if mode == "vowel-tail":
        targets = [(item.alias, item.phones, [item]) for item in candidates if item.provenance_kind == ProvenanceKind.original and item.landmarks is None and item.unit_role == "sustain_vowel" and item.alias not in present and (alias_filter is None or item.alias == alias_filter)]
    else:
        inventory = language(profile(config.export_profiles[0]).language_id).inventory["units"]
        for alias, seq in inventory.items():
            if alias in present or (alias_filter is not None and alias != alias_filter) or len(seq) < 2:
                continue
            # Exact canonical onset keeps shi/su, chi/tsu and palatalized/plain
            # consonants separate. A different source vowel makes this P7b, not P7a.
            onsets = [item for item in candidates if item.provenance_kind == ProvenanceKind.original
                      and item.phones[:-1] == list(seq[:-1]) and item.alias != alias
                      and item.source_language in config.target_languages and item.review_status != "rejected"
                      and "vowel_span" in item.phonetic_context]
            onsets.sort(key=lambda item: (item.recording_session is None, -(item.quality_score or 0), item.source_id, item.span.start))
            targets.append((alias, list(seq), onsets[:12]))
    existing_by_id = {item.candidate_id: item for item in candidates}
    existing_ids = set(existing_by_id)
    f0_cache: dict[str, float | None] = {}
    attempted_pairs = 0
    for alias, seq, onsets in targets:
        reasons: set[str] = set()
        if not onsets:
            failures.append({"alias": alias, "reason": "NO_COMPATIBLE_CONSONANT_ONSET"})
            continue
        made = False
        for source in onsets:
            vowel_span = SampleSpan.model_validate(source.phonetic_context["vowel_span"])
            if mode == "cv-splice" and vowel_span.start - source.span.start < round(0.01 * 44100):
                reasons.add("NO_CONSONANT_AUDIO")
                continue
            vowel_donors = [donor for donor in donors if donor.phones and donor.phones[-1] == seq[-1]
                            and donor.speaker_id == source.speaker_id and donor.source_language == source.source_language
                            and donor.candidate_id != source.candidate_id]
            if not vowel_donors:
                reasons.add("NO_COMPATIBLE_VOWEL_DONOR")
                continue
            if source.candidate_id not in f0_cache:
                f0_cache[source.candidate_id] = _source_join_f0(project_dir, config, source, assets)
            source_f0 = f0_cache[source.candidate_id]
            if source_f0 is None:
                reasons.add("UNMEASURABLE_JOIN_F0")
                continue
            vowel_donors.sort(key=lambda item: (item.recording_session != source.recording_session,
                               abs(12 * math.log2(source_f0 / item.metrics.f0_median_hz)) if item.metrics.f0_median_hz else float("inf"),
                               -(item.metrics.stable_vowel_ms or 0), item.source_id, item.span.start))
            for donor in vowel_donors[:16]:
                if not donor.metrics.f0_median_hz or abs(12 * math.log2(source_f0 / donor.metrics.f0_median_hz)) > 1.0:
                    reasons.add("DONOR_PITCH_MISMATCH")
                    continue
                identifier = _derived_id(mode, source, donor, alias, config)
                if identifier in existing_by_id:
                    if existing_by_id[identifier].review_status == "rejected":
                        reasons.add("PREVIOUS_SPLICE_REJECTED")
                        continue
                    reused_aliases.append(alias)
                    made = True
                    break
                try:
                    attempted_pairs += 1
                    candidate = _derive(project_dir, config, source, donor, mode=mode, alias=alias,
                                        phones=seq, unit_key="+".join(seq), assets=assets, source_f0=source_f0)
                    if candidate.candidate_id not in existing_ids:
                        generated.append(candidate)
                        existing_ids.add(candidate.candidate_id)
                    made = True
                    break
                except ValueError as exc:
                    reasons.add(str(exc))
            if made:
                break
        if not made:
            failures.append({"alias": alias, "reason": ",".join(sorted(reasons)) or "NO_VALID_SPLICE"})
    write_jsonl(project_dir / "candidates.jsonl", [*candidates, *generated])
    preview_dir = project_dir / "review_splices"
    preview_rows: list[dict] = []
    if mode == "cv-splice":
        preview_dir.mkdir(exist_ok=True)
        export_profile = profile(config.export_profiles[0])
        inventory_version = language(export_profile.language_id).inventory["version"]
        for item in [*candidates, *generated]:
            if item.provenance_kind != ProvenanceKind.cv_spliced:
                continue
            unit = TargetUnit(target_language=item.source_language, inventory_version=inventory_version,
                              unit_type="CV", unit_key=item.unit_key)
            stem = export_profile.filename_stem(unit)
            preview_name = f"{stem}_{item.candidate_id[:8]}.wav"
            if not (preview_dir / preview_name).exists():
                shutil.copyfile(project_dir / item.phonetic_context["derived_audio_path"], preview_dir / preview_name)
            preview_rows.append({"alias": item.alias, "romaji_wav": preview_name,
                                 "candidate_id": item.candidate_id, "parent_candidate_ids": item.parent_candidate_ids,
                                 "review_status": item.review_status, "quality_flags": item.quality_flags})
        write_json(preview_dir / "manifest.json", {"candidates": preview_rows,
                   "warning": "Experimental C+V previews require listening and acceptance before voicebank selection."})
    report = {"mode": mode, "generated": len(generated), "generated_aliases": [item.alias for item in generated],
              "already_generated": len(reused_aliases), "already_generated_aliases": reused_aliases,
              "failed": len(failures), "attempted_pairs": attempted_pairs, "failures": failures,
              "default_review_status": "unreviewed", "preview_directory": str(preview_dir) if preview_rows else None,
              "preview_count": len(preview_rows)}
    write_json(project_dir / "runs" / f"repair_{mode}_{uuid5(NAMESPACE_URL, str(len(candidates)) + str(len(generated)))}.json", report)
    return report
