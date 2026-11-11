"""Conservative acoustic analysis and initial CV timing proposals."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .audio import read_region
from .config import Config
from .languages import language, profile
from .models import Candidate, Landmarks, Phone, ProvenanceKind, SampleSpan, SourceAsset, Utterance
from .storage import read_jsonl, write_jsonl


def _features(audio: np.ndarray, fs: int, config: Config) -> dict[str, np.ndarray]:
    try:
        import pyworld
    except ImportError as exc:
        raise RuntimeError("PyWORLD missing; install with uv sync --extra pitch") from exc
    waveform = audio.astype("float64")
    f0, times = pyworld.harvest(waveform, fs, f0_floor=config.features.f0_min_hz, f0_ceil=config.features.f0_max_hz, frame_period=config.features.hop_ms)
    f0 = pyworld.stonemask(waveform, f0, times, fs)
    centers = np.clip(np.rint(times * fs).astype(int), 0, len(audio) - 1)
    half_window = round(0.0125 * fs)
    lag_min = max(1, fs // config.features.f0_max_hz)
    lag_max = fs // config.features.f0_min_hz
    rms_db = np.empty(len(centers))
    periodicity = np.empty(len(centers))
    spectral = np.empty((len(centers), 128), dtype="float32")
    for i, center in enumerate(centers):
        frame = np.zeros(2 * half_window, dtype="float64")
        lower = max(0, center - half_window)
        upper = min(len(audio), center + half_window)
        frame[lower - (center - half_window):upper - (center - half_window)] = waveform[lower:upper]
        rms = float(np.sqrt(np.mean(frame * frame)))
        rms_db[i] = 20 * math.log10(max(rms, 1e-8))
        windowed = frame * np.hanning(len(frame))
        fft_size = 1 << (2 * len(frame) - 1).bit_length()
        spectrum = np.fft.rfft(windowed, n=fft_size)
        acf = np.fft.irfft(np.abs(spectrum) ** 2, n=fft_size)
        periodicity[i] = float(np.max(acf[lag_min:min(lag_max + 1, len(acf))]) / acf[0]) if acf[0] > 1e-10 else 0.0
        log_spectrum = np.log1p(np.abs(spectrum[:128]))
        norm = np.linalg.norm(log_spectrum)
        spectral[i] = log_spectrum / norm if norm else log_spectrum
    change = np.zeros(len(centers))
    change[1:] = np.mean(np.abs(np.diff(spectral, axis=0)), axis=1)
    return {"centers": centers, "f0": f0, "rms_db": rms_db, "periodicity": periodicity, "spectral_change": change}


def _longest_sustain(candidate: Candidate, features: dict[str, np.ndarray], utterance: Utterance, config: Config, *, force: bool = False) -> tuple[Landmarks | None, dict[str, float | None]]:
    if candidate.unit_role not in {"sustain_vowel", "nasal"}:
        return None, {"reason": None}
    vowel = SampleSpan.model_validate(candidate.phonetic_context["vowel_span"])
    absolute = features["centers"] + utterance.span.start
    within = (absolute >= vowel.start) & (absolute < vowel.end)
    f0 = features["f0"]
    rms = features["rms_db"]
    periodicity = features["periodicity"]
    change = features["spectral_change"]
    reference = change[(f0 > 0) & (rms > -45)]
    if len(reference) >= 20:
        median = float(np.median(reference))
        mad = float(np.median(np.abs(reference - median)))
        change_limit = median + 2.5 * max(mad, 0.001)
    else:
        change_limit = 0.03
    acceptable = within & (f0 > 0) & (periodicity >= config.sustain.min_periodicity) & (rms > -45) & (change <= change_limit)
    starts = np.flatnonzero(acceptable & ~np.r_[False, acceptable[:-1]])
    ends = np.flatnonzero(acceptable & ~np.r_[acceptable[1:], False]) + 1
    best: tuple[int, int] | None = None
    best_metrics: dict[str, float | None] = {}
    for left, right in zip(starts, ends, strict=True):
        duration_ms = (absolute[right - 1] - absolute[left] + round(config.features.hop_ms * 44.1)) * 1000 / 44100
        if not force and duration_ms < config.sustain.min_ms:
            continue
        values = f0[left:right]
        median_f0 = float(np.median(values))
        f0_mad = float(np.median(np.abs(1200 * np.log2(values / median_f0))))
        rms_spread = float(1.4826 * np.median(np.abs(rms[left:right] - np.median(rms[left:right]))))
        if not force and (f0_mad > config.sustain.max_f0_mad_cents or rms_spread > config.sustain.max_rms_robust_spread_db):
            continue
        if best is None or (right - left) > (best[1] - best[0]):
            best = (left, right)
            best_metrics = {"stable_vowel_ms": duration_ms, "f0_median_hz": median_f0, "f0_mad_cents": f0_mad, "voiced_ratio": 1.0, "periodicity": float(np.median(periodicity[left:right])), "rms_spread_db": rms_spread, "spectral_stability": float(max(0, 1 - np.median(change[left:right]) / max(change_limit, 1e-6)))}
    if best is None:
        if force:
            # No acoustic sustain evidence: use aligned vowel boundaries without
            # reporting invented stability metrics. This is an explicit override.
            start = max(candidate.span.start, utterance.span.start)
            end = min(vowel.end, candidate.span.end, utterance.span.end)
            if end <= start:
                raise ValueError("音素边界与当前切片没有有效交集")
            preutter = max(start, min(end - 1, vowel.start))
            fixed = min(end - 1, preutter + round(config.features.hop_ms * 44.1))
            marks = Landmarks(start=start, overlap=start, preutter=preutter, fixed_end=fixed, end=end)
            marks.validate_cv(utterance.span.end)
            return marks, {}
        return None, {}
    left, right = best
    start = candidate.span.start
    fixed = int(absolute[left])
    end = min(vowel.end, int(absolute[right - 1] + round(config.features.hop_ms * 44.1)))
    if candidate.source_language=="zh":
        # Mandarin diphthongs and nasal codas must remain in the used window.
        end=vowel.end
    preutter = min(max(start, vowel.start), fixed)
    marks = Landmarks(start=start, overlap=start, preutter=preutter, fixed_end=fixed, end=end)
    marks.validate_cv(utterance.span.end)
    return marks, best_metrics


def _score(candidate: Candidate, config: Config) -> None:
    metrics = candidate.metrics
    components: dict[str, float | None] = {
        "sustain": min(1, (metrics.stable_vowel_ms or 0) / max(config.sustain.preferred_ms, 1)) if metrics.stable_vowel_ms is not None else None,
        "spectral_stability": metrics.spectral_stability,
        "pitch_stability": max(0, 1 - (metrics.f0_mad_cents or 0) / config.sustain.max_f0_mad_cents) if metrics.f0_mad_cents is not None else None,
        "cleanliness": None,
        "boundary_quality": 1.0 if candidate.landmarks is not None else None,
        "target_pitch_proximity": None,
    }
    weights = {"sustain": .30, "spectral_stability": .20, "pitch_stability": .15, "cleanliness": .15, "boundary_quality": .10, "target_pitch_proximity": .10}
    available = sum(weights[k] for k, v in components.items() if v is not None)
    metrics.components = components
    metrics.evidence_completeness = available
    candidate.quality_score = sum(weights[k] * max(0, min(1, v)) for k, v in components.items() if v is not None) / available if available else None


def analyze_project(project_dir: Path, config: Config) -> dict[str, int]:
    previous = read_jsonl(project_dir / "candidates.jsonl", Candidate)
    phones = read_jsonl(project_dir / "phones.jsonl", Phone)
    if not phones:
        raise ValueError("no aligned phones; run align first")
    utterances = {utt.utterance_id: utt for utt in read_jsonl(project_dir / "utterances.jsonl", Utterance)}
    assets = {asset.source_id: asset for asset in read_jsonl(project_dir / "sources" / "manifest.jsonl", SourceAsset)}
    by_utterance: dict[str, list[Phone]] = {}
    for phone in phones:
        by_utterance.setdefault(phone.utterance_id, []).append(phone)
    from .breaths import breath_candidates
    result: list[Candidate] = breath_candidates(phones, utterances)
    for utt_id, items in by_utterance.items():
        utt = utterances[utt_id]
        pack = language(utt.language_id)
        prof = profile(next((pid for pid in config.export_profiles if profile(pid).language_id==utt.language_id),utt.language_id+"-cv"))
        candidates = pack.propose_units(items, prof)
        if not candidates:
            continue
        features = _features(read_region(project_dir, assets[utt.source_id], utt.span.start, utt.span.end), 44100, config)
        for candidate in candidates:
            candidate.speaker_id = utt.speaker_id
            candidate.recording_session = assets[utt.source_id].recording_session
            if candidate.unit_role in {"sustain_vowel", "nasal"}:
                marks, metrics = (None, {}) if candidate.phonetic_context.get("devoiced_vowel") else _longest_sustain(candidate, features, utt, config)
                if marks is None and candidate.source_language=="zh":
                    vowel=SampleSpan.model_validate(candidate.phonetic_context["vowel_span"])
                    marks=Landmarks(start=candidate.span.start,overlap=candidate.span.start,preutter=vowel.start,fixed_end=min(vowel.end-1,vowel.start+882),end=vowel.end)
                    candidate.quality_flags=["ZH_TIMING_REVIEW"]
                    candidate.phonetic_context["timing_evidence"]="aligned-boundaries-only"
                candidate.landmarks = marks
                candidate.metrics = candidate.metrics.model_copy(update=metrics)
                if marks is None:
                    candidate.quality_flags = ["DEVOICED_VOWEL" if candidate.phonetic_context.get("devoiced_vowel") else "NO_SUSTAIN_REGION"]
                elif candidate.phonetic_context.get("timing_evidence")!="aligned-boundaries-only":
                    candidate.quality_flags = []
            else:
                candidate.quality_flags = ["SPECIAL_UNIT_REVIEW_REQUIRED"]
            candidate.automatic_landmarks = candidate.landmarks
            _score(candidate, config)
            result.append(candidate)
    old_by_identity = {(c.alignment_revision, c.source_id, c.utterance_id, c.unit_key, c.span.start, c.span.end): c for c in previous}
    retained_ids: set[str] = set()
    for candidate in result:
        identity = (candidate.alignment_revision, candidate.source_id, candidate.utterance_id, candidate.unit_key, candidate.span.start, candidate.span.end)
        old = old_by_identity.get(identity)
        if old:
            candidate.candidate_id = old.candidate_id
            candidate.review_status = old.review_status
            candidate.locked_fields = old.locked_fields
            if old.review_status != "unreviewed" or old.locked_fields:
                candidate.landmarks = old.landmarks
            retained_ids.add(old.candidate_id)
    current_ids = {candidate.candidate_id for candidate in result}
    for old in previous:
        if old.phonetic_context.get("manual_import"):
            result.append(old)
            continue
        if old.provenance_kind != ProvenanceKind.original:
            if (old.phonetic_context.get("cross_language") and not set(old.phonetic_context.get("phone_ids",[]))<={p.phone_id for p in phones}) or not set(old.parent_candidate_ids) <= current_ids:
                old.quality_flags = sorted(set([*old.quality_flags, "STALE_EDIT"]))
            result.append(old)
            continue
        if old.candidate_id not in retained_ids and (old.review_status != "unreviewed" or old.locked_fields):
            old.quality_flags = sorted(set([*old.quality_flags, "STALE_EDIT"]))
            result.append(old)
    write_jsonl(project_dir / "candidates.jsonl", result)
    return {"candidates": len(result), "timed": sum(c.landmarks is not None for c in result), "unknown_phones": sum(p.unknown for p in phones)}
