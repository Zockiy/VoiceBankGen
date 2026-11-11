"""Human gold annotations and measured candidate quality, without invented ground truth."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .models import Candidate, ReviewEvent
from .storage import json_hash, read_jsonl, sha256_file, write_json, write_jsonl


class GoldAnnotation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidate_id: str
    split: Literal["development", "validation"] = "validation"
    gold_alias: str | None = None
    sustainable: bool | None = None
    preutter_range: tuple[int, int] | None = None
    fixed_end_range: tuple[int, int] | None = None
    rejection_reason: str | None = None

    @model_validator(mode="after")
    def check_ranges(self) -> "GoldAnnotation":
        for label, interval in (("preutter_range", self.preutter_range), ("fixed_end_range", self.fixed_end_range)):
            if interval is not None and (interval[0] < 0 or interval[1] < interval[0]):
                raise ValueError(f"{label} must be ordered nonnegative absolute sample coordinates")
        if self.sustainable is False and (self.preutter_range is not None or self.fixed_end_range is not None):
            raise ValueError("a non-sustainable example cannot have a positive sustain timing range")
        return self


def create_gold_template(project_dir: Path, output: Path | None = None) -> dict:
    candidates = [candidate for candidate in read_jsonl(project_dir / "candidates.jsonl", Candidate) if "STALE_EDIT" not in candidate.quality_flags]
    if not candidates:
        raise ValueError("run analyze before creating the gold template")
    output = output or project_dir / "evaluation" / "gold_annotations.jsonl"
    if output.exists():
        raise FileExistsError(f"gold file already exists; preserving human annotations: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    records = [GoldAnnotation(candidate_id=candidate.candidate_id) for candidate in candidates]
    write_jsonl(output, records)
    write_json(output.parent / "candidate_reference.json", {
        "schema_version": 1,
        "annotations_file": str(output.resolve()),
        "candidates_sha256": sha256_file(project_dir / "candidates.jsonl"),
        "candidates": [
            {"candidate_id": candidate.candidate_id, "suggested_alias": candidate.alias, "raw_phones": candidate.raw_phones,
             "source_id": candidate.source_id, "utterance_id": candidate.utterance_id, "flags": candidate.quality_flags,
             "suggested_landmarks": candidate.landmarks.model_dump() if candidate.landmarks else None}
            for candidate in candidates
        ],
    })
    return {"path": str(output.resolve()), "candidates": len(records), "status": "awaiting_human_annotations"}


def get_gold_annotation(project_dir: Path, candidate_id: str) -> GoldAnnotation:
    path = project_dir / "evaluation" / "gold_annotations.jsonl"
    records = read_jsonl(path, GoldAnnotation)
    return next((row for row in records if row.candidate_id == candidate_id), GoldAnnotation(candidate_id=candidate_id))


def update_gold_annotation(project_dir: Path, annotation: GoldAnnotation) -> None:
    known = {candidate.candidate_id for candidate in read_jsonl(project_dir / "candidates.jsonl", Candidate)}
    if annotation.candidate_id not in known:
        raise ValueError("gold candidate_id is absent from the current project")
    path = project_dir / "evaluation" / "gold_annotations.jsonl"
    records = read_jsonl(path, GoldAnnotation)
    previous = next((row.model_dump(mode="json") for row in records if row.candidate_id == annotation.candidate_id), {})
    for index, row in enumerate(records):
        if row.candidate_id == annotation.candidate_id:
            records[index] = annotation
            break
    else:
        records.append(annotation)
    write_jsonl(path, records)
    event_path = project_dir / "edits.jsonl"
    events = read_jsonl(event_path, ReviewEvent)
    events.append(ReviewEvent(kind="gold", candidate_id=annotation.candidate_id, before=previous, after=annotation.model_dump(mode="json")))
    write_jsonl(event_path, events)


def _interval_error_ms(value: int, interval: tuple[int, int]) -> float:
    return max(interval[0] - value, 0, value - interval[1]) * 1000 / 44100


def _percentile(values: list[float], percentile: int) -> float | None:
    return float(np.percentile(values, percentile)) if values else None


def _metrics(rows: list[tuple[GoldAnnotation, Candidate]], selected_ids: set[str]) -> dict:
    positive = [(gold, candidate) for gold, candidate in rows if gold.sustainable is True]
    negative = [(gold, candidate) for gold, candidate in rows if gold.sustainable is False]
    suggestions = [(gold, candidate) for gold, candidate in rows if candidate.candidate_id in selected_ids]
    timed = [(gold, candidate) for gold, candidate in positive if gold.preutter_range and gold.fixed_end_range and candidate.landmarks]
    p_errors = [_interval_error_ms(candidate.landmarks.preutter, gold.preutter_range) for gold, candidate in timed]
    f_errors = [_interval_error_ms(candidate.landmarks.fixed_end, gold.fixed_end_range) for gold, candidate in timed]
    baseline = [(gold, candidate.phonetic_context["vowel_span"]["start"]) for gold, candidate in positive
                if gold.preutter_range and gold.fixed_end_range and isinstance(candidate.phonetic_context.get("vowel_span"), dict)]
    baseline_p = [_interval_error_ms(start, gold.preutter_range) for gold, start in baseline]
    baseline_f = [_interval_error_ms(start, gold.fixed_end_range) for gold, start in baseline]
    alias_correct = sum(candidate.alias == gold.gold_alias for gold, candidate in suggestions if gold.gold_alias is not None)
    alias_total = sum(gold.gold_alias is not None for gold, _ in suggestions)
    negative_false_positive = sum(candidate.landmarks is not None and not candidate.quality_flags for _, candidate in negative)
    return {
        "annotated": len(rows), "positive": len(positive), "negative": len(negative),
        "selected_suggestions": len(suggestions), "suggestion_alias_correct": alias_correct, "suggestion_alias_total": alias_total,
        "suggestion_alias_accuracy_pct": 100 * alias_correct / alias_total if alias_total else None,
        "timed_positive": len(timed), "preutter_error_median_ms": median(p_errors) if p_errors else None,
        "preutter_error_p90_ms": _percentile(p_errors, 90), "fixed_end_error_median_ms": median(f_errors) if f_errors else None,
        "fixed_end_error_p90_ms": _percentile(f_errors, 90),
        "baseline_timed": len(baseline), "mfa_baseline_preutter_median_ms": median(baseline_p) if baseline_p else None,
        "mfa_baseline_preutter_p90_ms": _percentile(baseline_p, 90),
        "mfa_baseline_fixed_end_median_ms": median(baseline_f) if baseline_f else None,
        "mfa_baseline_fixed_end_p90_ms": _percentile(baseline_f, 90),
        "negative_false_sustain": negative_false_positive,
        "negative_false_sustain_pct": 100 * negative_false_positive / len(negative) if negative else None,
        "human_reviewed": sum(candidate.review_status != "unreviewed" for _, candidate in rows),
    }


def evaluate(project_dir: Path, annotations: Path) -> dict:
    candidates = {candidate.candidate_id: candidate for candidate in read_jsonl(project_dir / "candidates.jsonl", Candidate)}
    gold = read_jsonl(annotations, GoldAnnotation)
    identifiers = [row.candidate_id for row in gold]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("duplicate gold candidate_id")
    missing = sorted(set(identifiers) - set(candidates))
    if missing:
        raise ValueError(f"gold references missing or stale candidates: {missing[:10]}")
    complete = [row for row in gold if row.sustainable is False or (row.sustainable is True and row.gold_alias is not None and row.preutter_range is not None and row.fixed_end_range is not None)]
    stale_complete = [row.candidate_id for row in complete if "STALE_EDIT" in candidates[row.candidate_id].quality_flags]
    if stale_complete:
        raise ValueError(f"gold annotations refer to stale candidates: {stale_complete[:10]}")
    selection_path = project_dir / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8")) if selection_path.exists() else {"chosen": {}}
    selected_ids = set(selection["chosen"].values())
    rows = [(row, candidates[row.candidate_id]) for row in complete]
    total = _metrics(rows, selected_ids)
    development = _metrics([(row, candidate) for row, candidate in rows if row.split == "development"], selected_ids)
    validation = _metrics([(row, candidate) for row, candidate in rows if row.split == "validation"], selected_ids)
    gate_reasons = []
    if total["annotated"] < 60 or total["positive"] < 20 or total["negative"] < 20:
        gate_reasons.append("INSUFFICIENT_GOLD_60_20_20")
    if validation["annotated"] < 20 or validation["positive"] < 10 or validation["negative"] < 10 or validation["suggestion_alias_total"] < 10:
        gate_reasons.append("VALIDATION_SPLIT_INCOMPLETE")
    if validation["suggestion_alias_accuracy_pct"] is not None and validation["suggestion_alias_accuracy_pct"] < 95:
        gate_reasons.append("ALIAS_ACCURACY_BELOW_TARGET")
    if validation["preutter_error_median_ms"] is not None and validation["preutter_error_median_ms"] > 20:
        gate_reasons.append("PREUTTER_MEDIAN_BELOW_TARGET")
    if validation["fixed_end_error_median_ms"] is not None and validation["fixed_end_error_median_ms"] > 20:
        gate_reasons.append("FIXED_END_MEDIAN_BELOW_TARGET")
    if validation["preutter_error_p90_ms"] is not None and validation["preutter_error_p90_ms"] > 50:
        gate_reasons.append("PREUTTER_P90_BELOW_TARGET")
    if validation["fixed_end_error_p90_ms"] is not None and validation["fixed_end_error_p90_ms"] > 50:
        gate_reasons.append("FIXED_END_P90_BELOW_TARGET")
    if validation["negative_false_sustain_pct"] is not None and validation["negative_false_sustain_pct"] > 10:
        gate_reasons.append("FALSE_SUSTAIN_ABOVE_TARGET")
    report = {"status": "candidate_metrics_passed" if not gate_reasons else "not_passed", "gate_reasons": gate_reasons,
              "gold_sha256": sha256_file(annotations), "candidate_set_sha256": sha256_file(project_dir / "candidates.jsonl"),
              "annotated": len(complete), "incomplete": len(gold) - len(complete), "total": total,
              "development": development, "validation": validation,
              "method": "sample interval error; same gold entries compared with direct MFA vowel-start baseline",
              "scope": "candidate proposal metrics only; not a voicebank release decision",
              "not_measured": ["actual rendered audio quality", "automatic acceptance precision"]}
    report_path = project_dir / "evaluation" / f"report_{json_hash({'gold': report['gold_sha256'], 'candidate_set': report['candidate_set_sha256']})[:16]}.json"
    write_json(report_path, report)
    return {**report, "report_path": str(report_path.resolve())}
