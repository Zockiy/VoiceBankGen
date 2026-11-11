import json

import pytest

from utau_forge.evaluation import GoldAnnotation, create_gold_template, evaluate, get_gold_annotation, update_gold_annotation
from utau_forge.models import Candidate, Landmarks, ReviewEvent, SampleSpan
from utau_forge.project import initialize
from utau_forge.review import undo_last_edit
from utau_forge.storage import read_jsonl, write_json, write_jsonl


def test_gold_evaluation_compares_intervals_and_mfa_baseline(tmp_path):
    project = tmp_path / "project"
    initialize(project)
    positive = Candidate(candidate_id="good", speaker_id="speaker_001", source_id="s", utterance_id="u", alignment_revision="r", source_language="ja", unit_role="sustain_vowel", unit_key="k+a", alias="か", phones=["k", "a"], raw_phones=["k", "a"], span=SampleSpan(start=0, end=10000), landmarks=Landmarks(start=0, overlap=0, preutter=2000, fixed_end=3000, end=9000), phonetic_context={"vowel_span": {"start": 1000, "end": 9000}})
    negative = Candidate(candidate_id="bad", speaker_id="speaker_001", source_id="s", utterance_id="u", alignment_revision="r", source_language="ja", unit_role="sustain_vowel", unit_key="s+i", alias="し", phones=["s", "i"], raw_phones=["s", "i"], span=SampleSpan(start=10000, end=20000), landmarks=None)
    write_jsonl(project / "candidates.jsonl", [positive, negative])
    write_json(project / "selection.json", {"chosen": {"か": "good", "し": "bad"}})
    template = create_gold_template(project)
    assert template["candidates"] == 2
    with pytest.raises(FileExistsError):
        create_gold_template(project)
    gold = project / "evaluation" / "gold.jsonl"
    write_jsonl(gold, [GoldAnnotation(candidate_id="good", gold_alias="か", sustainable=True, preutter_range=(1900, 2100), fixed_end_range=(2900, 3100)), GoldAnnotation(candidate_id="bad", gold_alias="し", sustainable=False, rejection_reason="devoiced")])
    result = evaluate(project, gold)
    assert result["status"] == "not_passed"
    assert "INSUFFICIENT_GOLD_60_20_20" in result["gate_reasons"]
    assert result["validation"]["suggestion_alias_accuracy_pct"] == 100
    assert result["validation"]["preutter_error_median_ms"] == 0
    assert result["validation"]["mfa_baseline_fixed_end_median_ms"] > 0
    assert result["validation"]["negative_false_sustain"] == 0
    assert json.loads((project / "evaluation" / "candidate_reference.json").read_text(encoding="utf-8"))["candidates_sha256"]


def test_candidate_metric_gate_needs_a_real_validation_split(tmp_path):
    project = tmp_path / "project"
    initialize(project)
    candidates = []
    annotations = []
    chosen = {}
    for index in range(60):
        alias = f"unit_{index}"
        positive = index % 2 == 0
        identifier = f"c{index}"
        candidates.append(Candidate(candidate_id=identifier, speaker_id="speaker_001", source_id="s", utterance_id="u", alignment_revision="r", source_language="ja", unit_role="sustain_vowel", unit_key=alias, alias=alias, phones=["a"], raw_phones=["a"], span=SampleSpan(start=0, end=10000), landmarks=Landmarks(start=0, overlap=0, preutter=2000, fixed_end=3000, end=9000) if positive else None, phonetic_context={"vowel_span": {"start": 1000, "end": 9000}}))
        annotations.append(GoldAnnotation(candidate_id=identifier, split="validation" if index < 20 else "development", gold_alias=alias, sustainable=positive, preutter_range=(1900, 2100) if positive else None, fixed_end_range=(2900, 3100) if positive else None))
        chosen[alias] = identifier
    write_jsonl(project / "candidates.jsonl", candidates)
    write_json(project / "selection.json", {"chosen": chosen})
    gold = project / "gold.jsonl"
    write_jsonl(gold, annotations)
    result = evaluate(project, gold)
    assert result["status"] == "candidate_metrics_passed"
    assert result["validation"]["positive"] == 10
    assert result["validation"]["negative"] == 10
    assert result["validation"]["fixed_end_error_p90_ms"] == 0


def test_gold_annotation_edit_is_logged_and_undo_restores_template_row(tmp_path):
    project = tmp_path / "project"
    initialize(project)
    candidate = Candidate(candidate_id="c1", speaker_id="speaker_001", source_id="s", utterance_id="u", alignment_revision="r", source_language="ja", unit_role="sustain_vowel", unit_key="s+i", alias="し", phones=["s", "i"], raw_phones=["s", "i"], span=SampleSpan(start=0, end=10000))
    write_jsonl(project / "candidates.jsonl", [candidate])
    create_gold_template(project)
    original = get_gold_annotation(project, "c1")
    update_gold_annotation(project, GoldAnnotation(candidate_id="c1", gold_alias="し", sustainable=False, rejection_reason="devoiced"))
    assert get_gold_annotation(project, "c1").rejection_reason == "devoiced"
    events = read_jsonl(project / "edits.jsonl", ReviewEvent)
    assert events[-1].kind == "gold"
    assert events[-1].before == original.model_dump(mode="json")
    assert undo_last_edit(project) == "gold"
    assert get_gold_annotation(project, "c1") == original
    assert read_jsonl(project / "edits.jsonl", ReviewEvent)[-1].kind == "undo"
