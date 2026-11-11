import json

import numpy as np
import soundfile as sf

from utau_forge.build import build, select
from utau_forge.models import Candidate, Landmarks, Metrics, SampleSpan, SourceAsset
from utau_forge.project import initialize, load
from utau_forge.repair import repair_project
from utau_forge.review import edit_candidate
from utau_forge.storage import read_jsonl, write_jsonl


def test_vowel_tail_creates_reviewable_provenance(tmp_path):
    project = tmp_path / "p"
    initialize(project)
    fs = 44100
    t = np.arange(fs) / fs
    audio = sum((0.14 / harmonic) * np.sin(2 * np.pi * 220 * harmonic * t) for harmonic in range(1, 9)).astype("float32")
    sf.write(project / "audio" / "canonical" / "s.wav", audio, fs, subtype="FLOAT")
    source = SourceAsset(source_id="s", original_path="s.wav", sha256="x", original_sample_rate=fs, original_channels=1, original_duration_s=1, canonical_path="audio/canonical/s.wav", canonical_samples=fs, speaker_id="speaker_001", language_id="ja")
    write_jsonl(project / "sources" / "manifest.jsonl", [source])
    short = Candidate(candidate_id="short", speaker_id="speaker_001", source_id="s", utterance_id="u", alignment_revision="r", source_language="ja", unit_role="sustain_vowel", unit_key="k+a", alias="か", phones=["k", "a"], raw_phones=["k", "a"], span=SampleSpan(start=0, end=5292), phonetic_context={"vowel_span": {"start": 0, "end": 5292}})
    donor = Candidate(candidate_id="donor", speaker_id="speaker_001", source_id="s", utterance_id="u", alignment_revision="r", source_language="ja", unit_role="sustain_vowel", unit_key="a", alias="あ", phones=["a"], raw_phones=["a"], span=SampleSpan(start=10000, end=24000), landmarks=Landmarks(start=10000, overlap=10000, preutter=10500, fixed_end=11000, end=23000), metrics=Metrics(stable_vowel_ms=272, f0_median_hz=220), quality_flags=[])
    write_jsonl(project / "candidates.jsonl", [short, donor])
    result = repair_project(project, load(project)[1], mode="vowel-tail", alias_filter="か")
    assert result["generated"] == 1, result
    candidates = read_jsonl(project / "candidates.jsonl", Candidate)
    repaired = candidates[-1]
    assert repaired.provenance_kind == "vowel_extended"
    assert repaired.parent_candidate_ids == ["short", "donor"]
    assert repaired.landmarks is not None
    assert repaired.review_status == "unreviewed"
    assert repaired.phonetic_context["derived_audio_path"].startswith("derived/")


def test_cv_splice_searches_past_incompatible_donor(tmp_path):
    project = tmp_path / "p"
    initialize(project)
    fs = 44100
    t = np.arange(fs) / fs
    audio = sum((0.14 / harmonic) * np.sin(2 * np.pi * 220 * harmonic * t) for harmonic in range(1, 9)).astype("float32")
    sf.write(project / "audio" / "canonical" / "s.wav", audio, fs, subtype="FLOAT")
    asset = SourceAsset(source_id="s", original_path="s.wav", sha256="x", original_sample_rate=fs,
                        original_channels=1, original_duration_s=1, canonical_path="audio/canonical/s.wav",
                        canonical_samples=fs, speaker_id="speaker_001", language_id="ja")
    write_jsonl(project / "sources" / "manifest.jsonl", [asset])
    onset = Candidate(candidate_id="ka", speaker_id="speaker_001", source_id="s", utterance_id="u",
                      alignment_revision="r", source_language="ja", unit_role="sustain_vowel", unit_key="k+a",
                      alias="か", phones=["k", "a"], raw_phones=["k", "a"], span=SampleSpan(start=0, end=6000),
                      phonetic_context={"vowel_span": {"start": 1000, "end": 6000}})
    def donor(identifier, pitch, stable):
        return Candidate(candidate_id=identifier, speaker_id="speaker_001", source_id="s", utterance_id="u",
                         alignment_revision="r", source_language="ja", unit_role="sustain_vowel", unit_key="u",
                         alias="う", phones=["u"], raw_phones=["u"], span=SampleSpan(start=20000, end=40000),
                         landmarks=Landmarks(start=20000, overlap=20000, preutter=20500, fixed_end=21000, end=39000),
                         metrics=Metrics(stable_vowel_ms=stable, f0_median_hz=pitch))
    write_jsonl(project / "candidates.jsonl", [onset, donor("wrong", 440, 400), donor("right", 220, 200)])
    config = load(project)[1]
    result = repair_project(project, config, mode="cv-splice", experimental=True, alias_filter="く")
    assert result["generated"] == 1, result
    repaired = read_jsonl(project / "candidates.jsonl", Candidate)[-1]
    assert repaired.alias == "く"
    assert repaired.provenance_kind == "cv_spliced"
    assert repaired.parent_candidate_ids == ["ka", "right"]
    assert repaired.review_status == "unreviewed"
    assert set(repaired.quality_flags) == {"REPAIR_REVIEW_REQUIRED", "COARTICULATION_UNCERTAIN"}
    assert (project / "review_splices" / f"ku_{repaired.candidate_id[:8]}.wav").exists()
    repeated = repair_project(project, config, mode="cv-splice", experimental=True, alias_filter="く")
    assert repeated["generated"] == 0
    assert repeated["already_generated"] == 1
    select(project, config)
    assert "く" not in json.loads((project / "selection.json").read_text(encoding="utf-8"))["chosen"]
    experimental = build(project, config, "preserve", "auto", "draft",
                         include_unreviewed_splices=True, singer_name="example_project")
    assert experimental["cv_spliced"] == 1
    assert experimental["validation_errors"] == 0
    assert experimental["needs_review"] >= 1
    bank = project / "exports" / experimental["build_id"] / "voicebank"
    assert (bank / "ku.wav").exists()
    assert (bank / "character.txt").read_text(encoding="cp932").strip() == "name=example_project"
    edit_candidate(project, repaired.candidate_id, review_status="accepted")
    select(project, config)
    assert json.loads((project / "selection.json").read_text(encoding="utf-8"))["chosen"]["く"] == repaired.candidate_id
