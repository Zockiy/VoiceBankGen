import json
from pathlib import Path

import numpy as np
import soundfile as sf
from zipfile import ZipFile

from utau_forge.build import build, finalize_export, select, validate_export
from utau_forge.models import Candidate, Landmarks, Metrics, SampleSpan, SourceAsset
from utau_forge.project import initialize, load
from utau_forge.review import build_draft_from_review
from utau_forge.slices import export_slices
from utau_forge.storage import write_jsonl


def test_draft_build_is_pcm16_and_reports_missing_units(tmp_path):
    project = tmp_path / "project"
    initialize(project)
    samples = 22050
    audio = 0.15 * np.sin(2 * np.pi * 220 * np.arange(samples) / 44100)
    canonical = project / "audio" / "canonical" / "source.wav"
    sf.write(canonical, audio, 44100, subtype="FLOAT")
    source = SourceAsset(source_id="s", original_path="sources/original/source.wav", sha256="test", original_sample_rate=44100, original_channels=1, original_duration_s=0.5, canonical_path="audio/canonical/source.wav", canonical_samples=samples, speaker_id="speaker_001", language_id="ja")
    write_jsonl(project / "sources" / "manifest.jsonl", [source])
    candidate = Candidate(candidate_id="c", speaker_id="speaker_001", source_id="s", utterance_id="u", alignment_revision="r", source_language="ja", unit_role="sustain_vowel", unit_key="a", alias="あ", phones=["a"], raw_phones=["a"], span=SampleSpan(start=0, end=samples), landmarks=Landmarks(start=0, overlap=0, preutter=500, fixed_end=1000, end=samples), metrics=Metrics(stable_vowel_ms=450, f0_median_hz=220), quality_score=0.8, quality_flags=[])
    write_jsonl(project / "candidates.jsonl", [candidate])
    config = load(project)[1]
    assert select(project, config)["selected"] == 1
    result = build(project, config, "preserve", "auto", "draft")
    assert result["exported"] == 1
    assert result["missing"] == 100
    assert result["validation_errors"] == 0
    root = project / "exports" / result["build_id"]
    assert sf.info(root / "voicebank" / "a.wav").subtype == "PCM_16"
    assert "a.wav=あ" in (root / "voicebank" / "oto.ini").read_bytes().decode("cp932")
    with ZipFile(root / "voicebank.zip") as archive:
        assert "voicebank/a.wav" in archive.namelist()
    assert json.loads((root / "reports" / "build_manifest.json").read_text(encoding="utf-8"))["pitch_mode"] == "preserve"
    slices = export_slices(project)
    assert (Path(slices["path"]) / "a_001.wav").is_file()
    flattened = build_draft_from_review(project, "flatten", "A3")
    flat_root = project / "exports" / flattened["build_id"]
    flat_manifest = json.loads((flat_root / "reports" / "build_manifest.json").read_text(encoding="utf-8"))
    assert flat_manifest["pitch_mode"] == "flatten"
    assert flat_manifest["target_note_requested"] == "A3"
    assert round(flat_manifest["target_hz"]) == 220
    assert (flat_root / "voicebank" / "a.wav").is_file()
    assert validate_export(root)["errors"] == 0
    assert json.loads((root / "reports" / "coverage.json").read_text(encoding="utf-8"))["original"] == 1
    write_jsonl(project / "candidates.jsonl", [candidate.model_copy(update={"candidate_id": "replacement"})])
    release = finalize_export(root)
    assert not release["strict_ready"]
    assert release["missing_candidate_ids"] == ["c"]
    assert "CANDIDATES_CHANGED_SINCE_BUILD" in release["reasons"]
    assert json.loads((root / "reports" / "release_check.json").read_text(encoding="utf-8")) == release
