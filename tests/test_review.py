import json

import numpy as np
import pytest
import soundfile as sf

from utau_forge.models import Candidate, Landmarks, SampleSpan, SourceAsset
from utau_forge.project import initialize, load
from utau_forge.build import select
from utau_forge.review import choose_candidate, edit_candidate, reset_landmark, undo_last_edit
from utau_forge.storage import read_jsonl, write_jsonl


def test_edit_lock_reset_and_undo_are_persistent(tmp_path):
    project = tmp_path / "p"
    initialize(project)
    sf.write(project / "audio" / "canonical" / "s.wav", np.zeros(44100), 44100)
    source = SourceAsset(source_id="s", original_path="s.wav", sha256="x", original_sample_rate=44100, original_channels=1, original_duration_s=1, canonical_path="audio/canonical/s.wav", canonical_samples=44100, speaker_id="speaker_001", language_id="ja")
    write_jsonl(project / "sources" / "manifest.jsonl", [source])
    original = Landmarks(start=0, overlap=0, preutter=1000, fixed_end=2000, end=30000)
    candidate = Candidate(candidate_id="c", speaker_id="speaker_001", source_id="s", utterance_id="u", alignment_revision="r", source_language="ja", unit_role="sustain_vowel", unit_key="a", alias="あ", phones=["a"], raw_phones=["a"], span=SampleSpan(start=0, end=30000), landmarks=original, automatic_landmarks=original)
    write_jsonl(project / "candidates.jsonl", [candidate])
    edit_candidate(project, "c", landmarks=original.model_copy(update={"preutter": 1200}), locked_fields=["preutter"])
    with pytest.raises(ValueError, match="locked"):
        edit_candidate(project, "c", landmarks=original.model_copy(update={"preutter": 1300}))
    reset_landmark(project, "c", "preutter")
    current = read_jsonl(project / "candidates.jsonl", Candidate)[0]
    assert current.landmarks.preutter == 1000
    assert current.locked_fields == []
    assert undo_last_edit(project) == "edit"
    current = read_jsonl(project / "candidates.jsonl", Candidate)[0]
    assert current.landmarks.preutter == 1200
    assert current.locked_fields == ["preutter"]


def test_manual_selection_survives_automatic_rescoring(tmp_path):
    project = tmp_path / "p"
    initialize(project)
    marks = Landmarks(start=0, overlap=0, preutter=1000, fixed_end=2000, end=30000)
    candidates = [Candidate(candidate_id=identifier, speaker_id="speaker_001", source_id="s", utterance_id="u", alignment_revision="r", source_language="ja", unit_role="sustain_vowel", unit_key="a", alias="あ", phones=["a"], raw_phones=["a"], span=SampleSpan(start=0, end=30000), landmarks=marks, quality_score=score) for identifier, score in (("best", 0.9), ("manual", 0.4))]
    write_jsonl(project / "candidates.jsonl", candidates)
    config = load(project)[1]
    select(project, config)
    choose_candidate(project, "manual")
    select(project, config)
    selection = json.loads((project / "selection.json").read_text(encoding="utf-8"))
    assert selection["chosen"]["あ"] == "manual"
    assert undo_last_edit(project) == "choose"
    select(project, config)
    selection = json.loads((project / "selection.json").read_text(encoding="utf-8"))
    assert selection["chosen"]["あ"] == "best"
