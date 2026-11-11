"""Missing sustain evidence can be explicitly overridden without inventing it."""
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
import pytest

from utau_forge.analyze import _longest_sustain
from utau_forge.config import Config
from utau_forge.desktop_service import audition_file, force_automatic_timing, force_missing_sustain_timings, preview_file, preview_interval
from utau_forge.desktop_store import ProjectStore
from utau_forge.models import Candidate, Landmarks, SampleSpan, SourceAsset, Utterance
from utau_forge.storage import sha256_file


def fixture(tmp_path, monkeypatch):
    monkeypatch.setenv("UFORGE_DATA_DIR", str(tmp_path / "userdata"))
    store = ProjectStore.create(tmp_path / "project", "ForceOto")
    bank_id = store.banks()[0]["id"]
    path = store.directory / "audio/canonical/source.wav"
    sf.write(path, np.zeros(44100, dtype="float32"), 44100)
    asset = SourceAsset(source_id="source", original_path=str(path), sha256=sha256_file(path),
                        original_sample_rate=44100, original_channels=1, original_duration_s=1,
                        canonical_path="audio/canonical/source.wav", canonical_samples=44100,
                        speaker_id=store.meta("speaker_id"), language_id="ja")
    candidate = Candidate(candidate_id="missing", source_id="source", speaker_id=asset.speaker_id,
                          utterance_id="utt", alignment_revision="r", source_language="ja",
                          unit_role="sustain_vowel", unit_key="k+a", alias="か", phones=["k", "a"],
                          raw_phones=["k", "a"], span=SampleSpan(start=11025, end=22050),
                          quality_flags=["NO_SUSTAIN_REGION"],
                          phonetic_context={"vowel_span": {"start": 13230, "end": 22050}})
    store.ingest("assets", [asset])
    store.ingest("candidates", [candidate])
    return store, bank_id, candidate


def test_no_oto_plays_entire_cached_slice_then_uses_draft_oto(tmp_path, monkeypatch):
    store, bank_id, candidate = fixture(tmp_path, monkeypatch)
    preview = preview_file(store, candidate.candidate_id)
    start, end = preview_interval(candidate, preview)
    assert start == 0 and end == sf.info(preview["path"]).frames
    audio = audition_file(store, Path(preview["path"]), start, end)
    assert sf.info(audio).frames == end
    marks = Landmarks(start=12000, overlap=12000, preutter=13230, fixed_end=15000, end=21000)
    store.edit_timing(bank_id, candidate.candidate_id, marks, [])
    assert preview_interval(store.candidate(candidate.candidate_id, bank_id), preview) == (
        marks.start - preview["start"], marks.end - preview["start"])
    assert store.history(bank_id)
    assert preview_interval(store.candidate(candidate.candidate_id, bank_id), preview) == (start, end)


def test_forced_silence_has_editable_boundaries_without_stability_claim(tmp_path, monkeypatch):
    store, bank_id, candidate = fixture(tmp_path, monkeypatch)
    preview = preview_file(store, candidate.candidate_id)
    marks = force_automatic_timing(store, candidate, preview)
    marks.validate_cv(44100)
    assert marks.start == candidate.span.start and marks.preutter == 13230 and marks.end == 22050
    assert store.candidate(candidate.candidate_id, bank_id).landmarks is None
    store.edit_timing(bank_id, candidate.candidate_id, marks, [])
    restored = ProjectStore(store.directory).candidate(candidate.candidate_id, bank_id)
    assert restored.landmarks == marks and restored.automatic_landmarks is None
    assert restored.metrics == candidate.metrics and restored.quality_flags == candidate.quality_flags
    assert restored.review_status == "unreviewed" and store.selections(bank_id) == {}
    assert store.history(bank_id) and store.candidate(candidate.candidate_id, bank_id).landmarks is None
    assert store.history(bank_id, redo=True) and store.candidate(candidate.candidate_id, bank_id).landmarks == marks


def test_force_uses_existing_automatic_proposal_without_audio_analysis(tmp_path, monkeypatch):
    store, _, candidate = fixture(tmp_path, monkeypatch)
    automatic = Landmarks(start=11025, overlap=11025, preutter=13230, fixed_end=15435, end=22050)
    candidate.automatic_landmarks = automatic
    with patch("utau_forge.analyze._features", side_effect=AssertionError("Unnecessary acoustic analysis")):
        assert force_automatic_timing(store, candidate, {}) == automatic


def test_force_relaxes_minimum_sustain_length_but_normal_analysis_remains_conservative():
    count = 10
    samples = count * 220
    span = SampleSpan(start=0, end=samples)
    candidate = Candidate(source_id="source", speaker_id="speaker", utterance_id="utt", alignment_revision="r",
                          source_language="ja", unit_role="sustain_vowel", unit_key="a", alias="あ",
                          phones=["a"], raw_phones=["a"], span=span,
                          phonetic_context={"vowel_span": span.model_dump()})
    utterance = Utterance(source_id="source", speaker_id="speaker", span=span, vad_span=span,
                          corpus_offset_samples=0, language_id="ja")
    features = {"centers": np.arange(count) * 220, "f0": np.full(count, 220.),
                "rms_db": np.full(count, -20.), "periodicity": np.ones(count), "spectral_change": np.zeros(count)}
    assert _longest_sustain(candidate, features, utterance, Config())[0] is None
    marks, metrics = _longest_sustain(candidate, features, utterance, Config(), force=True)
    assert marks is not None and 0 < metrics["stable_vowel_ms"] < Config().sustain.min_ms


def test_batch_force_repairs_all_missing_sustain_aliases_as_one_undo_step(tmp_path, monkeypatch):
    store, bank_id, missing=fixture(tmp_path,monkeypatch)
    short=missing.model_copy(update={"candidate_id":"short","span":SampleSpan(start=13230,end=20000),"quality_score":1})
    vowel=missing.model_copy(update={"candidate_id":"vowel","alias":"あ","unit_key":"a","phones":["a"]})
    marks=Landmarks(start=11025,overlap=11025,preutter=13230,fixed_end=15000,end=22050)
    selected=missing.model_copy(update={"candidate_id":"selected","alias":"し","unit_key":"sh+i","landmarks":marks,"quality_flags":[]})
    ignored=[missing.model_copy(update={"candidate_id":"no-reason","alias":"す","quality_flags":[]}),
             missing.model_copy(update={"candidate_id":"rejected","alias":"く","review_status":"rejected"}),
             missing.model_copy(update={"candidate_id":"bad","alias":"け","quality_flags":["NO_SUSTAIN_REGION","STALE_ALIGNMENT"]})]
    store.ingest("candidates",[short,vowel,selected,*ignored])
    store.choose(bank_id,selected.candidate_id)
    generation=store.bank(bank_id)["generation"]
    revision=store.revision
    events=[]
    result=force_missing_sustain_timings(store,bank_id,lambda **event:events.append(event))
    assert result["eligible"]==2 and result["failures"]==[]
    assert {p["candidate_id"] for p in result["proposals"]}=={"missing","vowel"}
    assert store.candidate("missing",bank_id).landmarks is None and store.revision==revision
    assert events[-1]["current"]==events[-1]["total"]==2
    assert store.apply_forced_missing(bank_id,result["proposals"],generation)==2
    assert store.revision==revision+1 and set(store.selections(bank_id))=={"か","あ","し"}
    assert store.candidate("short",bank_id).landmarks is None
    assert store.candidate("missing",bank_id).quality_flags==missing.quality_flags
    assert store.candidate("missing",bank_id).review_status=="unreviewed"
    reopened=ProjectStore(store.directory)
    assert reopened.history(bank_id)
    assert set(reopened.selections(bank_id))=={"し"}
    assert reopened.candidate("missing",bank_id).landmarks is None and reopened.candidate("vowel",bank_id).landmarks is None
    assert reopened.history(bank_id,redo=True) and set(reopened.selections(bank_id))=={"か","あ","し"}
    assert force_missing_sustain_timings(reopened,bank_id)["proposals"]==[]


def test_batch_force_rejects_stale_result_without_publishing_any_edit(tmp_path, monkeypatch):
    store,bank_id,_=fixture(tmp_path,monkeypatch)
    generation=store.bank(bank_id)["generation"]
    result=force_missing_sustain_timings(store,bank_id)
    store.update_bank(bank_id,"Renamed",store.bank(bank_id)["preset"])
    with pytest.raises(ValueError,match="工程发生修改"):
        store.apply_forced_missing(bank_id,result["proposals"],generation)
    assert store.selections(bank_id)=={} and store.candidate("missing",bank_id).landmarks is None


def test_batch_force_empty_does_not_consume_undo_history(tmp_path, monkeypatch):
    store,bank_id,missing=fixture(tmp_path,monkeypatch)
    revision=store.revision
    assert store.apply_forced_missing(bank_id,[],store.bank(bank_id)["generation"])==0
    assert store.revision==revision


def test_batch_force_bad_second_proposal_keeps_first_unmodified(tmp_path, monkeypatch):
    store,bank_id,missing=fixture(tmp_path,monkeypatch)
    second=missing.model_copy(update={"candidate_id":"second","alias":"あ","unit_key":"a"})
    store.ingest("candidates",[second])
    result=force_missing_sustain_timings(store,bank_id)
    result["proposals"][1]["marks"]["end"]=500000
    with pytest.raises(ValueError):
        store.apply_forced_missing(bank_id,result["proposals"],store.bank(bank_id)["generation"])
    assert store.candidate("missing",bank_id).landmarks is None and store.selections(bank_id)=={}
