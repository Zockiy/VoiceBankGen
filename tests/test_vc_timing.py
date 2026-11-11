import json

import numpy as np
import pytest
import soundfile as sf

from utau_forge.desktop_store import ProjectStore
from utau_forge.desktop_service import compile_bank
from utau_forge.models import Candidate, Landmarks, Phone, SampleSpan, SourceAsset, Utterance
from utau_forge.oto import parse_oto
from utau_forge.storage import sha256_file
from utau_forge.vc_timing import crossfade_marks, prepare_cvvc_version, propose_vc_optimization


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setenv("UFORGE_DATA_DIR",str(tmp_path/"userdata"))
    store=ProjectStore.create(tmp_path/"project","Crossfade")
    cv=store.banks()[0]
    path=store.directory/"audio/canonical/s.wav"
    sf.write(path,.1*np.sin(2*np.pi*220*np.arange(44100)/44100),44100)
    asset=SourceAsset(source_id="s",speaker_id=store.meta("speaker_id"),language_id="ja",original_path=str(path),
                      canonical_path="audio/canonical/s.wav",sha256=sha256_file(path),original_sample_rate=44100,
                      original_channels=1,original_duration_s=1,canonical_samples=44100)
    store.ingest("assets",[asset])
    utt=Utterance(utterance_id="u",source_id="s",speaker_id=asset.speaker_id,language_id="ja",span=SampleSpan(start=0,end=44100),vad_span=SampleSpan(start=0,end=44100),corpus_offset_samples=0)
    store.ingest("utterances",[utt])
    def item(identifier,alias,key,role,start,preutter,end):
        marks=Landmarks(start=start,overlap=start,preutter=preutter,fixed_end=end-1,end=end)
        return Candidate(candidate_id=identifier,source_id="s",speaker_id=asset.speaker_id,utterance_id="u",alignment_revision="r",source_language="ja",unit_role=role,unit_key=key,alias=alias,phones=key.split("+"),raw_phones=[],span=SampleSpan(start=start,end=end),landmarks=marks,automatic_landmarks=marks)
    ka=item("ka","か","k+a","sustain_vowel",0,4410,17640)
    ga=item("ga","が","g+a","sustain_vowel",17640,22050,33075)
    vc=item("vc","a g","a+g","VC",13230,17640,22050)
    store.ingest("candidates",[ka,ga,vc])
    store.choose(cv["id"],ka.candidate_id)
    store.choose(cv["id"],ga.candidate_id)
    return store,cv,ka,ga,vc


def test_bilateral_crossfade_preserves_alignment_and_whole_batch_undo(project):
    store,cv,ka,ga,vc=project
    bank=store.create_bank("ja-cvvc",copy_units_from=cv["id"])
    store.choose(bank["id"],vc.candidate_id)
    result=propose_vc_optimization(store,bank["id"])
    assert result["vc"]==1 and result["cv"]==1
    before=store.revision
    assert store.apply_timing_batch(bank["id"],result["changes"],store.bank(bank["id"])["generation"])==2
    assert store.revision==before+1
    for original in [ga,vc]:
        marks=store.candidate(original.candidate_id,bank["id"]).landmarks
        assert marks.overlap-marks.start==1323
        assert {f:getattr(marks,f) for f in ("start","preutter","fixed_end","end")}=={f:getattr(original.landmarks,f) for f in ("start","preutter","fixed_end","end")}
    assert store.candidate("ga",cv["id"]).landmarks==store.candidate("ga",bank["id"]).landmarks
    assert store.candidate("ka",bank["id"]).landmarks==ka.landmarks
    assert store.history(bank["id"])
    assert store.candidate("vc",bank["id"]).landmarks==vc.landmarks
    assert store.candidate("ga",bank["id"]).landmarks==ga.landmarks
    assert store.history(bank["id"],redo=True)
    assert propose_vc_optimization(store,bank["id"])["changes"]==[]


def test_locked_and_short_context_and_stale_proposals(project):
    store,cv,_,ga,vc=project
    bank=store.create_bank("ja-cvvc",copy_units_from=cv["id"])
    store.edit_timing(bank["id"],ga.candidate_id,ga.landmarks,["overlap"])
    result=propose_vc_optimization(store,bank["id"])
    assert result["locked"]==1 and result["cv"]==0
    generation=store.bank(bank["id"])["generation"]
    store.edit_timing(bank["id"],vc.candidate_id,vc.landmarks,["end"])
    with pytest.raises(ValueError,match="工程发生修改"):
        store.apply_timing_batch(bank["id"],result["changes"],generation)
    tiny=Landmarks(start=0,overlap=0,preutter=100,fixed_end=200,end=300)
    assert crossfade_marks(tiny).overlap==50
    with pytest.raises(ValueError):
        propose_vc_optimization(store,cv["id"])


@pytest.mark.parametrize("state",["editable","locked","no_preutter"])
def test_vv_uses_same_crossfade_guards_and_batch_undo(project,state):
    store,cv,_,_,vc=project
    marks=vc.landmarks
    if state=="no_preutter":
        marks=marks.model_copy(update={"preutter":marks.start})
    vv=vc.model_copy(update={"candidate_id":"vv","unit_role":"VV","unit_key":"a+i","alias":"a い","phones":["a","i"],"landmarks":marks,"automatic_landmarks":marks})
    store.ingest("candidates",[vv])
    bank=store.create_bank("ja-cvvc",copy_units_from=cv["id"])
    store.choose(bank["id"],vv.candidate_id)
    if state=="locked":
        store.edit_timing(bank["id"],vv.candidate_id,marks,["overlap"])
    result=propose_vc_optimization(store,bank["id"])
    assert result["vv"]==(1 if state=="editable" else 0)
    if state=="locked": assert result["locked"]==1
    if state=="no_preutter": assert result["insufficient"]==1
    store.apply_timing_batch(bank["id"],result["changes"],store.bank(bank["id"])["generation"])
    actual=store.candidate(vv.candidate_id,bank["id"]).landmarks
    assert actual==(crossfade_marks(marks) if state=="editable" else marks)
    assert store.history(bank["id"])
    assert store.candidate(vv.candidate_id,bank["id"]).landmarks==marks


def test_new_cvvc_inherits_choices_preset_and_manual_timings_without_models(project,monkeypatch):
    store,cv,_,ga,_=project
    store.update_bank(cv["id"],"Custom",{**cv["preset"],"alias_format":"romaji","target_note":"D3"})
    edited=ga.landmarks.model_copy(update={"fixed_end":31000})
    store.edit_timing(cv["id"],ga.candidate_id,edited,[])
    def phone(label,start,end):
        return Phone(phone_id=f"{label}{start}",language_id="ja",phoneme_id="ja@1:"+label,raw_model_label=label,span=SampleSpan(start=start,end=end),source_id="s",utterance_id="u",alignment_revision="r")
    store.ingest("alignments",[phone("a",0,17640),phone("g",17640,22050),phone("a",22050,33075),phone("i",33075,40000)])
    monkeypatch.setattr("utau_forge.desktop_engine.prepare_engine",lambda *args:pytest.fail("No models/engine snapshot should be loaded"))
    bank=store.create_bank("ja-cvvc",copy_units_from=cv["id"])
    assert bank["preset"]["target_note"]=="D3" and bank["preset"]["alias_format"]=="romaji"
    assert store.candidate("ga",bank["id"]).landmarks==edited
    result=prepare_cvvc_version(store,bank["id"],bank["generation"])
    assert result["added"]==2 and result["cv"]==0 and result["vv"]==1
    assert store.selections(bank["id"])["が"]["candidate_id"]=="ga"
    assert "a g" in store.selections(bank["id"])
    assert store.candidate("ga",cv["id"]).landmarks==edited
    assert store.candidate("ga",bank["id"]).landmarks==edited
    # Running extraction again keeps candidate payloads, reviews and user oto.
    assert prepare_cvvc_version(store,bank["id"],store.bank(bank["id"])["generation"])["added"]==0


def test_legacy_timing_fallback_is_readable_and_base_edits_are_shared(project):
    store,cv,_,ga,_=project
    legacy=ga.landmarks.model_copy(update={"overlap":18000})
    with store.connection() as connection:
        connection.execute("INSERT INTO candidate_timings(candidate_id,marks,locks,preset_hash) VALUES(?,?,?,?)",("ga",legacy.model_dump_json(),"[]",store.preset_key(cv["id"])))
    assert store.candidate("ga",cv["id"]).landmarks==legacy
    bank=store.create_bank("ja-cvvc",copy_units_from=cv["id"])
    change=legacy.model_copy(update={"overlap":19000})
    store.edit_timing(bank["id"],"ga",change,[])
    assert store.candidate("ga",cv["id"]).landmarks==change
    assert store.candidates(bank_id=bank["id"])[0] is not None
    assert "ga" in store.manual_timing_ids(bank["id"])


def test_oto_only_optimization_reuses_audio_and_exports_both_overlaps(project):
    store,cv,_,_,vc=project
    bank=store.create_bank("ja-cvvc",copy_units_from=cv["id"])
    store.choose(bank["id"],vc.candidate_id)
    first=compile_bank(store,bank["id"],store.bank(bank["id"])["generation"])
    result=propose_vc_optimization(store,bank["id"])
    store.apply_timing_batch(bank["id"],result["changes"],store.bank(bank["id"])["generation"])
    second=compile_bank(store,bank["id"],store.bank(bank["id"])["generation"])
    assert second["rendered"]==0
    root=__import__("pathlib").Path(second["path"])
    entries=[parse_oto(line) for line in (root/"oto.ini").read_text(encoding="cp932").splitlines()]
    assert next(o for o in entries if o.alias=="a g").overlap_ms==30
    assert next(o for o in entries if o.alias=="が").overlap_ms==30


def test_batch_timing_validation_does_not_reopen_sqlite_per_candidate(project,monkeypatch):
    store,cv,_,ga,_=project
    items=[ga.model_copy(update={"candidate_id":f"ga-{i}"}) for i in range(40)]
    store.ingest("candidates",items)
    changes=[{"candidate_id":c.candidate_id,"marks":crossfade_marks(c.landmarks).model_dump(),"locks":[]} for c in items]
    generation=store.bank(cv["id"])["generation"]
    connections=[]
    original=store.connection
    def connection():
        connections.append(1)
        return original()
    monkeypatch.setattr(store,"connection",connection)
    assert store.apply_timing_batch(cv["id"],changes,generation)==40
    assert len(connections)<15
