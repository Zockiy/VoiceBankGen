from pathlib import Path
from unittest.mock import patch
import io
import json

import numpy as np
import pytest
import soundfile as sf

from utau_forge.bank_aliases import export_aliases
from utau_forge.breaths import breath_candidates
from utau_forge.desktop_import import recover_bank
from utau_forge.desktop_adopt import adopt_external
from utau_forge.desktop_service import compile_bank
from utau_forge.desktop_store import ProjectStore
from utau_forge.desktop_sync import bind, sync_installation
from utau_forge.models import Candidate, Landmarks, Phone, SampleSpan, SourceAsset, Utterance
from utau_forge.storage import sha256_file
from utau_forge.oto import compile_oto


@pytest.fixture
def bank_project(tmp_path, monkeypatch):
    monkeypatch.setenv("UFORGE_DATA_DIR", str(tmp_path / "userdata"))
    store = ProjectStore.create(tmp_path / "project", "Options")
    path = store.directory / "audio/canonical/source.wav"
    audio = .1 * np.sin(2*np.pi*220*np.arange(44100)/44100)
    audio[25000:30000] = np.random.default_rng(1).normal(0,.025,5000)
    sf.write(path, audio, 44100, subtype="PCM_16")
    asset = SourceAsset(source_id="source", original_path="audio/canonical/source.wav", sha256=sha256_file(path), original_sample_rate=44100,
                        original_channels=1, original_duration_s=1, canonical_path="audio/canonical/source.wav", canonical_samples=44100,
                        speaker_id=store.meta("speaker_id"), language_id="ja")
    span = SampleSpan(start=0,end=44100)
    utt = Utterance(utterance_id="utt", source_id=asset.source_id, speaker_id=asset.speaker_id, span=span, vad_span=span, corpus_offset_samples=0, language_id="ja")
    store.ingest("assets", [asset])
    store.ingest("utterances", [utt])
    candidates = []
    for index, start in enumerate((1000,12000)):
        marks = Landmarks(start=start, overlap=start, preutter=start+500,fixed_end=start+1000,end=start+8000)
        candidates.append(Candidate(candidate_id=f"vowel-{index}", speaker_id=asset.speaker_id,source_id=asset.source_id,utterance_id=utt.utterance_id,
                                    alignment_revision="r",source_language="ja",unit_role="sustain_vowel",unit_key="a",alias="あ",phones=["a"],raw_phones=["ja/a"],
                                    span=SampleSpan(start=start,end=start+8000),landmarks=marks,automatic_landmarks=marks,quality_score=.9-index*.1))
    store.ingest("candidates",candidates)
    bank = store.banks()[0]
    store.update_bank(bank["id"],bank["name"],{**bank["preset"],"alias_format":"romaji"})
    store.choose(bank["id"], candidates[0].candidate_id)
    return store, bank["id"], candidates, utt


def build(store, bank_id):
    return compile_bank(store,bank_id,store.bank(bank_id)["generation"])


@pytest.mark.parametrize("timing_flag", ["NO_SUSTAIN_REGION", "DEVOICED_VOWEL"])
def test_manual_oto_without_sustain_can_sync_and_reuses_wav(bank_project, tmp_path, timing_flag):
    store, bank_id, candidates, _ = bank_project
    original = candidates[0]
    missing = original.model_copy(update={"landmarks": None, "automatic_landmarks": None,
                                          "quality_flags": [timing_flag]})
    store.ingest("candidates", [missing])
    store.edit_timing(bank_id, missing.candidate_id, original.landmarks, [])
    store = ProjectStore(store.directory)
    # Simulate the previous version rejecting an already saved manual oto.
    # After upgrading, retry must rebuild omitted selections without another edit.
    with patch.object(store, "manual_timing_ids", return_value=set()):
        failed = build(store, bank_id)
    assert failed["failures"] and not failed["manifest"]["entries"]
    first = build(store, bank_id)
    assert first["failures"] == []
    assert len(first["manifest"]["entries"]) == 1
    installation = bind(store, bank_id, tmp_path / "installed")
    sync_installation(store, installation)
    before = (tmp_path / "installed/oto.ini").read_bytes()
    store.edit_timing(bank_id, missing.candidate_id,
                      original.landmarks.model_copy(update={"preutter": original.landmarks.preutter + 100}), [])
    assert (tmp_path / "installed/oto.ini").read_bytes() == before
    with patch("utau_forge.desktop_service.render_candidate", side_effect=AssertionError("unexpected DSP")):
        second = build(store, bank_id)
    assert second["rendered"] == 0 and second["updated_timings"] == 1
    assert sync_installation(store, installation)["copied_wavs"] == 0
    assert (tmp_path / "installed/oto.ini").read_bytes() != before
    persisted = store.candidate(missing.candidate_id, bank_id)
    assert persisted.quality_flags == [timing_flag]
    assert persisted.automatic_landmarks is None and persisted.review_status == "unreviewed"
    assert persisted.metrics == original.metrics


def test_no_sustain_requires_manual_oto_and_undo_removes_permission(bank_project):
    store, bank_id, candidates, _ = bank_project
    original = candidates[0]
    # Even non-null suggested landmarks alone do not count as a manual edit.
    missing = original.model_copy(update={"automatic_landmarks": None,
                                          "quality_flags": ["NO_SUSTAIN_REGION"]})
    store.ingest("candidates", [missing])
    assert "未找到稳定元音区" in build(store, bank_id)["failures"][0]["reason"]
    missing.landmarks = None
    store.ingest("candidates", [missing])
    store.edit_timing(bank_id, missing.candidate_id, original.landmarks, [])
    assert build(store, bank_id)["failures"] == []
    assert store.history(bank_id)
    assert store.candidate(missing.candidate_id, bank_id).landmarks is None
    assert missing.candidate_id not in store.manual_timing_ids(bank_id)
    assert build(store, bank_id)["failures"]
    assert store.history(bank_id, True)
    assert build(store, bank_id)["failures"] == []


@pytest.mark.parametrize("timing_flag", ["NO_SUSTAIN_REGION", "DEVOICED_VOWEL"])
def test_manual_no_sustain_strict_requires_listening_acceptance(bank_project, timing_flag):
    store, bank_id, candidates, _ = bank_project
    original = candidates[0]
    missing = original.model_copy(update={"landmarks": None, "automatic_landmarks": None,
                                          "quality_flags": [timing_flag]})
    store.ingest("candidates", [missing])
    store.edit_timing(bank_id, missing.candidate_id, original.landmarks, [])
    bank = store.bank(bank_id)
    store.update_bank(bank_id, bank["name"], {**bank["preset"], "release_mode": "strict"})
    assert "试听接受候选" in build(store, bank_id)["failures"][0]["reason"]
    store.review(bank_id, missing.candidate_id, "accepted")
    assert build(store, bank_id)["failures"] == []


def test_manual_oto_only_waives_missing_sustain_flag(bank_project):
    store, bank_id, candidates, _ = bank_project
    original = candidates[0]
    missing = original.model_copy(update={"landmarks": None, "automatic_landmarks": None,
                                          "quality_flags": ["NO_SUSTAIN_REGION", "ALIGNMENT_STALE"]})
    store.ingest("candidates", [missing])
    store.edit_timing(bank_id, missing.candidate_id, original.landmarks, [])
    result = build(store, bank_id)
    assert not result["manifest"]["entries"]
    assert "未找到稳定元音区" not in result["failures"][0]["reason"]
    assert result["failures"]


def test_unordered_manual_oto_sync_recovery_and_undo(bank_project,tmp_path):
    store,bank_id,candidates,_=bank_project
    original=candidates[0]
    marks=original.landmarks.model_copy(update={"overlap":6500,"preutter":3000,"fixed_end":1500})
    store.edit_timing(bank_id,original.candidate_id,marks,[])
    assert store.candidate(original.candidate_id,bank_id).landmarks==marks
    result=build(store,bank_id)
    assert result["failures"]==[]
    installation=bind(store,bank_id,tmp_path/"installed")
    sync_installation(store,installation)
    recovered=recover_bank(tmp_path/"installed",tmp_path/"recovered","Recovered")
    other=ProjectStore(Path(recovered["project"]))
    imported=other.candidates(bank_id=recovered["bank_id"],selected_only=True)[0]
    assert imported.landmarks.overlap>imported.landmarks.preutter>imported.landmarks.fixed_end
    assert build(other,recovered["bank_id"])["failures"]==[]
    assert store.history(bank_id)
    assert store.candidate(original.candidate_id,bank_id).landmarks==original.landmarks
    assert store.history(bank_id,True)
    assert store.candidate(original.candidate_id,bank_id).landmarks==marks


def test_manual_fixed_end_at_end_does_not_require_empty_tail(bank_project):
    store,bank_id,candidates,_=bank_project
    original=candidates[0]
    store.ingest("candidates",[original.model_copy(update={"landmarks":None,"automatic_landmarks":None,"quality_flags":["NO_SUSTAIN_REGION"]})])
    store.edit_timing(bank_id,original.candidate_id,original.landmarks.model_copy(update={"fixed_end":original.landmarks.end}),[])
    result=build(store,bank_id)
    assert result["failures"]==[] and result["exported"]==1


def test_breath_labels_do_not_classify_silence_or_ep(bank_project):
    store,bank_id,candidates,utt = bank_project
    phones = [Phone(phone_id=label,utterance_id=utt.utterance_id,source_id=utt.source_id,language_id="ja",raw_model_label=label,phoneme_id="ja@2:sp",
                    span=SampleSpan(start=25000,end=30000),alignment_revision="r",features={"backend":"tifa"}) for label in ("AP","br","SP","EP","sil","pau")]
    assert [c.raw_phones for c in breath_candidates(phones,{utt.utterance_id:utt})] == [["AP"],["br"]]
    assert not breath_candidates([phones[0].model_copy(update={"features":{"backend":"mfa"}})],{utt.utterance_id:utt})
    store.ingest("alignments",phones)
    assert "br" not in dict(store.targets(bank_id))
    bank = store.bank(bank_id)
    store.update_bank(bank_id,bank["name"],{**bank["preset"],"include_breaths":True,"pitch_mode":"flatten","target_note":"A3"})
    store.choose(bank_id,store.candidates("br")[0].candidate_id)
    # No WORLD or vowel normalization is allowed for breath-only banks.
    with store.connection() as connection:
        connection.execute("DELETE FROM bank_selections WHERE alias='あ'")
    store.add_variant(bank_id,store.candidates("br")[1].candidate_id)
    with patch("utau_forge.desktop_service._flatten",side_effect=AssertionError("breath pitch")), patch("utau_forge.desktop_service._level",side_effect=AssertionError("breath level")):
        result = build(store,bank_id)
    assert result["failures"] == []
    entries = result["manifest"]["entries"]
    assert {e["alias"] for e in entries} == {"br","br_1"}
    for entry in entries:
        audio,_ = sf.read(Path(result["path"])/entry["filename"],dtype="int16")
        original,_ = sf.read(store.directory/"audio/canonical/source.wav",dtype="int16")
        start = entry["marks"]["start"]
        assert np.array_equal(audio[start:start+5000],original[25000:30000])
        assert entry["marks"]["fixed_end"] == entry["marks"]["end"]-1
    store.update_bank(bank_id,bank["name"],{**store.bank(bank_id)["preset"],"include_breaths":False})
    assert not build(store,bank_id)["manifest"]["entries"]
    assert {"br","br_1"} <= store.selections(bank_id).keys()


def test_variants_persist_recommend_undo_and_change_only_selected_alias(bank_project):
    store,bank_id,candidates,utt = bank_project
    first = store.add_variant(bank_id,candidates[1].candidate_id)
    second = store.add_variant(bank_id,candidates[0].candidate_id)
    assert (first,second) == ("あ_1","あ_2")
    store = ProjectStore(store.directory)
    store.recommend(bank_id)
    assert [c.candidate_id for c in store.candidates(first,bank_id)] == [c.candidate_id for c in candidates]
    store.choose(bank_id,candidates[1].candidate_id,second)
    assert store.selections(bank_id)["あ"]["candidate_id"] == candidates[0].candidate_id
    store.remove_variant(bank_id,first)
    assert first not in dict(store.targets(bank_id))
    assert store.history(bank_id) and first in dict(store.targets(bank_id))
    assert store.history(bank_id,True) and first not in store.selections(bank_id)
    with pytest.raises(ValueError):
        store.remove_variant(bank_id,"あ")


def test_notation_and_variants_reuse_wav_and_sync_alias_roster(bank_project,tmp_path):
    store,bank_id,candidates,utt = bank_project
    initial = build(store,bank_id)
    assert [e["alias"] for e in initial["manifest"]["entries"]] == ["a"]
    installation = bind(store,bank_id,tmp_path/"Singers/Options")
    sync_installation(store,installation)
    oto = tmp_path/"Singers/Options/oto.ini"
    with oto.open("ab") as stream:
        stream.write(b"external.wav=external,0,0,0,0,0\r\n")
    first = store.add_variant(bank_id,candidates[0].candidate_id)
    store.add_variant(bank_id,candidates[1].candidate_id)
    result = build(store,bank_id)
    assert result["rendered"] == 1 and result["unchanged"] == 1
    assert {e["alias"] for e in result["manifest"]["entries"]} == {"a","a_1","a_2"}
    assert all(e["filename"].startswith(e["alias"]+"_") for e in result["manifest"]["entries"])
    sync_installation(store,installation)
    bank = store.bank(bank_id)
    store.update_bank(bank_id,bank["name"],{**bank["preset"],"alias_format":"hiragana"})
    store.remove_variant(bank_id,first)
    with patch("utau_forge.desktop_service.render_candidate",side_effect=AssertionError("unexpected DSP")):
        result = build(store,bank_id)
    assert result["rendered"] == 0 and result["missing"] == 100
    assert {e["alias"] for e in result["manifest"]["entries"]} == {"あ","あ_2"}
    synced = sync_installation(store,installation)
    assert synced["copied_wavs"] == 0
    lines = oto.read_bytes().decode("cp932")
    assert "=a," not in lines and "=a_1," not in lines and "=a_2," not in lines
    assert "=あ," in lines and "=あ_2," in lines and "external.wav=external" in lines
    assert export_aliases("ja-cvvc","a い_2","romaji") == ["a i_2"]
    assert export_aliases("ja-cvvc","a い_2","hiragana") == ["a い_2"]


def test_recover_romaji_numbered_and_breath_entries(bank_project,tmp_path):
    store,bank_id,candidates,utt = bank_project
    store.add_variant(bank_id,candidates[1].candidate_id)
    result=build(store,bank_id)
    root=Path(result["path"])
    with (root/"oto.ini").open("ab") as stream:
        filename=result["manifest"]["entries"][0]["filename"]
        stream.write(f"{filename}=br,0,10,0,0,0\r\n".encode("cp932"))
    recovered=recover_bank(root,tmp_path/"recovery","Recovered")
    other=ProjectStore(Path(recovered["project"]))
    assert {"あ","あ_1","br"} <= other.selections(recovered["bank_id"]).keys()
    assert other.bank(recovered["bank_id"])["preset"]["include_breaths"]


def test_external_variant_timing_isolated_from_primary(bank_project,tmp_path):
    store,bank_id,candidates,utt=bank_project
    variant=store.add_variant(bank_id,candidates[0].candidate_id)
    result=build(store,bank_id)
    identifier=bind(store,bank_id,tmp_path/"installed")
    sync_installation(store,identifier)
    entry=next(e for e in result["manifest"]["entries"] if e["alias"]=="a_1")
    marks=Landmarks.model_validate(entry["marks"]).model_copy(update={"preutter":entry["marks"]["preutter"]+10})
    changed=compile_oto(entry["filename"],entry["alias"],marks,entry["samples"])
    oto=tmp_path/"installed/oto.ini"
    oto.write_bytes(oto.read_bytes().replace(entry["line"].encode("cp932"),changed.encode("cp932")))
    before=store.candidate(candidates[0].candidate_id,bank_id).landmarks
    adopt_external(store,identifier,store.bank(bank_id)["generation"],lambda **event:None)
    selection=store.selections(bank_id)
    assert selection["あ"]["candidate_id"]==candidates[0].candidate_id
    assert selection[variant]["candidate_id"]!=candidates[0].candidate_id
    assert store.candidate(candidates[0].candidate_id,bank_id).landmarks==before
    build(store,bank_id)
    sync_installation(store,identifier)


def test_worker_protocol_uses_utf8_even_with_ansi_console(bank_project,tmp_path):
    from utau_forge.desktop_worker import main
    store,bank_id,candidates,utt=bank_project
    request=tmp_path/"request.json"
    request.write_text(json.dumps({"job_id":"utf8","generation":0,"action":"未知操作","project":str(store.directory),"bank_id":bank_id}),encoding="utf-8")
    buffer=io.BytesIO()
    output=io.TextIOWrapper(buffer,encoding="cp1252")
    with patch("sys.stdout",output),patch("utau_forge.devices.prepare_dlls"):
        with pytest.raises(SystemExit):
            main(str(request))
    output.flush()
    events=[json.loads(line) for line in buffer.getvalue().decode("utf-8").splitlines()]
    assert any("stage" in item for item in events)
    event=events[-1]
    assert event["error"]=="未知任务：未知操作"
