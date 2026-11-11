import csv
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from utau_forge.config import write_config
from utau_forge.models import Candidate, Metrics, SampleSpan, SourceAsset
from utau_forge.project import initialize, load
from utau_forge.slices import export_slices
from utau_forge.storage import sha256_file, write_jsonl


@pytest.fixture
def source_project(tmp_path):
    directory=tmp_path/"project"
    initialize(directory)
    path=directory/"audio/canonical/source.wav"
    audio=np.linspace(-.5,.5,44100,dtype=np.float32)
    sf.write(path,audio,44100,subtype="PCM_16")
    source=SourceAsset(source_id="source",original_path="source.wav",canonical_path="audio/canonical/source.wav",sha256=sha256_file(path),original_sample_rate=44100,original_channels=1,original_duration_s=1,canonical_samples=44100,speaker_id="speaker",language_id="ja")
    write_jsonl(directory/"sources/manifest.jsonl",[source])
    return directory,path


def candidate(identifier,alias,key,role,start=500,end=5000,score=.9,provenance="original"):
    return Candidate(candidate_id=identifier,speaker_id="speaker",source_id="source",utterance_id="utt",alignment_revision="r",source_language="ja",unit_role=role,unit_key=key,alias=alias,phones=key.split("+"),raw_phones=[],span=SampleSpan(start=start,end=end),quality_score=score,provenance_kind=provenance)


@pytest.mark.parametrize("profile_id",["ja-cv","ja-cvvc"])
@pytest.mark.parametrize("all_candidates",[False,True])
def test_mixed_original_slices_ignore_active_bank_inventory(source_project,profile_id,all_candidates):
    directory,path=source_project
    config=load(directory)[1]; config.export_profiles=[profile_id]; write_config(directory/"config.yaml",config)
    # CV な and VV n あ share n+a but must have different filenames.
    items=[candidate("cv","な","n+a","sustain_vowel"),candidate("vv","n あ","n+a","VV"),candidate("vc","a n","a+n","VC"),candidate("vc-low","a n","a+n","VC",start=6000,end=10000,score=.1),candidate("derived","a n","a+n","VC",provenance="cv_spliced")]
    write_jsonl(directory/"candidates.jsonl",items)
    before=sha256_file(path)
    events=[]
    result=export_slices(directory,all_candidates,top_k=1,padding_ms=10,emit=lambda **event:events.append(event))
    root=Path(result["path"])
    manifest=json.loads((root/"manifest.json").read_text(encoding="utf-8"))["slices"]
    names={row["filename"] for row in manifest}
    assert names==({"na_001.wav","vv_n_a_001.wav","vc_a_n_001.wav","vc_a_n_002.wav"} if all_candidates else {"na_001.wav","vv_n_a_001.wav","vc_a_n_001.wav"})
    assert "derived" not in {row["candidate_id"] for row in manifest}
    source=sf.read(path,dtype="float32")[0]
    for row in manifest:
        audio,rate=sf.read(root/row["filename"],dtype="float32")
        assert rate==44100 and sf.info(root/row["filename"]).subtype=="PCM_16"
        np.testing.assert_array_equal(audio,source[row["source_start"]:row["source_end"]])
        assert row["unit_start_in_file"]==row["padding_before_samples"]==441
        assert row["unit_end_in_file"]==len(audio)-441
    assert sha256_file(path)==before
    assert events[-1]["current"]==events[-1]["total"]==result["slices"]
    with (root/"sources.csv").open(encoding="utf-8-sig",newline="") as stream:
        assert len(list(csv.DictReader(stream)))==result["slices"]


def test_breath_and_custom_slices_use_safe_user_stems(source_project):
    directory,path=source_project
    items=[candidate("breath","br","br","breath"),candidate("custom","internal-custom","internal-custom","custom"),candidate("legacy","笑声","笑声","custom")]
    write_jsonl(directory/"candidates.jsonl",items)
    result=export_slices(directory,custom_units=[{"alias":"internal-custom","label":"笑声","filename_stem":"xiaosheng"}])
    root=Path(result["path"])
    names={p.name for p in root.glob("*.wav")}
    assert "br_001.wav" in names and "xiaosheng_001.wav" in names
    assert any(name.startswith("custom_") for name in names)
    assert all(name.isascii() for name in names)


def test_unknown_unit_fails_before_publishing_partial_export(source_project):
    directory,path=source_project
    write_jsonl(directory/"candidates.jsonl",[candidate("valid","あ","a","sustain_vowel"),candidate("unknown","???","invalid","VC")])
    with pytest.raises(ValueError,match="没有可用的切片命名定义"):
        export_slices(directory)
    assert not list((directory/"exports").iterdir())


@pytest.mark.parametrize("top_k,expected", [(1,["chosen"]),(3,["chosen","stable-long","stable-short"]),(10,["chosen","stable-long","stable-short","unaligned-long"])])
def test_best_x_keeps_formal_choice_then_prefers_longest_stable_vowel(source_project, top_k, expected):
    directory,_=source_project
    items=[candidate("chosen","あ","a","sustain_vowel",end=1000,score=.1),
           candidate("stable-short","あ","a","sustain_vowel",end=20000,score=1),
           candidate("stable-long","あ","a","sustain_vowel",end=40000,score=.2),
           candidate("unaligned-long","あ","a","sustain_vowel",end=44000,score=1)]
    items[1].metrics=Metrics(stable_vowel_ms=200)
    items[2].metrics=Metrics(stable_vowel_ms=600)
    write_jsonl(directory/"candidates.jsonl",items)
    result=export_slices(directory,top_k=top_k,selected_candidate_ids=["chosen"])
    manifest=json.loads((Path(result["path"])/"manifest.json").read_text(encoding="utf-8"))["slices"]
    assert [row["candidate_id"] for row in manifest]==expected
    assert manifest[0]["filename"]=="a_001.wav" and manifest[0]["selected"]
    assert all(not row["selected"] for row in manifest[1:])


def test_no_stable_vowel_orders_by_slice_length_not_quality(source_project):
    directory,_=source_project
    items=[candidate("short","あ","a","sustain_vowel",end=1000,score=1),
           candidate("long","あ","a","sustain_vowel",end=40000,score=.1),
           candidate("middle","あ","a","sustain_vowel",end=20000,score=.5)]
    items[0].metrics=Metrics(stable_vowel_ms=900)
    items[0].quality_flags=["NO_SUSTAIN_REGION"]
    write_jsonl(directory/"candidates.jsonl",items)
    result=export_slices(directory,top_k=2)
    rows=json.loads((Path(result["path"])/"manifest.json").read_text(encoding="utf-8"))["slices"]
    assert [row["candidate_id"] for row in rows]==["long","middle"]
    assert all(row["ranking_basis"]=="slice_duration_ms" for row in rows)


def test_formally_selected_splice_exports_its_own_audio(source_project):
    directory,_=source_project
    path=directory/"derived/selected.wav"
    path.parent.mkdir(exist_ok=True)
    sf.write(path,np.full(10000,.2,dtype="float32"),44100,subtype="PCM_16")
    chosen=candidate("chosen","あ","a","sustain_vowel",start=0,end=10000,provenance="cv_spliced")
    chosen.phonetic_context["derived_audio_path"]="derived/selected.wav"
    write_jsonl(directory/"candidates.jsonl",[chosen,candidate("original","あ","a","sustain_vowel")])
    result=export_slices(directory,top_k=2,selected_candidate_ids=["chosen"])
    root=Path(result["path"])
    rows=json.loads((root/"manifest.json").read_text(encoding="utf-8"))["slices"]
    assert [row["candidate_id"] for row in rows]==["chosen","original"]
    np.testing.assert_array_equal(sf.read(root/"a_001.wav")[0],sf.read(path)[0])
