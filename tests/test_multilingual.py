import json
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
from utau_forge.languages import language,profile
from utau_forge.languages.zh import transition_candidates
from utau_forge.models import Candidate,Phone,SampleSpan,Landmarks,SourceAsset,Utterance
from utau_forge.desktop_store import ProjectStore
from utau_forge.desktop_service import compile_bank
from utau_forge.desktop_sync import bind,sync_installation
from utau_forge.desktop_import import recover_bank
from utau_forge.cross_language import recipe,generate,blend
from utau_forge.storage import sha256_file


def test_mandarin_inventory_aliases_and_full_finals():
    inv=language("zh").inventory
    assert len(inv["units"])==411
    assert inv["units"]["shi"]==["sh","ir"]
    assert inv["units"]["si"]==["s","i0"]
    assert inv["units"]["shang"]==["sh","ang"]
    assert inv["units"]["lv"]==["l","v"]
    units=profile("zh-cv").required_units(inv,"core")
    assert len({profile("zh-cv").filename_stem(u) for u in units})==len(units)
    assert all(profile("zh-cv").filename_stem(u).isascii() for u in units)


def phone(code,label,start,end,source="s",utt="u"):
    return Phone(phone_id=f"{code}-{label}-{start}",language_id=code,phoneme_id=code+"@1:"+label,raw_model_label=code+"/"+label,span=SampleSpan(start=start,end=end),utterance_id=utt,source_id=source,alignment_revision="r")


def test_native_mandarin_does_not_span_gap_or_source_and_keeps_nasal_tail():
    pack=language("zh"); prof=profile("zh-cv")
    p=[phone("zh","sh",0,2205),phone("zh","ang",2205,11025)]
    c=pack.propose_units(p,prof)[0]
    assert c.alias=="shang" and c.span.end==11025
    assert all(c.alias!="shang" and c.span.start>=2205 for c in pack.propose_units([p[0],p[1].model_copy(update={"source_id":"other"})],prof))
    assert all(c.alias!="shang" and c.span.start>=3000 for c in pack.propose_units([p[0],p[1].model_copy(update={"span":SampleSpan(start=3000,end=11025)})],prof))
    u=Utterance(utterance_id="u",source_id="s",speaker_id="sp",language_id="zh",span=SampleSpan(start=0,end=22050),vad_span=SampleSpan(start=0,end=22050),corpus_offset_samples=0)
    assert not transition_candidates([p[1],phone("zh","k",11025,14000,utt="other")],{"u":u},pack.inventory)


def test_directional_rules_do_not_equate_voicing_aspiration_or_compound_rimes():
    assert recipe("zh","ja","b",False) is None
    assert recipe("zh","ja","b",True)[0]==["b"]
    assert "浊" in recipe("zh","ja","b",True)[1]
    assert recipe("ja","zh","b",True)[0]==["p"]
    assert recipe("ja","zh","p",False) is None
    assert recipe("zh","ja","a",True)[0]==["a"]
    assert recipe("ja","zh","ang",True)[0]==["a","N"]
    assert recipe("ja","zh","er",True) is None
    assert recipe("ja","ja","a",True) is None


@pytest.fixture
def source_project(tmp_path,monkeypatch):
    monkeypatch.setenv("UFORGE_DATA_DIR",str(tmp_path/"userdata"))
    store=ProjectStore.create(tmp_path/"project","日中测试",source_language="ja")
    labels=["a","i","o","u","e","N","m","n","s","p","t","k","r","ch","sh","ts","y","w","f","h"]
    size=6615; audio=.15*np.sin(2*np.pi*220*np.arange(size*len(labels))/44100)
    file=store.directory/"audio/canonical/s.wav"; sf.write(file,audio,44100,subtype="PCM_16")
    asset=SourceAsset(source_id="s",original_path="audio/canonical/s.wav",canonical_path="audio/canonical/s.wav",sha256=sha256_file(file),original_sample_rate=44100,original_channels=1,original_duration_s=len(audio)/44100,canonical_samples=len(audio),speaker_id=store.meta("speaker_id"),language_id="ja")
    store.ingest("assets",[asset])
    store.ingest("alignments",[phone("ja",label,i*size,(i+1)*size) for i,label in enumerate(labels)])
    return store


def test_language_isolation_covers_choose_variants_and_render(source_project):
    store=source_project; bank=store.create_bank("zh-cv")
    c=Candidate(speaker_id=store.meta("speaker_id"),source_id="s",utterance_id="u",alignment_revision="r",source_language="ja",alias="a",unit_key="a",unit_role="sustain_vowel",phones=["a"],raw_phones=[],span=SampleSpan(start=0,end=6615),landmarks=Landmarks(start=0,overlap=0,preutter=0,fixed_end=882,end=6615))
    store.ingest("candidates",[c])
    assert not store.candidates("a",bank_id=bank["id"])
    for method in (store.choose,store.add_variant):
        with pytest.raises(ValueError,match="目标语言"):method(bank["id"],c.candidate_id)


def test_cross_generation_has_actual_spans_and_no_silent_placeholders(source_project):
    store=source_project; bank=store.create_bank("zh-cv")
    result=generate(store,bank["id"],bank["generation"],"ja",True,15)
    assert result["generated"]>100 and result["missing"]
    assert any(item["alias"]=="er" for item in result["missing"])
    candidates=store.candidates(bank_id=bank["id"])
    c=next(c for c in candidates if c.alias=="shang")
    assert c.source_language=="ja" and c.target_language=="zh"
    context=c.phonetic_context["cross_language"]
    assert context["grade"]=="approximate" and context["losses"]
    assert [x["raw_label"] for x in context["donors"]]==["ja/sh","ja/a","ja/N"]
    audio,sr=sf.read(store.directory/c.phonetic_context["derived_audio_path"])
    assert sr==44100 and np.max(np.abs(audio))>.01 and np.isfinite(audio).all()
    assert len(audio)==sum(d["span"]["end"]-d["span"]["start"] for d in context["donors"])-sum(context["actual_crossfade_samples"])
    assert audio[0]==audio[-1]==0
    assert not any(c.alias=="er" for c in candidates)
    store.choose(bank["id"],c.candidate_id)
    # Reruns preserve the human selection and review.
    generate(store,bank["id"],store.bank(bank["id"])["generation"],"ja",False,15)
    assert store.selections(bank["id"])["shang"]["candidate_id"]==c.candidate_id


def test_cross_stale_job_cannot_publish(source_project):
    store=source_project; bank=store.create_bank("zh-cv")
    result=generate(store,bank["id"],bank["generation"]-1,"ja",False)
    assert result=={"stale":True} and not store.candidates(bank_id=bank["id"])


def test_export_sync_recover_preserve_cross_provenance_and_chinese_name(source_project,tmp_path):
    store=source_project; bank=store.create_bank("zh-cvvc")
    generate(store,bank["id"],bank["generation"],"ja",True)
    c=next(c for c in store.candidates(bank_id=bank["id"]) if c.alias=="shang")
    store.choose(bank["id"],c.candidate_id)
    with store.connection() as conn:
        conn.execute("DELETE FROM bank_selections WHERE bank_id=? AND alias!=?",(bank["id"],"shang"))
    first=compile_bank(store,bank["id"],store.bank(bank["id"])["generation"])
    assert not first["failures"]
    root=Path(first["path"])
    assert "presamp.ini" in first["manifest"]["files"]
    assert "ChineseCVVCPhonemizer" in (root/"character.yaml").read_text(encoding="utf-8")
    assert "日中测试" in (root/"character.txt").read_text(encoding="utf-8")
    entry=next(e for e in first["manifest"]["entries"] if e["alias"]=="shang")
    assert entry["cross_language"]["grade"]=="approximate"
    dest=tmp_path/"installed"; binding=bind(store,bank["id"],dest); sync_installation(store,binding)
    assert (dest/"presamp.ini").is_file()
    recovered=recover_bank(dest,tmp_path/"recovered","恢复普通话","zh-cvvc")
    other=ProjectStore(Path(recovered["project"]))
    restored=next(c for c in other.candidates() if c.alias=="shang")
    assert restored.source_language=="ja" and restored.target_language=="zh" and restored.provenance_kind.value=="cv_spliced"
    assert restored.phonetic_context["cross_language"]==c.phonetic_context["cross_language"]
    second=compile_bank(store,bank["id"],store.bank(bank["id"])["generation"])
    assert second["rendered"]==0
    store.update_bank(bank["id"],bank["name"],{**store.bank(bank["id"])["preset"],"release_mode":"strict"})
    strict=compile_bank(store,bank["id"],store.bank(bank["id"])["generation"])
    assert strict["exported"]==0 and strict["failures"]


def test_wrong_speaker_cannot_supply_phoneme(source_project):
    store=source_project; bank=store.create_bank("zh-cv")
    with store.connection() as conn:
        payload=json.loads(conn.execute("SELECT payload FROM assets").fetchone()[0]);payload["speaker_id"]="other"
        conn.execute("UPDATE assets SET payload=?",(json.dumps(payload),))
    with pytest.raises(ValueError,match="真实音素"):generate(store,bank["id"],bank["generation"],"ja",True)


def test_fade_is_bounded_and_complementary():
    parts=[np.full(100,.2,dtype=np.float32),np.full(200,.4,dtype=np.float32)]
    output,bounds,joins=blend(parts,1000)
    assert joins==[25] and bounds==[0,75] and len(output)==275
    assert np.max(output)<=.4 and output[0]==output[-1]==0


def test_tifa_import_keeps_language_and_only_matching_syllable_readings(tmp_path,monkeypatch):
    from utau_forge.tifa_adapter import phones_from_grids
    u=Utterance(utterance_id="u",source_id="s",speaker_id="sp",language_id="zh",span=SampleSpan(start=44100,end=66150),vad_span=SampleSpan(start=44100,end=66150),corpus_offset_samples=0)
    (tmp_path/"u.TextGrid").write_text("grid")
    (tmp_path/"readings.json").write_text(json.dumps({"u":[{"alias":"shi","tone":4},{"alias":"ma","tone":1}]}))
    monkeypatch.setattr("utau_forge.tifa_adapter.parse_textgrid_phones",lambda p:[(0,.1,"zh/sh"),(.1,.2,"zh/ir"),(.2,.3,"AP"),(.3,.4,"alien")])
    phones,failed=phones_from_grids([u],tmp_path,"hash",{"sh","ir"})
    assert not failed and phones[0].language_id=="zh" and phones[0].phoneme_id=="zh@1:sh"
    assert phones[0].features["syllable_alias"]=="shi" and phones[0].features["lexical_tone"]==4
    assert phones[2].features["event_type"]=="breath" and phones[3].unknown
    assert "syllable_alias" not in phones[3].features
