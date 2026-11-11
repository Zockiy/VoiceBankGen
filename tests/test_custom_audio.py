import os
import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from utau_forge.bank_aliases import export_aliases
from utau_forge.desktop_audio_import import import_candidates
from utau_forge.desktop_import import recover_bank
from utau_forge.desktop_service import compile_bank
from utau_forge.desktop_store import ProjectStore
from utau_forge.desktop_sync import bind, sync_installation
from utau_forge.models import SourceAsset
from utau_forge.storage import sha256_file


@pytest.fixture
def project(tmp_path,monkeypatch):
    monkeypatch.setenv("UFORGE_DATA_DIR",str(tmp_path/"userdata"))
    store=ProjectStore.create(tmp_path/"project","Custom")
    return store,store.banks()[0]["id"]


@pytest.fixture
def wav(tmp_path,monkeypatch):
    bundled=Path(__file__).resolve().parents[1]/"dist/VoiceBankGen/runtime/ffmpeg"
    if bundled.exists():
        monkeypatch.setenv("PATH",str(bundled)+os.pathsep+os.environ.get("PATH",""))
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("audio import requires ffmpeg/ffprobe")
    path=tmp_path/"input.wav"
    samples=.1*np.sin(2*np.pi*220*np.arange(11025)/22050)
    sf.write(path,np.column_stack([samples,samples*.8]),22050,subtype="PCM_16")
    return path


def imported(store,bank_id,alias,wav):
    result=import_candidates(store,bank_id,store.bank(bank_id)["generation"],alias,[str(wav)])
    return store.candidate(result["candidate_ids"][0],bank_id),result


def build(store,bank_id):
    return compile_bank(store,bank_id,store.bank(bank_id)["generation"])


def test_custom_units_persist_validate_names_and_undo(project):
    store,bank_id=project
    unit=store.create_custom_unit(bank_id,"笑声","xiaosheng")
    assert store.display_alias(bank_id,unit["alias"])=="笑声"
    assert dict(store.targets(bank_id))[unit["alias"]]=="自定义"
    assert ProjectStore(store.directory).custom_units(bank_id)==[unit]
    assert store.history(bank_id) and not store.custom_units(bank_id)
    assert store.history(bank_id,True) and store.custom_units(bank_id)==[unit]
    for label,stem in [("a",None),("あ",None),("br",None),("笑声",None),("bad,label",None),("bad=label",None),("😀",None),("safe","../unsafe"),("safe","CON")]:
        with pytest.raises(ValueError):
            store.create_custom_unit(bank_id,label,stem)
    # Numbered-looking labels have their own stable internal identity.
    assert store.create_custom_unit(bank_id,"effect_1")["label"]=="effect_1"
    copied=store.create_bank("ja-cvvc",copy_units_from=bank_id)
    assert store.custom_units(copied["id"])==store.custom_units(bank_id)


def test_import_preserves_formal_choice_and_edits_on_reimport(project,wav):
    store,bank_id=project
    candidate,result=imported(store,bank_id,"あ",wav)
    assert result["imported"]==1 and not store.selections(bank_id)
    info=sf.info(store.directory/store.records("assets",SourceAsset)[0].canonical_path)
    assert info.samplerate==44100 and info.channels==1
    store.choose(bank_id,candidate.candidate_id)
    marks=candidate.landmarks.model_copy(update={"end":candidate.landmarks.end-100})
    store.edit_timing(bank_id,candidate.candidate_id,marks,["end"])
    _,again=imported(store,bank_id,"あ",wav)
    assert again["imported"]==0 and again["existing"]==1
    assert store.candidate(candidate.candidate_id,bank_id).landmarks==marks
    # A different file becomes another candidate, never replaces the selection.
    other=wav.with_name("other.wav")
    sf.write(other,.08*np.sin(2*np.pi*330*np.arange(22050)/44100),44100)
    new,_=imported(store,bank_id,"あ",other)
    assert len(store.candidates("あ",bank_id))==2
    assert store.selections(bank_id)["あ"]["candidate_id"]==candidate.candidate_id
    assert new.candidate_id!=candidate.candidate_id


def test_custom_and_breath_import_export_without_dsp_and_restore(project,wav,tmp_path):
    store,bank_id=project
    unit=store.create_custom_unit(bank_id,"笑声","xiaosheng")
    candidate,_=imported(store,bank_id,unit["alias"],wav)
    store.choose(bank_id,candidate.candidate_id)
    variant=store.add_variant(bank_id,candidate.candidate_id)
    bank=store.bank(bank_id)
    store.update_bank(bank_id,bank["name"],{**bank["preset"],"pitch_mode":"flatten","target_note":"auto","include_breaths":True})
    breath,_=imported(store,bank_id,"br",wav)
    store.choose(bank_id,breath.candidate_id)
    with patch("utau_forge.desktop_service._flatten",side_effect=AssertionError("custom pitch")),patch("utau_forge.desktop_service._level",side_effect=AssertionError("custom level")):
        result=build(store,bank_id)
    assert result["failures"]==[]
    assert {e["alias"] for e in result["manifest"]["entries"]}=={"笑声","笑声_1","br"}
    assert export_aliases("ja-cv",variant,"romaji",store.custom_units(bank_id))==["笑声_1"]
    assert result["manifest"]["custom_units"]==[unit]
    for entry in result["manifest"]["entries"]:
        assert entry["filename"].startswith("xiaosheng" if entry["unit_type"]=="自定义" else "br")
    installation=bind(store,bank_id,tmp_path/"installed")
    sync_installation(store,installation)
    old_hashes={p.name:sha256_file(p) for p in (tmp_path/"installed").glob("*.wav")}
    store.update_bank(bank_id,bank["name"],{**store.bank(bank_id)["preset"],"alias_format":"hiragana"})
    assert build(store,bank_id)["rendered"]==0
    assert sync_installation(store,installation)["copied_wavs"]==0
    assert old_hashes=={p.name:sha256_file(p) for p in (tmp_path/"installed").glob("*.wav")}
    recovered=recover_bank(Path(result["path"]),tmp_path/"restored","Restored")
    restored=ProjectStore(Path(recovered["project"]))
    assert [u["label"] for u in restored.custom_units(recovered["bank_id"])]==["笑声"]
    assert len(restored.selections(recovered["bank_id"]))==3


def test_numbered_custom_aliases_do_not_collide_with_other_labels(project,wav):
    store,bank_id=project
    sound=store.create_custom_unit(bank_id,"effect")
    store.create_custom_unit(bank_id,"effect_1")
    candidate,_=imported(store,bank_id,sound["alias"],wav)
    alias=store.add_variant(bank_id,candidate.candidate_id)
    assert store.display_alias(bank_id,alias)=="effect_2"


def test_import_stale_job_does_not_publish_candidates(project,wav):
    store,bank_id=project
    from utau_forge.audio import import_audio
    generation=store.bank(bank_id)["generation"]
    def changed(*args,**kwargs):
        asset=import_audio(*args,**kwargs)
        store.create_custom_unit(bank_id,"newer")
        return asset
    with patch("utau_forge.desktop_audio_import.import_audio",side_effect=changed):
        result=import_candidates(store,bank_id,generation,"あ",[str(wav)])
    assert result["stale"] and not store.candidates("あ")


def test_import_partial_error_and_explicit_auto_pitch(project,wav):
    store,bank_id=project
    result=import_candidates(store,bank_id,store.bank(bank_id)["generation"],"あ",[str(wav),str(wav.with_name("missing.wav"))])
    assert result["imported"]==1 and len(result["failures"])==1
    store.choose(bank_id,result["candidate_ids"][0])
    bank=store.bank(bank_id)
    store.update_bank(bank_id,bank["name"],{**bank["preset"],"pitch_mode":"flatten","target_note":"auto"})
    rendered=build(store,bank_id)
    assert rendered["failures"]==[] and rendered["exported"]==1
    # A pure sine is not a speech fixture; HARVEST need not recover exactly
    # 220 Hz. Verify that explicit auto pitch resolves a valid nearest note.
    hz=store.bank(bank_id)["preset"]["resolved_hz"]
    midi=69+12*np.log2(hz/440)
    assert 150<hz<300 and abs(midi-round(midi))<1e-6
