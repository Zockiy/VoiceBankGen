import json
from pathlib import Path

import pytest

from test_vc_timing import project
from utau_forge.desktop_store import ProjectStore
from utau_forge.desktop_service import compile_bank
from utau_forge.desktop_sync import bind,sync_installation,installations
from utau_forge.linked_sync import sync_linked_versions
from utau_forge.oto import parse_oto
from utau_forge.shared_banks import CV_BASE_KEY
from utau_forge.storage import sha256_file


def test_cv_authoritative_then_selection_and_oto_are_shared_bidirectionally(project):
    store,cv,_,ga,vc=project
    child=store.create_bank("ja-cvvc",copy_units_from=cv["id"])
    other=store.create_bank("ja-cvvc",copy_units_from=cv["id"])
    edited=ga.landmarks.model_copy(update={"overlap":19000})
    store.edit_timing(cv["id"],"ga",edited,["overlap"])
    assert store.candidate("ga",child["id"]).landmarks==edited
    assert store.candidate("ga",other["id"]).locked_fields==["overlap"]
    from_child=edited.model_copy(update={"preutter":22500})
    store.edit_timing(child["id"],"ga",from_child,["overlap"])
    assert store.candidate("ga",cv["id"]).landmarks==from_child
    assert store.candidate("ga",other["id"]).landmarks==from_child
    alternative=ga.model_copy(update={"candidate_id":"ga-choice"})
    store.ingest("candidates",[alternative])
    store.choose(child["id"],alternative.candidate_id)
    assert store.selections(cv["id"])["が"]["candidate_id"]==alternative.candidate_id
    assert store.selections(other["id"])["が"]["candidate_id"]==alternative.candidate_id
    variant=store.add_variant(child["id"],alternative.candidate_id)
    assert variant in store.selections(cv["id"]) and variant in dict(store.targets(other["id"]))
    store.remove_variant(cv["id"],variant)
    assert variant not in store.selections(child["id"])


def test_vc_remains_local_and_settings_custom_units_are_shared(project):
    store,cv,_,_,vc=project
    child=store.create_bank("ja-cvvc",copy_units_from=cv["id"])
    other=store.create_bank("ja-cvvc",copy_units_from=cv["id"])
    before=store.bank(cv["id"])["generation"]
    store.choose(child["id"],vc.candidate_id)
    store.edit_timing(child["id"],vc.candidate_id,vc.landmarks.model_copy(update={"overlap":14553}),[])
    assert store.bank(cv["id"])["generation"]==before
    assert "a g" not in store.selections(other["id"])
    assert store.candidate("vc",other["id"]).landmarks==vc.landmarks
    settings={**store.bank(child["id"])["preset"],"alias_format":"romaji","target_note":"D3","include_breaths":True}
    store.update_bank(child["id"],"CVVC name",settings)
    assert store.bank(cv["id"])["preset"]["alias_format"]=="romaji"
    assert store.bank(other["id"])["preset"]["target_note"]=="D3"
    assert store.bank(other["id"])["preset"]["include_breaths"]
    assert store.candidate("vc",child["id"]).landmarks.overlap==14553
    custom=store.create_custom_unit(child["id"],"笑声","laugh")
    assert custom in store.custom_units(cv["id"]) and custom in store.custom_units(other["id"])


def test_legacy_cvvc_migration_uses_existing_cv_and_keeps_old_records(project):
    store,cv,_,ga,vc=project
    child=store.create_bank("ja-cvvc",copy_units_from=cv["id"])
    with store.connection() as connection:
        connection.execute("UPDATE banks SET preset=? WHERE id=?",(json.dumps(cv["preset"]),child["id"]))
    previous=ga.landmarks.model_copy(update={"overlap":20000})
    store.edit_timing(child["id"],"ga",previous,[])
    store.choose(child["id"],vc.candidate_id)
    root_cv=ga.landmarks.model_copy(update={"overlap":18000})
    store.edit_timing(cv["id"],"ga",root_cv,[])
    reopened=ProjectStore(store.directory)
    assert reopened.cv_base_id(child["id"])==cv["id"]
    assert reopened.candidate("ga",child["id"]).landmarks==root_cv
    assert reopened.candidate("ga",cv["id"]).landmarks==root_cv
    assert reopened.selections(child["id"])["a g"]["candidate_id"]==vc.candidate_id
    assert reopened.meta("before_shared_cv:"+child["id"]) is not None
    with reopened.connection() as connection:
        assert connection.execute("SELECT marks FROM candidate_timings WHERE candidate_id='ga' AND preset_hash=?",(reopened.timing_key(child["id"]),)).fetchone()


def test_shared_history_is_visible_in_both_versions_and_retains_eight_steps(project):
    store,cv,_,ga,vc=project
    child=store.create_bank("ja-cvvc",copy_units_from=cv["id"])
    for index in range(10):
        owner=child["id"] if index%2 else cv["id"]
        store.edit_timing(owner,"ga",ga.landmarks.model_copy(update={"overlap":18000+index*10}),[])
    for _ in range(8):
        assert store.history(child["id"])
        assert store.candidate("ga",child["id"]).landmarks==store.candidate("ga",cv["id"]).landmarks
    assert not store.history(child["id"])
    for _ in range(8): assert store.history(child["id"],redo=True)
    assert not store.history(child["id"],redo=True)


def test_sync_updates_bound_cv_and_cvvc_and_keeps_basic_wavs_identical(project,tmp_path):
    store,cv,_,ga,vc=project
    child=store.create_bank("ja-cvvc",copy_units_from=cv["id"])
    store.choose(child["id"],vc.candidate_id)
    roots={cv["id"]:tmp_path/"Singers/CV",child["id"]:tmp_path/"Singers/CVVC"}
    for bank_id,root in roots.items():
        compile_bank(store,bank_id,store.bank(bank_id)["generation"])
        sync_installation(store,bind(store,bank_id,root))
    store.edit_timing(child["id"],"ga",ga.landmarks.model_copy(update={"overlap":18963}),[])
    assert all(installations(store,identifier)[0]["state"]=="pending" for identifier in roots)
    results=sync_linked_versions(store,cv["id"])
    assert len(results)==2 and all(result["state"]=="synced" for result in results)
    manifests={key:json.loads((root/"uforge.manifest.json").read_text(encoding="utf-8")) for key,root in roots.items()}
    cv_entries={e["selection_alias"]:e for e in manifests[cv["id"]]["entries"]}
    child_entries={e["selection_alias"]:e for e in manifests[child["id"]]["entries"]}
    for alias,entry in cv_entries.items():
        other=child_entries[alias]
        assert parse_oto(entry["line"]).overlap_ms==parse_oto(other["line"]).overlap_ms
        assert manifests[cv["id"]]["files"][entry["filename"]]==manifests[child["id"]]["files"][other["filename"]]
    assert "a g" in child_entries and "a g" not in cv_entries


def test_multiple_cv_versions_require_explicit_association(project):
    store,cv,_,_,_=project
    other=store.create_bank("ja-cv")
    child=store.create_bank("ja-cvvc")
    assert CV_BASE_KEY not in store.bank(child["id"])["preset"]
    store.link_cvvc(child["id"],other["id"])
    assert store.cv_base_id(child["id"])==other["id"]


def test_editing_shared_cv_branches_cvvc_redo(project):
    store,cv,_,ga,_=project
    child=store.create_bank("ja-cvvc",copy_units_from=cv["id"])
    unit=store.create_custom_unit(child["id"],"effect")
    assert store.history(child["id"])
    store.edit_timing(cv["id"],"ga",ga.landmarks.model_copy(update={"overlap":18000}),[])
    assert not store.history(child["id"],redo=True)
    assert unit not in store.custom_units(cv["id"])


def test_linked_sync_never_writes_singers_root_or_recreates_removed_binding(project,tmp_path):
    store,cv,_,_,_=project
    singers=tmp_path/"Singers"
    singers.mkdir()
    with pytest.raises(ValueError,match="Singers"):
        bind(store,cv["id"],singers)
    with store.connection() as connection:
        connection.execute("INSERT INTO installations VALUES(?,?,?,?,?,?,?)",("legacy",cv["id"],str(singers),store.revision,-1,"pending",None))
    missing=tmp_path/"removed_voicebank"
    bind(store,cv["id"],missing)
    results=sync_linked_versions(store,cv["id"])
    assert len(results)==2 and all(r["state"]=="skipped" for r in results)
    assert list(singers.iterdir())==[] and not missing.exists()


def test_linked_sync_preserves_external_cvvc_edits_for_review(project,tmp_path):
    store,cv,_,ga,vc=project
    child=store.create_bank("ja-cvvc",copy_units_from=cv["id"])
    store.choose(child["id"],vc.candidate_id)
    roots={cv["id"]:tmp_path/"Singers/CV",child["id"]:tmp_path/"Singers/CVVC"}
    for identifier,root in roots.items():
        compile_bank(store,identifier,store.bank(identifier)["generation"])
        sync_installation(store,bind(store,identifier,root))
    target=roots[child["id"]]/"oto.ini"
    lines=target.read_text(encoding="cp932").splitlines()
    parts=lines[0].split(",")
    parts[-1]=str(float(parts[-1])+1)
    lines[0]=",".join(parts)
    target.write_text("\r\n".join(lines)+"\r\n",encoding="cp932")
    external=target.read_bytes()
    store.edit_timing(cv["id"],"ga",ga.landmarks.model_copy(update={"overlap":18000}),[])
    results=sync_linked_versions(store,cv["id"])
    assert any(r["bank_id"]==cv["id"] and r["state"]=="synced" for r in results)
    assert any(r["bank_id"]==child["id"] and r["state"]!="synced" for r in results)
    assert target.read_bytes()==external


def test_new_cvvc_keeps_existing_automatic_cv_choice(project):
    from utau_forge.vc_timing import prepare_cvvc_version
    store,cv,_,ga,_=project
    alternative=ga.model_copy(update={"candidate_id":"ga-better","quality_score":1.0})
    store.ingest("candidates",[alternative])
    with store.connection() as connection:
        store._put_selection(connection,cv["id"],"が",ga.candidate_id,False)
    child=store.create_bank("ja-cvvc",copy_units_from=cv["id"])
    prepare_cvvc_version(store,child["id"],store.bank(child["id"])["generation"])
    assert store.selections(cv["id"])["が"]["candidate_id"]==ga.candidate_id
    assert store.selections(child["id"])["が"]["candidate_id"]==ga.candidate_id


def test_recovered_cvvc_foundation_keeps_original_wav_bytes(project,tmp_path):
    from utau_forge.desktop_import import recover_bank
    store,cv,_,_,vc=project
    child=store.create_bank("ja-cvvc",copy_units_from=cv["id"])
    store.choose(child["id"],vc.candidate_id)
    original=compile_bank(store,child["id"],store.bank(child["id"])["generation"])
    restored=recover_bank(Path(original["path"]),tmp_path/"recovered","Recovered","ja-cvvc")
    recovered=ProjectStore(Path(restored["project"]))
    base=recovered.cv_base_id(restored["bank_id"])
    assert base!=restored["bank_id"]
    rebuilt=compile_bank(recovered,base,recovered.bank(base)["generation"])
    entries={e["selection_alias"]:e for e in original["manifest"]["entries"]}
    for entry in rebuilt["manifest"]["entries"]:
        old=Path(original["path"])/entries[entry["selection_alias"]]["filename"]
        assert sha256_file(Path(rebuilt["path"])/entry["filename"])==sha256_file(old)
