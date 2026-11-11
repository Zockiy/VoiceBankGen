"""Recover editable candidates from actual installed WAV and oto coordinates."""
from __future__ import annotations
import json
import shutil
from decimal import Decimal
from pathlib import Path
from uuid import uuid4
import soundfile as sf
from .desktop_store import ProjectStore
from .desktop_service import safe_path, write_bank_metadata
from .models import Candidate, Landmarks, SampleSpan, SourceAsset, ProvenanceKind
from .languages import language, profile
from .oto import parse_oto
from .storage import sha256_file
from .bank_aliases import export_aliases, split_variant, BREATH_ALIAS, ALIAS_FORMATS

def recover_bank(root: Path, directory: Path, name: str, profile_id: str = "ja-cv", emit=lambda **event:None) -> dict:
    store=ProjectStore.create(directory,name,profile_id=profile_id)
    bank=store.banks()[0]
    if profile_id != bank["profile_id"]:
        bank=store.create_bank(profile_id)
    store.update_bank(bank["id"],name,{"pitch_mode":"preserve","target_note":"auto","repair":False})
    bank=store.bank(bank["id"])
    build=directory/"builds"/bank["id"]/"recovered"/"voicebank"
    build.mkdir(parents=True)
    # Retain original bytes and all unmanaged files, including unknown oto rows.
    shutil.copytree(root,build,dirs_exist_ok=True)
    oto=root/"oto.ini"
    raw=oto.read_bytes()
    import yaml
    settings=yaml.safe_load((root/"character.yaml").read_text(encoding="utf-8")) if (root/"character.yaml").exists() else {}
    encoding=(settings or {}).get("text_file_encoding",profile(profile_id).oto_encoding)
    try:
        lines=raw.decode(encoding).splitlines()
    except UnicodeDecodeError:
        lines=raw.decode("utf-8-sig").splitlines()
    units={profile(profile_id).aliases(unit)[0]:unit for unit in profile(profile_id).required_units(language(profile(profile_id).language_id).inventory,"core")}
    identities={alias:canonical for canonical in units for fmt in ALIAS_FORMATS for alias in export_aliases(profile_id,canonical,fmt)}
    saved_manifest=root/"uforge.manifest.json"
    custom_identities={}
    saved={}
    if saved_manifest.exists():
        # Managed custom definitions preserve labels, safe file stems and
        # numbered alternatives even when the original project was lost.
        saved=json.loads(saved_manifest.read_text(encoding="utf-8"))
        for definition in saved.get("custom_units",[]):
            custom=store.create_custom_unit(bank["id"],definition["label"],definition["filename_stem"])
            custom_identities[definition["label"]]=custom["alias"]
    saved_entries={(e["alias"],e["filename"]):e for e in saved.get("entries",[])}
    assets={}
    entries=[]
    warnings=[]
    for index,line in enumerate(lines):
        emit(stage="恢复音源",current=index,total=len(lines))
        try:
            parsed=parse_oto(line)
            path=safe_path(root,parsed.filename)
            info=sf.info(path)
            base,suffix=split_variant(parsed.alias)
            if parsed.alias in custom_identities:
                canonical,suffix=custom_identities[parsed.alias],""
            else:
                canonical=custom_identities.get(base) or identities.get(base, BREATH_ALIAS if base==BREATH_ALIAS else None)
            if canonical is None:
                warnings.append(f"保留未知别名：{parsed.alias}")
                continue
            if info.samplerate!=44100 or info.channels!=1:
                warnings.append(f"保留非标准 WAV：{parsed.filename}，请重新导入原素材")
                continue
            digest=sha256_file(path)
            destination=directory/"audio"/"canonical"/(digest+".wav")
            if not destination.exists():
                shutil.copy2(path,destination)
            if digest not in assets:
                asset=SourceAsset(original_path=destination.relative_to(directory).as_posix(),sha256=digest,original_sample_rate=44100,original_channels=1,original_duration_s=info.duration,canonical_path=destination.relative_to(directory).as_posix(),canonical_samples=info.frames,speaker_id=store.meta("speaker_id"),language_id=profile(profile_id).language_id)
                assets[digest]=asset
            asset=assets[digest]
            absolute=parsed.absolute_ms(Decimal(info.frames)*1000/44100)
            marks=Landmarks(**{field:int(round(value*Decimal("44.1"))) for field,value in absolute.items()})
            marks.validate_oto(info.frames)
            unit=units.get(canonical)
            role="custom" if canonical.startswith("custom:") else ("breath" if canonical==BREATH_ALIAS else (unit.unit_type if unit.unit_type in {"VC","VV"} else "sustain_vowel"))
            key=unit.unit_key if unit else (canonical if role=="custom" else BREATH_ALIAS)
            candidate=Candidate(speaker_id=asset.speaker_id,source_id=asset.source_id,utterance_id="imported-oto",alignment_revision="imported-oto-v1",source_language=profile(profile_id).language_id,unit_role=role,unit_key=key,alias=canonical,phones=key.split("+"),raw_phones=[],span=SampleSpan(start=marks.start,end=marks.end),landmarks=marks,automatic_landmarks=marks,review_status="accepted",phonetic_context={"derived_audio_path":asset.canonical_path,"audio_samples":info.frames,"imported_filename":parsed.filename,"original_oto_line":line})
            original=saved_entries.get((parsed.alias,parsed.filename),{})
            candidate.source_language=original.get("source_language",candidate.source_language)
            candidate.target_language=original.get("target_language",profile(profile_id).language_id)
            candidate.provenance_kind=ProvenanceKind(original.get("provenance_kind","original"))
            candidate.phonetic_context["cross_language"]=original.get("cross_language")
            candidate.phonetic_context["timing_evidence"]=original.get("timing_evidence")
            if original.get("cross_language") and original["cross_language"].get("grade")=="approximate": candidate.quality_flags=["CROSS_LANGUAGE_APPROXIMATE"]
            store.ingest("candidates",[candidate])
            with store.connection() as connection:
                store._put_selection(connection,bank["id"],canonical+suffix,candidate.candidate_id,True)
            entries.append({"entry_id":str(uuid4()),"alias":parsed.alias,"selection_alias":canonical+suffix,"candidate_id":candidate.candidate_id,"filename":parsed.filename,"line":line,"marks":marks.model_dump(),"samples":info.frames,"window_start":0,"unit_type":store.unit_type(candidate),"provenance_kind":candidate.provenance_kind.value,"parents":[],"review_status":"accepted","source_language":candidate.source_language,"target_language":candidate.target_language,"cross_language":candidate.phonetic_context.get("cross_language")})
        except (ValueError,OSError,RuntimeError) as error:
            warnings.append(f"保留未支持的 oto 行 {index+1}：{error}")
    store.ingest("assets",list(assets.values()))
    store.set_meta("source_languages",sorted({c.source_language for c in store.candidates()}))
    store._write_identity()
    ordinary=[entry["alias"] for entry in entries if entry["unit_type"] not in {"自定义","呼吸"}]
    notation=saved.get("alias_format")
    if notation not in ALIAS_FORMATS:
        notation="katakana" if any(any("\u30a1"<=char<="\u30f6" for char in alias) for alias in ordinary) else ("hiragana" if any(any("\u3041"<=char<="\u3096" for char in alias) for alias in ordinary) else "romaji")
    store.update_bank(bank["id"],name,{**bank["preset"],"alias_format":notation,"include_breaths":any(entry["unit_type"]=="呼吸" for entry in entries)})
    bank=store.bank(bank["id"])
    if not entries:
        raise ValueError("没有可编辑的 44.1kHz 单声道、标准 CV 坐标；原文件仍完整保留在恢复工程")
    missing=[{"alias":alias,"unit_type":kind} for alias,kind in store.targets(bank["id"]) if alias not in store.selections(bank["id"])]
    from .storage import write_json
    write_json(build/"missing-units.json",{"missing":missing,"warnings":warnings})
    write_bank_metadata(store,bank,build,entries,missing)
    with store.connection() as connection:
        connection.execute("UPDATE banks SET build_path=? WHERE id=?",(build.relative_to(directory).as_posix(),bank["id"]))
    # No installed bytes change on opening/importing. User can subsequently bind.
    return {"project":str(directory),"bank_id":bank["id"],"imported":len(entries),"warnings":warnings}
