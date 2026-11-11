"""Explicitly adopt external oto/WAV edits into the existing project."""
from __future__ import annotations
from decimal import Decimal
from pathlib import Path
import shutil
from uuid import uuid4
import soundfile as sf
from .desktop_store import ProjectStore
from .desktop_service import load_manifest,safe_path
from .models import Candidate,Landmarks,SampleSpan,SourceAsset
from .oto import parse_oto
from .storage import sha256_file, write_json
from .bank_aliases import split_variant
from collections import Counter

def adopt_external(store:ProjectStore,installation_id:str,generation:int,emit):
    with store.connection() as connection:
        row=connection.execute("SELECT * FROM installations WHERE id=?",(installation_id,)).fetchone()
        if row is None:
            raise ValueError("没有此安装目录")
    bank_id=row["bank_id"]
    if store.bank(bank_id)["generation"]!=generation:
        return {"stale":True}
    root=Path(row["path"])
    _,manifest=load_manifest(root)
    snapshot={name:sha256_file(safe_path(root,name)) for name in (*manifest["files"],"uforge.manifest.json","uforge.project-link.json") if safe_path(root,name).is_file()}
    from .languages import profile
    encoding=manifest.get("oto_encoding",profile(store.bank(bank_id)["profile_id"]).oto_encoding)
    current={}
    for line in (root/"oto.ini").read_bytes().decode(encoding).splitlines():
        try:
            entry=parse_oto(line)
            if (entry.alias,entry.filename) in current:
                raise ValueError("外部 oto 有重复别名/文件组合；请先处理重复行")
            current[(entry.alias,entry.filename)]=entry
        except ValueError:
            if "=" in line:
                raise
    adopted=set()
    references=Counter(item["candidate_id"] for item in store.selections(bank_id).values())
    for index,entry in enumerate(manifest["entries"]):
        candidate=store.candidate(entry["candidate_id"],bank_id)
        selection_alias=entry.get("selection_alias",candidate.alias+split_variant(entry["alias"])[1])
        if selection_alias in adopted:
            continue
        oto=current.get((entry["alias"],entry["filename"]))
        if not oto:
            raise ValueError(f"外部 oto 缺少 {entry['alias']}；无法自动采用删除，请换选候选或暂缓")
        path=safe_path(root,entry["filename"])
        info=sf.info(path)
        if info.samplerate!=44100 or info.channels!=1:
            raise ValueError("采用外部 WAV 需要 44.1kHz 单声道；可另行恢复导入")
        digest=sha256_file(path)
        window_start=entry["window_start"]
        changed=digest!=manifest["files"][entry["filename"]]
        absolute=oto.absolute_ms(Decimal(info.frames)*1000/44100)
        local=Landmarks(**{field:round(value*Decimal("44.1")) for field,value in absolute.items()})
        local.validate_oto(info.frames)
        # A numbered alternative may reference the primary's candidate. Editing
        # its external WAV/oto must not replace or retime the other selections.
        isolated=references[entry["candidate_id"]]>1 and oto!=parse_oto(entry["line"])
        if changed or isolated:
            target=store.directory/"audio/canonical"/(digest+".wav")
            if not target.exists():
                shutil.copy2(path,target)
            relative=target.relative_to(store.directory).as_posix()
            asset=SourceAsset(original_path=relative,sha256=digest,original_sample_rate=44100,original_channels=1,original_duration_s=info.duration,canonical_path=relative,canonical_samples=info.frames,speaker_id=store.meta("speaker_id"),language_id=candidate.source_language)
            candidate=candidate.model_copy(update={"candidate_id":str(uuid4()),"source_id":asset.source_id,"span":SampleSpan(start=local.start,end=local.end),"landmarks":local,"automatic_landmarks":local,"locked_fields":[],"review_status":"accepted","quality_flags":[],"parent_candidate_ids":[candidate.candidate_id],"phonetic_context":{"derived_audio_path":relative,"audio_samples":info.frames,"adopted_processed_preset":store.preset_key(bank_id),"external_filename":entry["filename"],"cross_language":candidate.phonetic_context.get("cross_language")}})
            store.ingest("assets",[asset])
            store.ingest("candidates",[candidate])
            store.choose(bank_id,candidate.candidate_id,selection_alias)
        else:
            marks=Landmarks(**{field:getattr(local,field)+window_start for field in Landmarks.model_fields})
            store.edit_timing(bank_id,candidate.candidate_id,marks,[])
        adopted.add(selection_alias)
        emit(stage="采用外部修改",current=index+1,total=len(manifest["entries"]))
    write_json(store.directory/"analyses"/("adoption_"+installation_id+".json"), {"root":str(root),"hashes":snapshot,"revision":store.revision})
    return {"pending":True,"adopted":len(adopted),"message":"外部修改已暂存；点击同步修改统一应用"}
