"""One cancellable process per job, newline JSON progress protocol."""
from __future__ import annotations
import json
import os
import sys
import traceback
from pathlib import Path
from uuid import uuid4

def main(request_path: str):
    # Frozen Python can ignore PYTHONIOENCODING and use the Windows ANSI code
    # page. QProcess reads this protocol as UTF-8, including kana aliases.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    request=json.loads(Path(request_path).read_text(encoding="utf-8"))
    def emit(**event):
        print(json.dumps({"job_id":request["job_id"],"generation":request.get("generation"),**event},ensure_ascii=False,default=str),flush=True)
    try:
        emit(stage="后台已响应，正在加载任务组件",current=0,total=0)
        from .desktop_engine import application_root
        from .devices import prepare_dlls
        root=application_root()
        os.environ["PATH"]=os.pathsep.join((str(root/"runtime"/"ffmpeg"/"bin"),str(root/"runtime"/"ffmpeg"),os.environ.get("PATH","")))
        prepare_dlls(root)
        action=request["action"]
        if action == "recover":
            from .desktop_import import recover_bank
            result=recover_bank(Path(request["source"]),Path(request["project"]),request["name"],request.get("profile","ja-cv"),emit)
        else:
            from .desktop_store import ProjectStore
            from .desktop_service import compile_bank, preview_file, refresh_timings, export_zip
            store=ProjectStore(Path(request["project"]))
            bank_id=request.get("bank_id")
            generation=request.get("generation")
            emit(stage="正在读取工程与准备任务",current=0,total=0)
            if action == "preview":
                result=preview_file(store,request["candidate_id"])
            elif action == "force_oto":
                from .desktop_service import force_automatic_timing, force_missing_sustain_timings
                if request.get("scope") == "missing_sustain":
                    result=force_missing_sustain_timings(store,bank_id,emit)
                else:
                    candidate=store.candidate(request["candidate_id"],bank_id)
                    emit(stage="强制标注当前切片 oto",current=0,total=1,device="cpu")
                    marks=force_automatic_timing(store,candidate,preview_file(store,candidate.candidate_id))
                    result={"candidate_id":candidate.candidate_id,"marks":marks.model_dump()}
                    emit(stage="强制标注当前切片 oto",current=1,total=1,device="cpu")
            elif action == "build":
                result=compile_bank(store,bank_id,generation,emit)
                if request.get("sync_related") and not result.get("stale"):
                    from .linked_sync import sync_linked_versions
                    result["linked_sync"]=sync_linked_versions(store,bank_id,emit)
            elif action == "timings":
                result=refresh_timings(store,bank_id,generation)
            elif action == "import_candidates":
                from .desktop_audio_import import import_candidates
                result=import_candidates(store,bank_id,generation,request["selection_alias"],request.get("files",[]),emit)
            elif action == "analyze":
                from .desktop_engine import analyze
                result=analyze(store,bank_id,generation,request.get("files",[]),request.get("device","auto"),request.get("backend","mfa"),emit,model_paths=request.get("model_paths"),source_language=request.get("source_language","ja"))
            elif action == "transitions":
                from .vc_timing import prepare_cvvc_version
                result=prepare_cvvc_version(store,bank_id,generation,emit)
            elif action == "optimize_vc":
                from .vc_timing import propose_vc_optimization
                result=propose_vc_optimization(store,bank_id,request.get("fade_ms",30),emit)
            elif action == "cross_language":
                from .cross_language import generate
                result=generate(store,bank_id,generation,request["source_language"],request.get("allow_approximate",False),request.get("crossfade_ms",15),emit)
            elif action == "repair":
                from .desktop_engine import prepare_engine,ingest_engine
                from .config import read_config
                from .repair import repair_project
                engine=prepare_engine(store)
                config=read_config(engine/"config.yaml")
                config.repair.crossfade_ms=store.bank(bank_id)["preset"].get("splice_crossfade_ms")
                if not store.bank(bank_id)["profile_id"].startswith("ja-"):
                    raise ValueError("普通话补音请使用独立的跨语种拼接入口；日语补尾规则不适用于普通话")
                config.export_profiles=["ja-cv"]
                emit(stage="约束补音",current=0,total=0,device="cpu")
                tails=repair_project(engine,config,mode="vowel-tail")
                result=repair_project(engine,config,mode="cv-splice",experimental=True)
                result["vowel_tails"]=tails
                if not ingest_engine(store,engine,generation,bank_id):
                    result={"stale":True}
                else:
                    store.recommend(bank_id)
            elif action == "zip":
                result={"path":str(export_zip(store,bank_id,Path(request["destination"])))}
            elif action == "render":
                from .desktop_render import render_note
                result=render_note(store,bank_id,request["candidate_id"],Path(request["engine"]),request.get("length_ms",600))
            elif action == "slices":
                from .desktop_engine import prepare_engine
                from .slices import export_slices
                from .config import read_config,write_config
                engine=prepare_engine(store)
                config=read_config(engine/"config.yaml")
                config.export_profiles=[store.bank(bank_id)["profile_id"]]
                write_config(engine/"config.yaml",config)
                custom_units=[unit for bank in store.banks() for unit in store.custom_units(bank["id"])]
                selected_ids=[selection["candidate_id"] for alias,selection in store.selections(bank_id).items()
                              if store.candidate(selection["candidate_id"],bank_id).alias == alias]
                result=export_slices(engine,all_candidates=request.get("all_candidates",False),top_k=request.get("top_k",1),
                                     selected_candidate_ids=selected_ids,custom_units=custom_units,emit=emit)
            elif action == "sync":
                from .desktop_sync import bind,sync_installation
                prepared=compile_bank(store,bank_id,generation,emit)
                if prepared.get("stale"):
                    emit(kind="result",result=prepared)
                    return
                if request.get("destination"):
                    identifier=bind(store,bank_id,Path(request["destination"]))
                else:
                    identifier=request["installation_id"]
                result=sync_installation(store,identifier,force=request.get("force",False))
                result["installation_id"]=identifier
                result["rendered"]=prepared.get("rendered",0)
                result["updated_timings"]=prepared.get("updated_timings",0)
                result["unchanged"]=prepared.get("unchanged",0)
                if request.get("sync_related") and result.get("state")=="synced":
                    from .linked_sync import sync_linked_versions
                    result["linked_sync"]=sync_linked_versions(store,bank_id,emit,exclude_installation=identifier)
            elif action == "adopt":
                from .desktop_adopt import adopt_external
                result=adopt_external(store,request["installation_id"],generation,emit)
            elif action == "backup":
                result={"path":str(store.backup(Path(request["destination"])))}
            else:
                raise ValueError(f"未知任务：{action}")
        emit(kind="result",result=result)
    except BaseException as error:
        emit(kind="error",error=str(error),details=traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main(sys.argv[1])
