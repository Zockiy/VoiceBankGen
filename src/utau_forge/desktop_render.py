"""Real UTAU compatible resampler audition; never a stretched-audio substitute."""
from __future__ import annotations
from pathlib import Path
from uuid import uuid4
import subprocess
import soundfile as sf
import numpy as np
from .desktop_store import ProjectStore
from .desktop_service import safe_path,load_manifest
from .oto import parse_oto
from .storage import write_json

def render_note(store:ProjectStore,bank_id:str,candidate_id:str,engine:Path,length_ms:int=600,note:str="A3"):
    if not engine.is_file() or engine.suffix.lower()!=".exe":
        raise ValueError("请选择已安装的 UTAU resampler EXE")
    bank=store.bank(bank_id)
    if not bank["build_path"]:
        raise ValueError("请先制作音源")
    root=safe_path(store.directory,bank["build_path"])
    _,manifest=load_manifest(root)
    entry=next((entry for entry in manifest["entries"] if entry["candidate_id"]==candidate_id),None)
    if entry is None:
        raise ValueError("此候选尚未制作；请先使用这个音并制作")
    oto=parse_oto(entry["line"])
    directory=store.directory/"analyses/render"/uuid4().hex
    directory.mkdir(parents=True)
    output=directory/(Path(oto.filename).stem+f"_{note}_{length_ms}ms.wav")
    argv=[str(engine),str(safe_path(root,oto.filename)),str(output),note,"100","",str(oto.offset_ms),str(length_ms),str(oto.consonant_ms),str(oto.cutoff_ms),"100","0","T120",""]
    completed=subprocess.run(argv,capture_output=True,text=True,encoding="utf-8",errors="replace",timeout=120,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
    write_json(directory/"render.json",{"engine":str(engine),"argv":argv,"returncode":completed.returncode,"stdout":completed.stdout,"stderr":completed.stderr,"candidate_id":candidate_id,"alias":entry["alias"]})
    if completed.returncode or not output.is_file():
        raise RuntimeError(f"实际渲染失败：{completed.stderr[-1000:]}；详情 {directory}")
    audio,sr=sf.read(output,dtype="float32")
    if not len(audio) or not np.isfinite(audio).all() or np.max(np.abs(audio))<1e-5:
        raise RuntimeError("实际渲染输出为空或静音")
    return {"path":str(output),"engine":engine.name,"sample_rate":sr,"duration_s":len(audio)/sr,"alias":entry["alias"],"actual_render":True}
