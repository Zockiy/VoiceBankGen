"""Isolated official TIFA process and canonical phone-coordinate import."""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path
from uuid import uuid4,uuid5,NAMESPACE_URL
import soundfile as sf
from .desktop_engine import application_root
from .models import Phone,SampleSpan,SourceAsset,Utterance
from .audio import read_region
from .storage import json_hash,read_jsonl,sha256_file,write_json,write_jsonl
from .align import parse_textgrid_phones
import os

def phones_from_grids(utterances,output,model_hash,known):
    phones=[]
    failed=[]
    for utt in utterances:
        grid=output/(utt.utterance_id+".TextGrid")
        if not grid.exists():
            failed.append(utt.utterance_id)
            continue
        revision=json_hash({"backend":"TIFA","model":model_hash,"grid":sha256_file(grid),"text_revision":utt.text_revision})
        for start_s,end_s,raw in parse_textgrid_phones(grid):
            label=raw.removeprefix(utt.language_id+"/")
            start=max(utt.span.start,utt.span.start+round(start_s*44100))
            end=min(utt.span.end,utt.span.start+round(end_s*44100))
            if start>=end:
                continue
            canonical=label if label in known else {"SP":"sil","AP":"sp","br":"sp","EP":"sp","GS":"cl"}.get(label)
            phones.append(Phone(phone_id=str(uuid5(NAMESPACE_URL,f"{utt.utterance_id}:{revision}:{start}:{end}:{raw}")),utterance_id=utt.utterance_id,source_id=utt.source_id,language_id=utt.language_id,phoneme_id=("ja@2:" if utt.language_id=="ja" else "zh@1:")+canonical if canonical else None,raw_model_label=raw,span=SampleSpan(start=start,end=end),alignment_revision=revision,unknown=canonical is None,features={"backend":"tifa","diagnostics":str(output/"statistics"),"event_type":"breath" if label in {"AP","br"} else "phone"}))
    palatal={"h":"hy","k":"ky","g":"gy","n":"ny","b":"by","p":"py","m":"my","r":"ry"}
    for left,right in zip(phones,phones[1:]):
        if left.utterance_id==right.utterance_id and left.phoneme_id and right.phoneme_id=="ja@2:i" and abs(left.span.end-right.span.start)<=1:
            onset=left.phoneme_id.rsplit(":",1)[-1]
            if onset in palatal:
                left.features["contextual_palatalization"]=True
                left.phoneme_id="ja@2:"+palatal[onset]
    readings_path=output/"readings.json"
    if readings_path.exists():
        from .languages import language
        units=language("zh").inventory["units"]
        readings=json.loads(readings_path.read_text(encoding="utf-8"))
        for utt in utterances:
            ordered=[p for p in phones if p.utterance_id==utt.utterance_id and p.phoneme_id and p.phoneme_id.rsplit(":",1)[-1] not in {"sil","sp"}]
            expected=readings.get(utt.utterance_id,[])
            sequence=[v for item in expected for v in units.get(item["alias"],[])]
            actual=[p.phoneme_id.rsplit(":",1)[-1] for p in ordered]
            # TIFA can omit unsupported or skipped syllables. Annotate only
            # complete syllables within an exact, monotonic matching block.
            from difflib import SequenceMatcher
            blocks=SequenceMatcher(a=sequence,b=actual,autojunk=False).get_matching_blocks()
            offset=0
            for item in expected:
                size=len(units.get(item["alias"],[]))
                block=next((block for block in blocks if size and block.a<=offset and offset+size<=block.a+block.size),None)
                if block:
                    start=block.b+offset-block.a
                    for phone in ordered[start:start+size]:
                        phone.features.update(syllable_alias=item["alias"],lexical_tone=item["tone"])
                offset+=size
    return phones,failed

def runtime_command(device: str) -> list[str]:
    root=application_root()
    shared=root/"shared-runtime.json"
    if getattr(sys,"frozen",False) and shared.is_file():
        capability=json.loads(shared.read_text(encoding="utf-8"))
        if device=="cuda" and not capability.get("cuda"):
            raise RuntimeError("此便携包仅支持 CPU；CUDA 需要 GPU 版")
        return [str(root/"VoiceBankGenWorker.exe"),"--tifa-worker"]
    executable=root/"runtime"/("tifa-gpu" if device=="cuda" else "tifa-cpu")/"TIFAWorker.exe"
    if executable.exists():
        return [str(executable)]
    python = root/"runtime"/("tifa-gpu" if device=="cuda" else "tifa-cpu")/"python.exe"
    bridge = root/"_internal"/"utau_forge"/"data"/"runtime"/"tifa_entry.py"
    if python.is_file() and bridge.is_file():
        return [str(python),str(bridge)]
    if getattr(sys,"frozen",False):
        raise RuntimeError(f"缺少 TIFA {device} 组件")
    python=root/".tifa-env"/"Scripts"/"python.exe" if device=="cuda" else Path(sys.executable)
    if not python.exists():
        raise RuntimeError("缺少 TIFA CUDA runtime")
    return [str(python),str(root/"packaging"/"tifa_entry.py")]

def align_tifa(project_dir: Path,requested: str,emit=lambda **event:None,*,model_paths: dict | None = None,source_language: str | None = None) -> dict:
    from .model_settings import load_model_paths, tifa_resources
    root=application_root()
    model,dictionary,lock=tifa_resources(root,load_model_paths() if model_paths is None else model_paths)
    utterances=read_jsonl(project_dir/"utterances.jsonl",Utterance)
    assets={asset.source_id:asset for asset in read_jsonl(project_dir/"sources/manifest.jsonl",SourceAsset)}
    transcribed=[utt for utt in utterances if (utt.revised_text or utt.asr_text) and (source_language is None or utt.language_id==source_language)]
    if not transcribed:
        raise ValueError("没有可用于 TIFA 的文本")
    vocabulary=json.loads(model.with_name("vocabulary.json").read_text(encoding="utf-8"))["symbols"]
    groups={lang:[utt for utt in transcribed if utt.language_id==lang] for lang in sorted({utt.language_id for utt in transcribed})}
    if len(groups)>1:
        # Each front end receives only its own source language.
        results=[align_tifa(project_dir,requested,emit,model_paths=model_paths,source_language=lang) for lang in groups]
        return {"backend":"tifa","groups":results,"phones":len(read_jsonl(project_dir/"phones.jsonl",Phone))}
    lang=next(iter(groups))
    known={label.removeprefix(lang+"/") for label in vocabulary if label.startswith(lang+"/")}
    front_files={}
    if lang=="zh":
        chinese_dictionary=model.parent/"dictionaries/ds-zh-pinyin-lite.txt"
        if not known or not chinese_dictionary.is_file():
            raise ValueError("所选 TIFA 模型缺少普通话词表或 dictionaries/ds-zh-pinyin-lite.txt；请使用完整多语言模型")
        front_files={"zh_dictionary":sha256_file(chinese_dictionary)}
    index_path=project_dir/"alignment/tifa-index.json"
    index=json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {"signatures":{}}
    existing=read_jsonl(project_dir/"phones.jsonl",Phone)
    existing_ids={phone.utterance_id for phone in existing if phone.features.get("backend")=="tifa"}
    signatures={utt.utterance_id:json_hash({"source":assets[utt.source_id].sha256,"span":utt.span.model_dump(),"text":utt.revised_text or utt.asr_text,"revision":utt.text_revision,"models":lock["files"],"device_policy":requested,"adapter":"tifa-3.1-multilingual","language":utt.language_id,"front_files":front_files}) for utt in transcribed}
    pending=[utt for utt in transcribed if not (utt.utterance_id in existing_ids and index["signatures"].get(utt.utterance_id)==signatures[utt.utterance_id])]
    if not pending:
        return {"backend":"tifa","cached":True,"device":index.get("device","cached"),"utterances":len(transcribed),"phones":len(existing),"failed":[]}
    pending_ids={utt.utterance_id for utt in pending}
    retained=[phone for phone in existing if phone.utterance_id not in pending_ids]
    run=project_dir/"alignment"/("tifa_"+uuid4().hex)
    corpus=run/"corpus"
    output=run/"output"
    corpus.mkdir(parents=True)
    output.mkdir()
    for utt in pending:
        sf.write(corpus/(utt.utterance_id+".wav"),read_region(project_dir,assets[utt.source_id],utt.span.start,utt.span.end),44100,subtype="PCM_16")
        (corpus/(utt.utterance_id+".txt")).write_text(utt.revised_text or utt.asr_text,encoding="utf-8")
    from .devices import cpu_only_package
    requested_device="cpu" if requested=="cpu" or (requested=="auto" and cpu_only_package(root)) else "cuda"
    def checkpoint(device):
        completed,failed=phones_from_grids(pending,output,lock["model_sha256"],known)
        if completed:
            write_jsonl(project_dir/"phones.jsonl",[*retained,*completed])
            for utt in pending:
                if utt.utterance_id not in failed:
                    index["signatures"][utt.utterance_id]=signatures[utt.utterance_id]
            index["device"]=device
            write_json(index_path,index)
        return completed,failed
    def inference(device,batch):
        request={"language":lang,"upstream":str(root/"resources/upstream/TIFA"),"model":str(model),"unidic":str(dictionary),"corpus":str(corpus),"output":str(output),"device":device,"batch_size":batch}
        path=run/(f"request_{device}_{batch}.json")
        write_json(path,request)
        log=[]
        environment={**os.environ,"PYTHONIOENCODING":"utf-8","PYTHONUTF8":"1"}
        process=subprocess.Popen([*runtime_command(device),str(path)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding="utf-8",errors="replace",env=environment,creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=="win32" else 0)
        for line in process.stdout:
            log.append(line)
            try:
                event=json.loads(line)
                if "stage" in event:
                    checkpoint(device)
                    emit(**event)
            except ValueError:
                pass
        process.wait()
        (run/f"{device}_{batch}.log").write_text("".join(log),encoding="utf-8")
        if process.returncode:
            raise RuntimeError("".join(log)[-3500:])
    actual=requested_device
    try:
        inference(actual,4 if actual=="cuda" else 1)
    except RuntimeError as error:
        if actual=="cuda" and "out of memory" in str(error).lower():
            emit(stage="TIFA 对齐",current=0,total=0,device="cuda",reason="显存不足，batch 降为 1")
            try:
                inference(actual,1)
            except RuntimeError:
                if requested!="auto":
                    raise
                actual="cpu"
                emit(stage="TIFA 对齐",current=0,total=0,device=actual,reason="CUDA 失败，自动回退 CPU")
                inference(actual,1)
        elif requested=="auto":
            actual="cpu"
            emit(stage="TIFA 对齐",current=0,total=0,device=actual,reason=str(error)[-350:])
            inference(actual,1)
        else:
            raise RuntimeError(f"TIFA {actual} 失败：{error}；可切换 CPU 后重试") from error
    completed,failed=checkpoint(actual)
    phones=[*retained,*completed]
    if not phones:
        raise RuntimeError("TIFA 未产生可用音素；详情见 alignment 日志")
    write_jsonl(project_dir/"phones.jsonl",phones)
    report={"backend":"tifa","device":actual,"utterances":len(transcribed)-len(failed),"failed":failed,"phones":len(phones),"unknown":sum(phone.unknown for phone in phones),"run_dir":str(run)}
    write_json(run/"result.json",report)
    return report
