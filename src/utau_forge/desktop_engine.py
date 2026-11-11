"""Disposable legacy engine snapshot; SQLite remains the editable truth."""
from __future__ import annotations
import json
import os
import shutil
import sys
from pathlib import Path

from .desktop_store import ProjectStore
from .models import Candidate, Phone, Project, SourceAsset, Utterance
from .storage import read_jsonl, write_json, write_jsonl
from .config import read_config, write_config

def application_root() -> Path:
    return Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[2]

def link_tree(source: Path, target: Path):
    if not source.exists():
        return
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        destination = target / relative
        if path.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        elif not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(path, destination)
            except OSError:
                shutil.copy2(path, destination)

def prepare_engine(store: ProjectStore, model_paths: dict | None = None) -> Path:
    from .model_settings import load_model_paths, validate_model_paths, apply_engine_model_paths, bundled_model_lock, asr_record
    model_paths = validate_model_paths(load_model_paths() if model_paths is None else model_paths)
    root = store.directory / "analyses" / "engine"
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "project.json", Project(project_id=store.project_id, speaker_id=store.meta("speaker_id"),source_languages=store.meta("source_languages"),target_languages=store.meta("target_languages")))
    for table, filename, model in (("assets","sources/manifest.jsonl",SourceAsset), ("utterances","utterances.jsonl",Utterance), ("alignments","phones.jsonl",Phone), ("candidates","candidates.jsonl",Candidate)):
        write_jsonl(root / filename, store.records(table, model))
    for directory in ("audio/canonical", "sources", "derived"):
        link_tree(store.directory / directory, root / directory)
    config = read_config(store.directory / "config.yaml")
    locked = store.directory / "models.lock.json"
    lock = bundled_model_lock(application_root())
    # Resolve the installed local model every run, including after cancellation.
    # The original project config may still say large-v3 after small was bundled.
    asr_models = [item for item in lock.get("models", []) if asr_record(item)]
    asr_models.sort(key=lambda item: item.get("backend") != "onnx_whisper")
    if not asr_models and not model_paths["asr"]:
        raise RuntimeError("未找到本地 Whisper 模型；请打开 设置 → 模型设置 指定模型目录")
    if asr_models and not model_paths["asr"]:
        config.asr.model = next((item["model_id"] for item in asr_models if item["model_id"] == config.asr.model or item["model_id"].endswith("faster-whisper-" + config.asr.model)), asr_models[0]["model_id"])
        chosen = next(item for item in asr_models if item["model_id"] == config.asr.model)
        config.asr.backend = chosen.get("backend", "faster_whisper")
        if not Path(chosen["local_absolute_path"]).is_dir():
            raise RuntimeError("软件目录中的 Whisper 模型缺失，请恢复软件资源或在模型设置中指定本地目录")
    config.runtime.offline = True
    runtime = application_root() / "runtime" / "mfa"
    if (runtime / "python.exe").exists():
        config.alignment.executable = str(runtime / "python.exe")
        config.alignment.root_directory = str(store.directory / "analyses" / "mfa-data")
    elif (application_root() / ".mfa-env" / "Scripts" / "mfa.exe").exists():
        config.alignment.executable = str(application_root() / ".mfa-env" / "Scripts" / "mfa.exe")
        config.alignment.root_directory = str(application_root() / ".mfa-data")
    acoustic=next((item for item in lock["models"] if item["model_id"]=="japanese_mfa_acoustic"),None)
    dictionary=next((item for item in lock["models"] if item["model_id"]=="japanese_mfa_dictionary"),None)
    if acoustic and dictionary:
        config.alignment.acoustic=str(Path(acoustic["local_absolute_path"])/"japanese_mfa.zip")
        config.alignment.dictionary=str(Path(dictionary["local_absolute_path"])/"japanese_mfa.dict")
    apply_engine_model_paths(config, lock, model_paths)
    # Lightweight references are kept for reproducibility; no model payloads
    # are linked or copied into either project directory.
    write_json(locked, lock)
    write_json(root / "models.lock.json", lock)
    write_config(root / "config.yaml", config)
    return root

def ingest_engine(store: ProjectStore, root: Path, generation: int, bank_id: str) -> bool:
    # Guard in the same transaction as publication, so an older worker cannot
    # restore a selection or overwrite an edit made after the job started.
    with store.connection() as connection:
        current = connection.execute("SELECT generation FROM banks WHERE id=?", (bank_id,)).fetchone()[0]
        if current != generation:
            return False
        for table, filename, model, key in (("assets","sources/manifest.jsonl",SourceAsset,"source_id"), ("utterances","utterances.jsonl",Utterance,"utterance_id"), ("alignments","phones.jsonl",Phone,"phone_id"), ("candidates","candidates.jsonl",Candidate,"candidate_id")):
            incoming=read_jsonl(root / filename, model)
            if table=="alignments":
                identifiers=sorted({item.utterance_id for item in incoming})
                for offset in range(0,len(identifiers),500):
                    group=identifiers[offset:offset+500]
                    connection.execute("DELETE FROM alignments WHERE json_extract(payload,'$.utterance_id') IN ("+",".join("?" for _ in group)+")",group)
            if table=="candidates" and incoming:
                identifiers={item.candidate_id for item in incoming}
                active_utterances={item.utterance_id for item in incoming if item.provenance_kind.value=="original"}
                for row in connection.execute("SELECT payload FROM candidates").fetchall():
                    previous=Candidate.model_validate_json(row[0])
                    if previous.provenance_kind.value=="original" and previous.utterance_id in active_utterances and previous.candidate_id not in identifiers:
                        previous.quality_flags=sorted(set(previous.quality_flags+["STALE_ALIGNMENT"]))
                        connection.execute("UPDATE candidates SET payload=? WHERE id=?",(previous.model_dump_json(),previous.candidate_id))
            for item in incoming:
                if table == "candidates":
                    existing = connection.execute("SELECT payload FROM candidates WHERE id=?", (item.candidate_id,)).fetchone()
                    if existing:
                        item.review_status = Candidate.model_validate_json(existing[0]).review_status
                    connection.execute("INSERT OR REPLACE INTO candidates VALUES(?,?,?,?)", (item.candidate_id,item.alias,store.unit_type(item),item.model_dump_json()))
                else:
                    connection.execute(f"INSERT OR REPLACE INTO {table} VALUES(?,?)", (getattr(item,key),item.model_dump_json()))
    link_tree(root / "audio/canonical", store.directory / "audio/canonical")
    link_tree(root / "sources", store.directory / "sources")
    link_tree(root / "derived", store.directory / "derived")
    return True

def analyze(store: ProjectStore, bank_id: str, generation: int, files: list[str], device: str, backend: str, emit, model_paths: dict | None = None, source_language: str = "ja"):
    from .audio import import_audio
    from .segment import segment_source
    from .transcribe import transcribe
    from .align import align
    from .analyze import analyze_project
    from .devices import whisper_device
    from .languages import language
    from .languages.ja_cvvc import transition_candidates
    from .model_settings import load_model_paths, detect_model_paths
    model_paths = load_model_paths() if model_paths is None else model_paths
    language(source_language)
    if files and source_language not in store.meta("source_languages"):
        store.set_meta("source_languages",[*store.meta("source_languages"),source_language])
        store._write_identity()
    root = prepare_engine(store, model_paths)
    config = read_config(root / "config.yaml")
    resolved = detect_model_paths(application_root(),store.directory)
    emit(stage="模型路径", current=0, total=0, models={key: model_paths.get(key) or path for key,path in resolved.items()})
    for index, filename in enumerate(files):
        emit(stage="导入素材",current=index,total=len(files))
        import_audio(root, Path(filename), store.meta("speaker_id"), language_id=source_language)
        ingest_engine(store,root,generation,bank_id)
    for asset in read_jsonl(root / "sources/manifest.jsonl", SourceAsset):
        emit(stage="分段",current=0,total=0,device="cpu")
        segment_source(root, asset, config)
    ingest_engine(store,root,generation,bank_id)
    # Keep the requested policy for ONNX; its sessions report the actual device.
    if config.asr.backend == "onnx_whisper":
        config.asr.device = device
        emit(stage="转写",current=0,total=0,device=device,reason="正在加载 ONNX Whisper")
    else:
        actual = whisper_device(device)
        config.asr.device, config.asr.compute_type = actual["device"], actual["precision"]
        emit(stage="转写",current=0,total=0,**actual)
    def checkpoint(**event):
        ingest_engine(store,root,generation,bank_id)
        emit(**event)
    transcribe(root,config,checkpoint)
    emit(stage="对齐",current=0,total=0,device="cpu" if backend == "mfa" else device)
    if backend == "tifa":
        from .tifa_adapter import align_tifa
        alignment = align_tifa(root,device,checkpoint,model_paths=model_paths)
    else:
        if any(u.language_id!="ja" for u in read_jsonl(root/"utterances.jsonl",Utterance)):
            raise ValueError("普通话素材请使用 TIFA 对齐；当前 MFA 仅随附日语模型")
        alignment = align(root,config)
    ingest_engine(store,root,generation,bank_id)
    emit(stage="音节与标线",current=0,total=0,device="cpu")
    config.export_profiles=["ja-cv","zh-cv"]
    result=analyze_project(root,config)
    phones=read_jsonl(root/"phones.jsonl",Phone)
    utterances={item.utterance_id:item for item in read_jsonl(root/"utterances.jsonl",Utterance)}
    transitions=all_transitions(phones,utterances)
    from .breaths import breath_candidates
    breaths=breath_candidates(phones,utterances)
    write_jsonl(root/"candidates.jsonl", [*read_jsonl(root/"candidates.jsonl",Candidate),*transitions])
    if not ingest_engine(store,root,generation,bank_id):
        return {"stale":True}
    store.recommend(bank_id)
    optimized = 0
    if store.bank(bank_id)["profile_id"].endswith("-cvvc"):
        from .vc_timing import propose_vc_optimization
        expected = store.bank(bank_id)["generation"]
        manual = store.manual_timing_ids(bank_id)
        proposal = propose_vc_optimization(store, bank_id, emit=emit)
        optimized = store.apply_timing_batch(bank_id, [c for c in proposal["changes"] if c["candidate_id"] not in manual], expected)
    return {**result,"transitions":len(transitions),"breaths":len(breaths),"alignment":alignment,"optimized_vc_timings":optimized}


def all_transitions(phones,utterances):
    from .languages import language
    from .languages.ja_cvvc import transition_candidates as ja_transitions
    from .languages.zh import transition_candidates as zh_transitions
    return [*ja_transitions([p for p in phones if p.language_id=="ja"],utterances,language("ja").inventory),*zh_transitions([p for p in phones if p.language_id=="zh"],utterances,language("zh").inventory)]
