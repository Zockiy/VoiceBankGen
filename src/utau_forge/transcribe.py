"""Source-language ASR and user transcript revisions."""

from __future__ import annotations

import json
from pathlib import Path

import soundfile as sf

from .audio import read_region
from .config import Config
from .languages import language
from .models import SourceAsset, Utterance
from .model_lock import verify_model
from .storage import read_jsonl, write_jsonl


def import_transcripts(project_dir: Path, path: Path) -> int:
    updates: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if set(record) != {"utterance_id", "text"}:
                raise ValueError("transcript JSONL needs utterance_id and text")
            updates[record["utterance_id"]] = str(record["text"])
    utterances = read_jsonl(project_dir / "utterances.jsonl", Utterance)
    known = {item.utterance_id for item in utterances}
    if set(updates) - known:
        raise ValueError(f"unknown utterance ids: {sorted(set(updates) - known)}")
    changed = 0
    for item in utterances:
        if item.utterance_id in updates and item.revised_text != updates[item.utterance_id]:
            item.revised_text = updates[item.utterance_id]
            item.text_revision += 1
            changed += 1
    write_jsonl(project_dir / "utterances.jsonl", utterances)
    return changed


def transcribe(project_dir: Path, config: Config, emit=lambda **event: None) -> int:
    if config.asr.backend == "onnx_whisper":
        return transcribe_onnx(project_dir, config, emit)
    from .devices import whisper_device
    actual = whisper_device(config.asr.device)
    config.asr.device, config.asr.compute_type = actual["device"], actual["precision"]
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("faster-whisper missing; install with uv sync --extra asr") from exc
    assets = {asset.source_id: asset for asset in read_jsonl(project_dir / "sources" / "manifest.jsonl", SourceAsset)}
    utterances = read_jsonl(project_dir / "utterances.jsonl", Utterance)
    pending = [utt for utt in utterances if not (utt.revised_text or utt.asr_text)]
    if not pending:
        return 0
    lock = json.loads((project_dir / "models.lock.json").read_text(encoding="utf-8"))
    matching = [item for item in lock.get("models", []) if item["model_id"] == config.asr.model or item["model_id"].endswith("faster-whisper-" + config.asr.model)]
    model_name = config.asr.model
    if matching:
        if not verify_model(project_dir, matching[0]):
            raise RuntimeError("locked ASR model hash mismatch")
        model_name = matching[0].get("local_absolute_path") or str(project_dir / matching[0]["local_relative_path"])
    model = WhisperModel(model_name, device=config.asr.device, compute_type=config.asr.compute_type, local_files_only=config.runtime.offline)
    count = 0
    for utt in pending:
        source = assets[utt.source_id]
        audio = read_region(project_dir, source, utt.span.start, utt.span.end)
        clip_dir = project_dir / "audio" / "analysis"
        clip_dir.mkdir(parents=True, exist_ok=True)
        clip = clip_dir / f"{utt.utterance_id}.wav"
        if not clip.exists():
            sf.write(clip, audio, 44100, subtype="PCM_16")
        spec = language(utt.language_id).asr_spec({})
        segments, _ = model.transcribe(str(clip), language=spec["language"], vad_filter=False, condition_on_previous_text=False)
        materialized = list(segments)
        utt.asr_text = "".join(item.text for item in materialized).strip()
        if not utt.asr_text:
            utt.quality_flags.append("ASR_EMPTY")
        if any(item.no_speech_prob is not None and item.no_speech_prob > 0.6 for item in materialized):
            utt.quality_flags.append("ASR_SPEECH_UNCERTAIN")
        count += 1
        write_jsonl(project_dir / "utterances.jsonl", utterances)
        emit(stage="转写", current=count, total=len(pending), device=config.asr.device, precision=config.asr.compute_type)
    return count


def transcribe_onnx(project_dir: Path, config: Config, emit=lambda **event: None) -> int:
    from .onnx_whisper import OnnxWhisper
    assets = {asset.source_id: asset for asset in read_jsonl(project_dir / "sources/manifest.jsonl", SourceAsset)}
    pending = [utt for utt in read_jsonl(project_dir / "utterances.jsonl", Utterance) if not (utt.revised_text or utt.asr_text)]
    if not pending:
        return 0
    utterances = read_jsonl(project_dir / "utterances.jsonl", Utterance)
    lock = json.loads((project_dir / "models.lock.json").read_text(encoding="utf-8"))
    record = next((r for r in lock.get("models", []) if r["model_id"] == config.asr.model), None)
    if not record or not verify_model(project_dir, record):
        raise RuntimeError("ONNX Whisper 本地模型缺失或哈希不匹配")
    directory = Path(record["local_absolute_path"]) if record.get("local_absolute_path") else project_dir / record["local_relative_path"]
    model = OnnxWhisper(directory, config.asr.device, emit)
    pending_ids = {u.utterance_id for u in pending}
    count = 0
    for utt in utterances:
        if utt.utterance_id not in pending_ids:
            continue
        audio = read_region(project_dir, assets[utt.source_id], utt.span.start, utt.span.end)
        text, no_speech = model.transcribe(audio, utt.language_id)
        utt.asr_text = text
        if not text and "ASR_EMPTY" not in utt.quality_flags:
            utt.quality_flags.append("ASR_EMPTY")
        if no_speech > .6 and "ASR_SPEECH_UNCERTAIN" not in utt.quality_flags:
            utt.quality_flags.append("ASR_SPEECH_UNCERTAIN")
        count += 1
        write_jsonl(project_dir / "utterances.jsonl", utterances)
        emit(stage="转写", current=count, total=len(pending), device=model.device, precision=model.precision)
    return count
