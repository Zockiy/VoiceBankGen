"""MFA subprocess adapter and long TextGrid phone-tier reader."""

from __future__ import annotations

import re
import os
import json
import shutil
import subprocess
from pathlib import Path
from uuid import uuid4

import soundfile as sf

from .audio import read_region
from .config import Config
from .languages import language
from .models import Phone, SampleSpan, SourceAsset, Utterance
from .storage import json_hash, read_jsonl, sha256_file, write_json, write_jsonl


_TIER = re.compile(r'^\s*name\s*=\s*"(.*)"')
_INTERVAL = re.compile(r"^\s*intervals\s*\[\d+\]:")
_NUMBER = re.compile(r"^\s*(xmin|xmax)\s*=\s*([\d.eE+-]+)")
_TEXT = re.compile(r'^\s*text\s*=\s*"(.*)"')


def parse_textgrid_phones(path: Path) -> list[tuple[float, float, str]]:
    tier = ""
    current: dict[str, str] = {}
    output: list[tuple[float, float, str]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        match = _TIER.match(line)
        if match:
            tier = match.group(1)
            continue
        if _INTERVAL.match(line):
            current = {}
            continue
        number = _NUMBER.match(line)
        if number and current is not None:
            current[number.group(1)] = number.group(2)
            continue
        text = _TEXT.match(line)
        if text and tier.lower() in {"phones", "phone"}:
            if "xmin" not in current or "xmax" not in current:
                raise ValueError("TextGrid phone interval lacks coordinates")
            start, end = float(current["xmin"]), float(current["xmax"])
            if end <= start:
                raise ValueError("TextGrid phone interval is empty")
            if text.group(1):
                output.append((start, end, text.group(1)))
            current = {}
    if not output:
        raise ValueError("TextGrid contains no phone intervals")
    return output


def align(project_dir: Path, config: Config) -> dict[str, int | str]:
    executable = shutil.which(config.alignment.executable) or (config.alignment.executable if Path(config.alignment.executable).is_file() else None)
    if executable is None:
        raise RuntimeError("MFA CLI missing; install in a separate conda environment and set alignment.executable")
    environment = os.environ.copy()
    executable_path = Path(executable).resolve()
    command=[executable]
    if executable_path.name.lower() in {"python.exe","python"}:
        command=[executable,"-m","montreal_forced_aligner"]
        prefix=executable_path.parent
        environment["PATH"]=os.pathsep.join((str(prefix),str(prefix/"Library/bin"),str(prefix/"Scripts"),environment.get("PATH","")))
    if executable_path.parent.name.lower() == "scripts":
        prefix = executable_path.parent.parent
        environment["PATH"] = os.pathsep.join((str(prefix), str(prefix / "Library" / "bin"), str(prefix / "Scripts"), environment.get("PATH", "")))
    environment["PYTHONIOENCODING"] = "utf-8"
    root = Path(config.alignment.root_directory) if config.alignment.root_directory else project_dir / ".mfa-data"
    environment["MFA_ROOT_DIR"] = str(root.resolve())
    version = subprocess.run([*command, "version"], capture_output=True, text=True, timeout=60, env=environment)
    if version.returncode != 0:
        raise RuntimeError(f"MFA version check failed: {version.stderr}")
    utterances = read_jsonl(project_dir / "utterances.jsonl", Utterance)
    assets = {asset.source_id: asset for asset in read_jsonl(project_dir / "sources" / "manifest.jsonl", SourceAsset)}
    transcribed = [utt for utt in utterances if (utt.revised_text or utt.asr_text)]
    if not transcribed:
        raise ValueError("no utterances with transcripts")
    index_path = project_dir / "alignment" / "index.json"
    old_index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {"signatures": {}}
    lock_path = project_dir / "models.lock.json"
    locked_models = {entry["model_id"]: entry for entry in json.loads(lock_path.read_text(encoding="utf-8")).get("models", [])} if lock_path.exists() else {}
    model_hashes = {}
    for model_id, name, category, suffix in (("japanese_mfa_acoustic", config.alignment.acoustic, "acoustic", ".zip"), ("japanese_mfa_dictionary", config.alignment.dictionary, "dictionary", ".dict")):
        candidate = Path(name)
        model_file = candidate if candidate.is_file() else root / "pretrained_models" / category / (name + suffix)
        if model_file.is_file():
            digest = sha256_file(model_file)
            model_hashes[model_id] = digest
            locked = locked_models.get(model_id)
            if locked and digest not in locked["files"].values():
                raise RuntimeError(f"active {model_id} differs from models.lock.json")
    existing = read_jsonl(project_dir / "phones.jsonl", Phone)
    existing_ids = {phone.utterance_id for phone in existing}
    signatures = {}
    for utt in transcribed:
        pack = language(utt.language_id)
        signatures[utt.utterance_id] = json_hash({
            "source_sha256": assets[utt.source_id].sha256,
            "span": utt.span.model_dump(),
            "text": pack.normalize_text(utt.revised_text or utt.asr_text or "", {}),
            "text_revision": utt.text_revision,
            "language_pack": pack.version,
            "alignment_config": config.alignment.model_dump(),
            "mfa_version": version.stdout.strip(),
            "model_hashes": model_hashes,
            "algorithm_version": "align-4",
        })
    old_success = old_index.get("signatures", {})
    old_failed = old_index.get("failed_signatures", {})
    pending = [utt for utt in transcribed if not (
        (old_success.get(utt.utterance_id) == signatures[utt.utterance_id] and utt.utterance_id in existing_ids)
        or old_failed.get(utt.utterance_id) == signatures[utt.utterance_id]
    )]
    if not pending:
        return {"utterances": 0, "failed": sum(utt.utterance_id in old_failed for utt in transcribed),
                "phones": len(existing), "unknown": sum(phone.unknown for phone in existing), "run_id": "cached"}
    run_id = uuid4().hex
    run_dir = project_dir / "alignment" / run_id
    corpus = run_dir / "corpus"
    output = run_dir / "output"
    output.mkdir(parents=True)
    for utt in pending:
        folder = corpus / utt.speaker_id
        folder.mkdir(parents=True, exist_ok=True)
        source = assets[utt.source_id]
        sf.write(folder / f"{utt.utterance_id}.wav", read_region(project_dir, source, utt.span.start, utt.span.end), 44100, subtype="PCM_16")
        normalized = language(utt.language_id).normalize_text(utt.revised_text or utt.asr_text or "", {})
        (folder / f"{utt.utterance_id}.txt").write_text(normalized, encoding="utf-8")
    argv = [*command, "align", str(corpus), config.alignment.dictionary, config.alignment.acoustic, str(output), "--clean"]
    try:
        completed = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=config.alignment.timeout_s, env=environment)
    except subprocess.TimeoutExpired as exc:
        write_json(run_dir / "error.json", {"kind": "timeout", "argv": argv, "timeout_s": config.alignment.timeout_s})
        raise RuntimeError("MFA alignment timed out") from exc
    (run_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (run_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"MFA failed ({completed.returncode}); see {run_dir / 'stderr.log'}")
    pending_ids = {utt.utterance_id for utt in pending}
    active_ids = {utt.utterance_id for utt in transcribed}
    all_phones: list[Phone] = [phone for phone in existing if phone.utterance_id in active_ids and phone.utterance_id not in pending_ids]
    observed: set[str] = set()
    unknown: set[str] = set()
    failed: list[dict[str, str]] = []
    successful_ids: set[str] = set()
    for utt in pending:
        grid = output / utt.speaker_id / f"{utt.utterance_id}.TextGrid"
        if not grid.exists():
            failed.append({"utterance_id": utt.utterance_id, "reason": "MISSING_TEXTGRID"})
            if "ALIGNMENT_FAILED" not in utt.quality_flags:
                utt.quality_flags.append("ALIGNMENT_FAILED")
            continue
        try:
            intervals = parse_textgrid_phones(grid)
            pack = language(utt.language_id)
            normalized = pack.normalize_phones([item[2] for item in intervals])
            revision = json_hash({"grid_sha256": sha256_file(grid), "text_revision": utt.text_revision, "model": config.alignment.acoustic, "language_pack_version": pack.version, "model_hashes": model_hashes})
            parsed: list[Phone] = []
            for (start_s, end_s, raw), canonical in zip(intervals, normalized, strict=True):
                start = utt.corpus_offset_samples + round(start_s * 44100)
                end = utt.corpus_offset_samples + round(end_s * 44100)
                if not (utt.span.start <= start < end <= utt.span.end + 1):
                    raise ValueError("TextGrid interval outside utterance")
                parsed.append(Phone(utterance_id=utt.utterance_id, source_id=utt.source_id, language_id=utt.language_id, phoneme_id=f"{utt.language_id}@{pack.version}:{canonical}" if canonical else None, ipa=raw if canonical else None, allophone=raw if canonical else None, features={"length": "long" if "ː" in raw else None, "devoiced": "̥" in raw}, raw_model_label=raw, span=SampleSpan(start=start, end=end), alignment_revision=revision, unknown=canonical is None))
        except ValueError as exc:
            failed.append({"utterance_id": utt.utterance_id, "reason": f"INVALID_TEXTGRID: {exc}"})
            if "ALIGNMENT_FAILED" not in utt.quality_flags:
                utt.quality_flags.append("ALIGNMENT_FAILED")
            continue
        all_phones.extend(parsed)
        observed.update(phone.raw_model_label for phone in parsed)
        unknown.update(phone.raw_model_label for phone in parsed if phone.unknown)
        successful_ids.add(utt.utterance_id)
        utt.quality_flags = [flag for flag in utt.quality_flags if flag != "ALIGNMENT_FAILED"]
    failed_ids = {item["utterance_id"] for item in failed}
    write_json(run_dir / "failed_utterances.json", failed)
    if not successful_ids and not all_phones:
        raise RuntimeError(f"MFA produced no usable phone intervals; see {run_dir / 'failed_utterances.json'}")
    write_jsonl(project_dir / "phones.jsonl", all_phones)
    write_jsonl(project_dir / "utterances.jsonl", utterances)
    write_json(index_path, {"schema_version": 1,
                            "signatures": {utt_id: signature for utt_id, signature in signatures.items() if utt_id in successful_ids or (utt_id not in pending_ids and utt_id in existing_ids)},
                            "failed_signatures": {utt_id: signature for utt_id, signature in signatures.items() if utt_id in failed_ids or (utt_id not in pending_ids and utt_id in old_failed)}})
    write_json(run_dir / "phones_observed.json", sorted(observed))
    write_json(run_dir / "unknown_phones.json", sorted(unknown))
    write_json(run_dir / "manifest.json", {"mfa_version": version.stdout.strip(), "argv": argv, "utterances": len(pending), "aligned_utterances": len(successful_ids), "failed_utterances": len(failed), "phones": len(all_phones), "unknown_labels": sorted(unknown)})
    return {"utterances": len(successful_ids), "failed": len(failed), "phones": len(all_phones), "unknown": sum(phone.unknown for phone in all_phones), "run_id": run_id}
