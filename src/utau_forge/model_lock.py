"""Content-addressed local model inventory; no placeholder hashes."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from .storage import sha256_file, write_json


def lock_local_model(project_dir: Path, source_dir: Path, *, provider: str, model_id: str, revision: str, license_id: str, source_url: str, sample_rate: int, device: str, precision: str) -> dict:
    source_dir = source_dir.resolve(strict=True)
    files = [source_dir] if source_dir.is_file() else sorted(path for path in source_dir.rglob("*") if path.is_file())
    if not files:
        raise ValueError("model source is empty")
    identity = hashlib.sha256(f"{provider}:{model_id}:{revision}".encode()).hexdigest()[:16]
    target = project_dir / "models" / identity
    if not target.exists():
        if source_dir.is_file():
            target.mkdir(parents=True)
            shutil.copy2(source_dir, target / source_dir.name)
        else:
            shutil.copytree(source_dir, target)
    source_names = {path.name if source_dir.is_file() else path.relative_to(source_dir).as_posix() for path in files}
    target_names = {path.relative_to(target).as_posix() for path in target.rglob("*") if path.is_file()}
    if source_names != target_names:
        raise ValueError("model copy is incomplete; inspect the existing target before retrying")
    hashes = {path.relative_to(target).as_posix(): sha256_file(path) for path in sorted(target.rglob("*")) if path.is_file()}
    if any(sha256_file(source_dir if source_dir.is_file() else source_dir / name) != digest for name, digest in hashes.items()):
        raise ValueError("model copy hash mismatch")
    aggregate = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    record = {"provider": provider, "model_id": model_id, "revision": revision, "sha256": aggregate, "files": hashes, "license": license_id, "source_url": source_url, "local_relative_path": target.relative_to(project_dir).as_posix(), "required_sample_rate": sample_rate, "device": device, "precision": precision}
    lock_path = project_dir / "models.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["models"] = [entry for entry in lock["models"] if not (entry["provider"] == provider and entry["model_id"] == model_id)] + [record]
    lock["status"] = "partial"  # Every required model must be checked before environment-ready status.
    write_json(lock_path, lock)
    return {"model_id": model_id, "revision": revision, "sha256": aggregate, "files": len(hashes), "local_relative_path": record["local_relative_path"]}


def verify_model(project_dir: Path, record: dict) -> bool:
    external = record.get("local_absolute_path")
    directory = Path(external).resolve() if external else (project_dir / record["local_relative_path"]).resolve()
    if external and not Path(external).is_absolute():
        raise ValueError("external model path must be absolute")
    if not external and not directory.is_relative_to(project_dir.resolve()):
        raise ValueError("model path escapes project")
    if not directory.is_dir() or {path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file()} != set(record["files"]):
        return False
    hashes = {name: sha256_file(directory / name) for name in sorted(record["files"])}
    aggregate = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return hashes == record["files"] and aggregate == record["sha256"]
