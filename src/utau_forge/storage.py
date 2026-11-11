"""UTF-8 JSON/JSONL storage and atomic stage output."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, TypeVar
from uuid import uuid4

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(value: Any) -> str:
    body = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def atomic_bytes(path: Path, contents: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid4().hex}.pending")
    try:
        with temporary.open("wb") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            history = path.parent / ".history"
            history.mkdir(exist_ok=True)
            os.replace(path, history / f"{path.name}.{uuid4().hex}")
        os.replace(temporary, path)
    except BaseException:
        # Preserve any incomplete file for inspection; it is never treated as committed.
        raise


def write_json(path: Path, value: BaseModel | Any) -> None:
    data = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    atomic_bytes(path, (json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def read_json(path: Path, cls: type[T]) -> T:
    return cls.model_validate_json(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, values: Iterable[BaseModel | dict[str, Any]]) -> None:
    lines = []
    for value in values:
        data = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
        lines.append(json.dumps(data, ensure_ascii=False, sort_keys=True))
    atomic_bytes(path, (("\n".join(lines) + "\n") if lines else "").encode("utf-8"))


def read_jsonl(path: Path, cls: type[T]) -> list[T]:
    if not path.exists():
        return []
    return [cls.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
