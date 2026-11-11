"""Generate pinned upstream downloads from the resources verified in this release."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def file_record(path: Path, relative: str, member: str | None = None) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    result = {"path": relative, "size": path.stat().st_size, "sha256": digest.hexdigest()}
    if member:
        result["member"] = member
    return result


def generated(relative: str, content: str) -> dict:
    data = content.encode("utf-8")
    return {"path": relative, "size": len(data), "sha256": hashlib.sha256(data).hexdigest(), "content": content}


def prepare(package: Path) -> dict:
    resources = package / "resources"
    lock = json.loads((resources / "tifa.lock.json").read_text(encoding="utf-8"))
    tifa, unidic = [], []
    for name, expected in lock["files"].items():
        item = file_record(resources / name, "resources/" + name, name.split("/", 1)[1])
        if item["sha256"] != expected:
            raise ValueError(f"Resource lock differs: {name}")
        if name.startswith("tifa/"):
            tifa.append(item)
        elif name != "unidic/mecabrc":
            # UniDic's upstream archive does not include the empty local mecabrc.
            unidic.append(item)
    models = json.loads((resources / "engine/models.lock.json").read_text(encoding="utf-8"))
    asr = next(item for item in models["models"] if item["model_id"] == "Systran/faster-whisper-small")
    artifacts = []
    for name, expected in asr["files"].items():
        relative = "resources/engine/" + asr["local_relative_path"] + "/" + name
        item = file_record(package / relative, relative)
        if item["sha256"] != expected:
            raise ValueError(f"Whisper lock differs: {name}")
        artifacts.append({"url": f"https://huggingface.co/{asr['model_id']}/resolve/{asr['revision']}/{name}",
                          "cache": "whisper-small/" + name, "format": "raw", "size": item["size"],
                          "sha256": expected, "files": [item]})
    return {"schema": 1, "version": "2026.10.07", "components": [
        {"id": "tifa-model", "name": "TIFA 多语言对齐模型", "version": "1.0.0", "required": True,
         "description": "日语与普通话对齐；官方 GitHub。模型 CC BY-NC-SA 4.0。", "artifacts": [
             {"url": "https://github.com/openvpi/TIFA/releases/download/v1.0.0/TIFA-1.0-ST.zip",
              "cache": "TIFA-1.0-ST.zip", "format": "zip", "size": 150789526,
              "sha256": "6da6832cd2cb981aae1ccc2d9330f8f7c085ce7977d1eb13bff63531d038a88b", "files": tifa}],
         "generated": [generated("resources/tifa.lock.json", json.dumps(lock, ensure_ascii=False, indent=2) + "\n")]},
        {"id": "unidic", "name": "UniDic 日语词典", "version": "2.3.0", "required": True,
         "description": "日语读音分析；维护者官方 GitHub。BSD 许可。", "artifacts": [
             {"url": "https://github.com/polm/unidic-py/releases/download/v1.0.2/unidic.zip",
              "cache": "unidic-2.3.0.zip", "format": "zip", "size": 608180035, "files": unidic}],
         "generated": [generated("resources/unidic/mecabrc", "")]},
        {"id": "whisper-small", "name": "Whisper small 转写模型", "version": asr["revision"], "required": True,
         "description": "日语与普通话转写；Systran 官方 Hugging Face，未在 GitHub 发布权重。MIT 许可。",
         "artifacts": artifacts, "generated": [generated("resources/engine/models.lock.json",
             json.dumps({"schema_version": 1, "models": [asr]}, ensure_ascii=False, indent=2) + "\n")]}
    ]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "src/utau_forge/data/resource-catalog.json")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        history = args.output.parent / ".history"
        history.mkdir(exist_ok=True)
        from uuid import uuid4
        args.output.replace(history / (args.output.name + "." + uuid4().hex))
    args.output.write_text(json.dumps(prepare(args.package), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)
