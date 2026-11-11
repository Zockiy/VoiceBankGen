"""Remember explicit paths; default model files live in the software package."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

from .desktop_store import user_data_dir
from .storage import sha256_file, write_json


MODEL_FIELDS = (
    ("asr", "Whisper 转写模型目录", "directory", "可选本项目准备的 ONNX Whisper（支持 DirectML）或 faster-whisper（CPU）。ONNX 目录包含 voicebankgen-onnx.json、config.json、generation_config.json、tokenizer.json 及 onnx/ 下的 encoder、两份解码图和共用 decoder.weights；旧格式包含 model.bin、config.json、tokenizer.json。"),
    ("tifa", "TIFA 模型目录", "directory", "包含 model.pt、config.yaml、vocabulary.json 和 dictionaries/japanese_dict_full.txt 的完整模型目录；普通话还需中文词表与 dictionaries/ds-zh-pinyin-lite.txt。"),
    ("unidic", "UniDic 日语词典目录", "directory", "包含 sys.dic、unk.dic、char.bin、matrix.bin、dicrc、mecabrc 的词典目录。"),
    ("mfa_acoustic", "MFA 声学模型文件", "file", "选择日语声学模型 ZIP；使用 TIFA 时不需要此项。"),
    ("mfa_dictionary", "MFA 发音词典文件", "file", "选择日语发音词典 .dict；使用 TIFA 时不需要此项。"),
)
REQUIRED_FILES = {
    "asr": ("model.bin", "config.json", "tokenizer.json"),
    "tifa": ("model.pt", "config.yaml", "vocabulary.json", "dictionaries/japanese_dict_full.txt"),
    "unidic": ("sys.dic", "unk.dic", "char.bin", "matrix.bin", "dicrc", "mecabrc"),
}


def missing_asr_files(path: Path) -> list[str]:
    from .onnx_whisper import is_onnx_model, missing_files
    if is_onnx_model(path) or (path / "onnx").is_dir():
        return missing_files(path)
    return [name for name in REQUIRED_FILES["asr"] if not (path / name).is_file()]


def asr_record(record: dict) -> bool:
    return "faster-whisper" in record["model_id"] or record.get("backend") == "onnx_whisper"


def load_model_paths() -> dict[str, str]:
    path = user_data_dir() / "model-paths.json"
    values = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return {key: str(values.get(key, "")) for key, *_ in MODEL_FIELDS}


def validate_model_paths(values: dict) -> dict[str, str]:
    normalized = {}
    for key, label, kind, _ in MODEL_FIELDS:
        value = str(values.get(key, "")).strip()
        if not value:
            normalized[key] = ""
            continue
        path = Path(value).expanduser()
        if not path.is_absolute():
            raise ValueError(f"{label}：请输入绝对路径")
        path = path.resolve()
        if kind == "directory":
            missing = missing_asr_files(path) if key == "asr" else [name for name in REQUIRED_FILES[key] if not (path / name).is_file()]
            if missing:
                raise ValueError(f"{label}：缺少 {', '.join(missing)}")
        elif not path.is_file() or path.suffix.lower() != (".zip" if key == "mfa_acoustic" else ".dict"):
            raise ValueError(f"{label}：请选择有效的 {'ZIP' if key == 'mfa_acoustic' else '.dict'} 文件")
        normalized[key] = str(path)
    return normalized


def save_model_paths(values: dict) -> dict[str, str]:
    normalized = validate_model_paths(values)
    write_json(user_data_dir() / "model-paths.json", normalized)
    return normalized


def resources_directory(root: Path) -> Path:
    # Source development shares the installed portable model store. Frozen
    # packages use their own resources directory and remain movable as a unit.
    portable = root / "dist/VoiceBankGen/resources"
    return portable if portable.is_dir() else root / "resources"


def bundled_model_lock(root: Path) -> dict:
    bundle = resources_directory(root) / "engine"
    path = bundle / "models.lock.json"
    lock = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"schema_version": 1, "models": []}
    for record in lock.get("models", []):
        relative = record.get("local_relative_path")
        if relative:
            directory = (bundle / relative).resolve()
            if not directory.is_relative_to(bundle.resolve()):
                raise ValueError("软件模型清单路径超出资源目录")
            record["local_absolute_path"] = str(directory)
            record.pop("local_relative_path")
    return lock


def detect_model_paths(root: Path, project: Path | None = None) -> dict[str, str]:
    detected = {key: "" for key, *_ in MODEL_FIELDS}
    preferred = ""
    if project and (project / "config.yaml").is_file():
        from .config import read_config
        preferred = read_config(project / "config.yaml").asr.model
    # Project configuration selects a model, but never supplies model files.
    # Explicit user paths still override these defaults at task launch.
    for lock in [bundled_model_lock(root)]:
        records = lock.get("models", [])
        records = sorted(records, key=lambda r: not (r["model_id"] == preferred or r["model_id"].endswith("faster-whisper-" + preferred)))
        for record in records:
            path = Path(record["local_absolute_path"])
            model_id = record["model_id"]
            if asr_record(record):
                key, candidate = "asr", path
                valid = not missing_asr_files(candidate)
            elif model_id in {"japanese_mfa_acoustic", "japanese_mfa_dictionary"}:
                key = "mfa_acoustic" if model_id.endswith("acoustic") else "mfa_dictionary"
                candidate = path / ("japanese_mfa.zip" if key == "mfa_acoustic" else "japanese_mfa.dict")
                valid = candidate.is_file()
            else:
                continue
            if valid and not detected[key]:
                detected[key] = str(candidate.resolve())
    resources = resources_directory(root)
    for key, path in (("tifa", resources / "tifa/TIFA-1.0-ST"), ("unidic", resources / "unidic")):
        if all((path / name).is_file() for name in REQUIRED_FILES[key]):
            detected[key] = str(path.resolve())
    return detected


def apply_engine_model_paths(config, lock: dict, values: dict) -> None:
    """Use external files directly; pin their hashes in this task's engine lock."""
    values = validate_model_paths(values)
    if values["asr"]:
        path = Path(values["asr"])
        files = {p.relative_to(path).as_posix(): sha256_file(p) for p in sorted(path.rglob("*")) if p.is_file()}
        record = {"model_id": "VoiceBankGen/local-asr", "local_absolute_path": str(path), "files": files, "sha256": hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()}
        config.asr.model = record["model_id"]
        from .onnx_whisper import is_onnx_model
        record["backend"] = "onnx_whisper" if is_onnx_model(path) else "faster_whisper"
        config.asr.backend = record["backend"]
        lock["models"] = [r for r in lock.get("models", []) if r["model_id"] != record["model_id"]] + [record]
    for key, attribute, model_id in (("mfa_acoustic", "acoustic", "japanese_mfa_acoustic"), ("mfa_dictionary", "dictionary", "japanese_mfa_dictionary")):
        if values[key]:
            path = Path(values[key])
            setattr(config.alignment, attribute, str(path))
            record = {"model_id": model_id, "files": {path.name: sha256_file(path)}}
            lock["models"] = [r for r in lock.get("models", []) if r["model_id"] != model_id] + [record]


def tifa_resources(root: Path, values: dict) -> tuple[Path, Path, dict]:
    values = validate_model_paths(values)
    resources = resources_directory(root)
    model_dir = Path(values["tifa"]) if values["tifa"] else resources / "tifa/TIFA-1.0-ST"
    dictionary = Path(values["unidic"]) if values["unidic"] else resources / "unidic"
    model = model_dir / "model.pt"
    lock_path = resources / "tifa.lock.json"
    if not lock_path.is_file() and not (values["tifa"] and values["unidic"]):
        raise RuntimeError("缺少自动识别的 TIFA 资源锁；请在模型设置中指定完整 TIFA 和 UniDic 目录")
    expected = json.loads(lock_path.read_text(encoding="utf-8"))["files"] if lock_path.is_file() else {}
    files = {}
    for key, directory, prefix in (("tifa", model_dir, "tifa/TIFA-1.0-ST/"), ("unidic", dictionary, "unidic/")):
        names = list(REQUIRED_FILES[key]) if values[key] else [name.removeprefix(prefix) for name in expected if name.startswith(prefix)]
        if not names:
            raise RuntimeError(f"缺少 {key} 模型资源")
        for name in names:
            path = directory / name
            if not path.is_file():
                raise RuntimeError(f"模型资源缺失：{path}")
            digest = sha256_file(path)
            if not values[key] and digest != expected[prefix + name]:
                raise RuntimeError(f"模型资源哈希不匹配：{path}")
            files[prefix + name] = digest
    return model, dictionary, {"files": files, "model_sha256": files["tifa/TIFA-1.0-ST/model.pt"]}
