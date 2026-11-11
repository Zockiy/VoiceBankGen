"""Prepare a compact portable edition from a frozen desktop and isolated DML runtime.

The main application and TIFA use one filesystem dependency tree. Only local
application modules are rebuilt; retained frozen dependencies keep their code
objects and native libraries together. All output paths must be new.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
import marshal
import os
from pathlib import Path
import shutil
import sys

from PyInstaller.archive.readers import CArchiveReader
from PyInstaller.archive.writers import CArchiveWriter, ZlibArchiveWriter

ROOT = Path(__file__).resolve().parents[1]
# Export/conversion and optional dataframe/task frameworks are not used by the
# desktop inference/editing workflows. Inference uses onnxruntime, not onnx.
BUILD_ONLY_ROOTS = {"onnx", "onnxscript", "onnx_ir", "ml_dtypes", "pandas", "dask"}
sys.path.insert(0, str(ROOT / "src"))
from utau_forge.resource_download import component_files, destination, validate_catalog
from build_single_exe import bundle, release_file
from prepare_download_catalog import file_record


def runtime_file(path: Path, relative: Path) -> bool:
    if any(part in {"__pycache__", "tests", "test", ".git"} for part in relative.parts):
        return False
    parts = relative.parts
    if parts[:2] == ("Lib", "site-packages"):
        parts = parts[2:]
    if parts and parts[0] in {"bin", "share"}:
        return False  # Generated console tools and manpages are not called.
    if parts[:2] in {("torch", "include"), ("torch", "share"), ("torch", "bin")}:
        return False
    if path.suffix.lower() in {".lib", ".pdb", ".exp"}:
        return False  # Static link and debug artifacts are unused for inference.
    if path.suffix.lower() in {".pyc", ".pyo", ".wav", ".ogg", ".flac", ".mp3"}:
        return False
    return release_file(relative.as_posix())


def link_file(source: Path, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def shared_roots(site: Path) -> set[str]:
    return {path.stem if path.is_file() else path.name for path in site.iterdir()
            if not path.name.endswith((".dist-info", ".data", ".libs"))
            and path.name not in {"bin", "share", "__pycache__"}}


def rebuild_pyz(gui: Path, work: Path, shared: set[str], excluded: set[str] | None = None) -> Path:
    source = CArchiveReader(str(gui)).open_embedded_archive("PYZ.pyz")
    entries, code = {}, {}
    removed = Counter()
    for name, row in source.toc.items():
        top = name.split(".", 1)[0]
        if top in shared or top in (excluded or set()):
            removed[top] += 1
            continue
        kind = row[0]
        if kind not in {0, 1, 3}:
            raise ValueError(f"Unsupported frozen entry {name}: {kind}")
        filename = None if kind == 3 else (name.replace(".", "/") + ("/__init__.py" if kind == 1 else ".py"))
        entries[name] = (name, filename, "PYMODULE")
        if kind != 3:
            code[name] = source.extract(name)
    source_root = ROOT / "src"
    for path in (source_root / "utau_forge").rglob("*.py"):
        if any(part in {".history", "__pycache__"} for part in path.parts):
            continue
        relative = path.relative_to(source_root).with_suffix("")
        parts = relative.parts[:-1] if relative.name == "__init__" else relative.parts
        name = ".".join(parts)
        entries[name] = (name, str(path), "PYMODULE")
        code[name] = compile(path.read_text(encoding="utf-8-sig"), str(path), "exec")
    target = work / "desktop.pyz"
    ZlibArchiveWriter(str(target), list(entries.values()), code_dict=code)
    (work / "shared-modules.json").write_text(json.dumps(dict(removed), indent=2), encoding="utf-8")
    print("Rebuilt desktop PYZ", len(entries), "modules; shared", sum(removed.values()), flush=True)
    return target


def rebuild_executable(source: Path, target: Path, pyz: Path, work: Path):
    reader = CArchiveReader(str(source))
    script_names = [name for name, row in reader.toc.items() if row[4] == "s"]
    entry = script_names[-1]
    script = marshal.dumps(compile((ROOT / "packaging/desktop_entry.py").read_text(encoding="utf-8"), "desktop_entry.py", "exec"))
    early_path = marshal.dumps(compile(
        "import os, sys\n"
        "sys.path.insert(0, os.path.join(os.path.dirname(sys.executable), "
        "'runtime', 'tifa-cpu', 'Lib', 'site-packages'))\n",
        "vbg_shared_runtime.py", "exec"))

    class Writer(CArchiveWriter):
        def _write_entry(self, stream, item):
            name, _, compress, kind = item
            if kind == "o":
                return self._write_blob(stream, b"", name, kind)
            blob = (early_path if name == "vbg_shared_runtime" else
                    pyz.read_bytes() if name == "PYZ.pyz" else
                    script if name == entry else reader.extract(name))
            return self._write_blob(stream, blob, name, kind, compress=compress)

    archive = work / (target.stem + ".pkg")
    entries = []
    for name, row in reader.toc.items():
        entries.append((name, "", row[3], row[4]))
        if name == "pyiboot01_bootstrap" and "vbg_shared_runtime" not in reader.toc:
            # Runtime hooks can import shared dependencies before desktop_entry.
            entries.append(("vbg_shared_runtime", "", True, "s"))
    Writer(str(archive), entries + [(name, "", False, "o") for name in reader.options], "python311.dll")
    with source.open("rb") as stream:
        prefix = stream.read(reader._start_offset)
    with target.open("xb") as stream:
        stream.write(prefix)
        with archive.open("rb") as payload:
            shutil.copyfileobj(payload, stream)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--onnx-model", type=Path, help="Replace the old Whisper weights with the pinned ONNX model")
    parser.add_argument("--onnx-runtime", type=Path, help="Extracted onnxruntime-directml wheel (site-packages layout)")
    args = parser.parse_args()
    base, runtime, output = args.base.resolve(), args.runtime.resolve(), args.output.resolve()
    if bool(args.onnx_model) != bool(args.onnx_runtime):
        parser.error("--onnx-model and --onnx-runtime must be supplied together")
    if args.onnx_model:
        from utau_forge.onnx_whisper import missing_files
        if missing_files(args.onnx_model) or not (args.onnx_runtime / "onnxruntime/capi/DirectML.dll").is_file():
            parser.error("Incomplete ONNX Whisper model or DirectML runtime")
    if output.exists():
        raise FileExistsError("Use a fresh output directory")
    site = runtime / "Lib/site-packages"
    if not (runtime / "python.exe").is_file() or not (site / "torch_directml/DirectML.dll").is_file():
        raise ValueError("A complete embedded DirectML runtime is required")
    if not (site / "fugashi/libmecab.dll").is_file():
        raise ValueError("Missing fugashi wheel data: install .data/data/Lib/site-packages/fugashi/libmecab.dll into runtime/Lib/site-packages/fugashi")
    output.mkdir(parents=True)
    package = output / "VoiceBankGen"
    package.mkdir()
    work = output / "build-records"
    work.mkdir()
    previous = validate_catalog(json.loads((base / "components.json").read_text(encoding="utf-8")))
    shared = shared_roots(site)
    shared_trees = {p.name for p in site.iterdir() if p.is_dir()}
    saved = 0
    for component in previous["components"]:
        if component["id"] == "tifa-gpu" or component["id"] == "tifa-cpu":
            continue
        for item in component_files(component):
            name = item["path"]
            if "content" in item or not release_file(name):
                continue
            if args.onnx_model and name.startswith("resources/engine/"):
                continue
            if name == "resources/unidic/model.bin":
                # CRF training model; MeCab inference uses sys.dic, matrix.bin,
                # char.bin, unk.dic and dicrc. Keep the complete runtime dictionary.
                continue
            parts = Path(name).parts
            if len(parts) > 1 and parts[0] == "_internal" and any(
                    parts[1] == root or parts[1].startswith((root + "-", root + ".")) for root in BUILD_ONLY_ROOTS):
                saved += item["size"]
                continue
            if name in {"VoiceBankGen.exe", "VoiceBankGenWorker.exe", "使用说明.txt", "第三方许可说明.txt", "LICENSE"}:
                continue
            if name.startswith(("_internal/utau_forge/data/", "LICENSES/")):
                continue
            if len(parts) > 1 and parts[0] == "_internal" and parts[1] in shared_trees:
                saved += item["size"]
                continue
            source = destination(base, name)
            if source.stat().st_size != item["size"]:
                raise ValueError("Base file size differs from its catalog: " + name)
            link_file(source, destination(package, name))
    for path in sorted(runtime.rglob("*")):
        relative = path.relative_to(runtime)
        if args.onnx_runtime and len(relative.parts) > 2 and relative.parts[:2] == ("Lib", "site-packages") and relative.parts[2].startswith("onnxruntime"):
            continue
        if path.is_file() and runtime_file(path, relative):
            link_file(path, package / "runtime/tifa-cpu" / relative)
    if args.onnx_runtime:
        for path in args.onnx_runtime.rglob("*"):
            relative = path.relative_to(args.onnx_runtime)
            if path.is_file() and runtime_file(path, relative):
                link_file(path, package / "runtime/tifa-cpu/Lib/site-packages" / relative)
    if args.onnx_model:
        from utau_forge.onnx_whisper import MODEL_FILES
        import hashlib
        target = package / "resources/engine/models/whisper-small-onnx"
        records = []
        for name in MODEL_FILES:
            path = args.onnx_model / name
            link_file(path, target / name)
            records.append(file_record(path, name))
        hashes = {r["path"]: r["sha256"] for r in records}
        aggregate = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
        lock = {"schema_version": 1, "models": [{"provider": "HuggingFace", "model_id": "onnx-community/whisper-small",
                "backend": "onnx_whisper", "revision": "36050c46d777d46dc4b5f43f6d90574fc38f8732",
                "files": hashes, "sha256": aggregate, "local_relative_path": "models/whisper-small-onnx",
                "source_url": "https://huggingface.co/onnx-community/whisper-small", "license": "apache-2.0",
                "required_sample_rate": 16000, "device": "directml", "precision": "fp32",
                "adaptation": "split-decoder-fp32-compute-fp16-storage-v1"}]}
        (package / "resources/engine/models.lock.json").write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
    # Runtime code/source and dictionary configuration are required by TIFA.
    for path in (base / "resources/upstream/TIFA").rglob("*"):
        if path.is_file() and release_file(path.relative_to(base).as_posix()):
            target = package / path.relative_to(base)
            if not target.exists():
                link_file(path, target)
    for path in (ROOT / "src/utau_forge/data").rglob("*"):
        if path.is_file() and release_file(path.relative_to(ROOT / "src/utau_forge/data").as_posix()):
            target = package / "_internal/utau_forge/data" / path.relative_to(ROOT / "src/utau_forge/data")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)  # Source inputs may be edited after a build.
    bridge = package / "_internal/utau_forge/data/runtime/tifa_entry.py"
    bridge.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "packaging/tifa_entry.py", bridge)
    for path in (ROOT / "LICENSES").rglob("*"):
        if path.is_file() and release_file(path.relative_to(ROOT / "LICENSES").as_posix()):
            target = package / "LICENSES" / path.relative_to(ROOT / "LICENSES")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    shutil.copy2(ROOT / "LICENSE", package / "LICENSE")
    shutil.copy2(ROOT / "packaging/PORTABLE_QUICK_START.txt", package / "使用说明.txt")
    shutil.copy2(ROOT / "packaging/RELEASE_NOTICES.txt", package / "第三方许可说明.txt")
    pyz = rebuild_pyz(base / "VoiceBankGen_UI_Fork.exe" if (base / "VoiceBankGen_UI_Fork.exe").is_file() else base / "VoiceBankGen.exe", work, shared, BUILD_ONLY_ROOTS)
    rebuild_executable(base / "VoiceBankGen.exe", package / "VoiceBankGen.exe", pyz, work)
    rebuild_executable(base / "VoiceBankGenWorker.exe", package / "VoiceBankGenWorker.exe", pyz, work)
    bad = [str(p.relative_to(package)) for p in package.rglob("*.dll")
           if any(key in p.name.lower() for key in ("torch_cuda", "cudnn", "cublas", "cudart", "nvrtc"))]
    if bad:
        raise ValueError("Unexpected CUDA libraries: " + repr(bad))
    print("Preparing fixed portable component records", flush=True)
    components = []
    groups = [("app-core", "VoiceBankGen 主程序", [p for p in package.rglob("*") if p.is_file()
               and p.relative_to(package).parts[0] not in {"runtime", "resources"}]),
              ("tifa-cpu", "DirectML 共享运行环境", [p for p in (package / "runtime/tifa-cpu").rglob("*") if p.is_file()]
               + [p for p in (package / "resources/upstream").rglob("*") if p.is_file()]),
              ("ffmpeg", "FFmpeg", [p for p in (package / "runtime/ffmpeg").rglob("*") if p.is_file()]),
              ("models", "语音识别、对齐模型与词典", [p for p in (package / "resources").rglob("*") if p.is_file()
               and "upstream" not in p.relative_to(package / "resources").parts])]
    for identifier, name, paths in groups:
        if paths:
            components.append(bundle(identifier, name, [(p, p.relative_to(package).as_posix()) for p in paths], output, version="2026.11.05-directml"))
    catalog = validate_catalog({"schema": 1, "version": "2026.11.05-directml", "edition": "portable", "components": components})
    (output / "install-catalog.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    (package / "components.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    (work / "preparation.json").write_text(json.dumps({"package": str(package), "removed_duplicate_native_bytes": saved,
        "shared_roots": sorted(shared), "source_branch": "codex/directml"}, indent=2), encoding="utf-8")
    print("PACKAGE", package, "removed duplicate native MiB", round(saved / 1024**2, 1), flush=True)


if __name__ == "__main__":
    main()
