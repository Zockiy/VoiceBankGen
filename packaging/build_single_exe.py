"""Build exactly two editions: complete offline and core-only online setup.

Large compressed payload is appended as ZIP, outside PyInstaller's extraction
archive. The bootloader locates its own cookie backwards; no bootloader patch.
Only the small Tk launcher is extracted before the visible progress window.
"""
from __future__ import annotations
import argparse
import copy
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

from prepare_download_catalog import file_record

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from utau_forge.resource_download import component_files, component_ready, digest, unpack, validate_catalog


def bundle(identifier, name, files, output, *, required=True, version="2026.10.07"):
    archive = output / (identifier + ".zip")
    if archive.exists():
        raise FileExistsError(archive)
    records = []
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as zipped:
        for source, relative in files:
            records.append(file_record(source, relative))
            zipped.write(source, relative)
    archive_record = file_record(archive, archive.name)
    print(f"{name}: {archive_record['size'] / 1024**2:.1f} MB / {len(records)} files", flush=True)
    return {"id": identifier, "name": name, "required": required, "version": version,
            "probe_files": [item["path"] for item in records if item["path"].endswith((".exe", "/python311.dll", "/base_library.zip", "/torch/lib/torch_cpu.dll", "/PySide6/Qt6Core.dll"))],
            "description": "随单文件 EXE 提供，无需 Python / Conda；安装后可从本地缓存修复。",
            "artifacts": [{"cache": archive.name, "format": "zip", "size": archive_record["size"],
                           "sha256": archive_record["sha256"], "files": records}]}


def tree(package, relative):
    directory = package / relative
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    return [(path, path.relative_to(package).as_posix()) for path in sorted(directory.rglob("*")) if path.is_file()
            and release_file(path.relative_to(package).as_posix())]


def release_file(name):
    path = Path(name)
    if any(part in {"__pycache__", ".git", ".history"} for part in path.parts):
        return False
    legal = any(word in path.name.lower() for word in ("license", "copying", "notice", "copyright"))
    if path.suffix.lower() == ".md" and not legal:
        return False
    if name.startswith("licenses/"):
        if name.startswith("licenses/GPU/"):
            return False  # Historical CUDA notices do not describe this fork.
        if path.suffix.lower() in {".dll", ".exe", ".pyd"}:
            return False
        if path.name in {"python-packages.json", "MFA-runtime-packages.json"}:
            return False
    return True


def cached_component(component, cache, output, *, expected_model=None):
    """Reuse verified local payloads; repack only when removing internal docs."""
    result = copy.deepcopy(component)
    expected = {f["path"]: f for f in component_files(expected_model)} if expected_model else {}
    for artifact in result["artifacts"]:
        source = cache / artifact["cache"]
        if digest(source) != artifact["sha256"] or source.stat().st_size != artifact["size"]:
            raise ValueError("Cached component checksum differs: " + source.name)
        kept = [f for f in artifact["files"] if release_file(f["path"])]
        actual = {f["path"]: f for f in kept}
        if expected:
            for name, record in expected.items():
                if name not in actual or ("content" not in record and
                        any(actual[name][key] != record[key] for key in ("size", "sha256"))):
                    raise ValueError("Cached model differs from pinned source: " + name)
        replacements = {name: f for name, f in expected.items() if "content" in f
                        and actual[name]["sha256"] != f["sha256"]}
        target = output / source.name
        if len(kept) == len(artifact["files"]) and not replacements:
            # A hardlink avoids duplicating gigabytes of immutable cache data.
            try:
                import os
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)
        else:
            with zipfile.ZipFile(source) as old, zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as new:
                for record in kept:
                    name = record["path"]
                    if name in replacements:
                        value = replacements[name]
                        new.writestr(name, value["content"].encode("utf-8"))
                        record.update(size=value["size"], sha256=value["sha256"])
                    else:
                        with old.open(record.get("member", name)) as src, new.open(name, "w", force_zip64=True) as dst:
                            shutil.copyfileobj(src, dst, 1024 * 1024)
                        record.pop("member", None)
        archive_record = file_record(target, target.name)
        artifact.update(cache=target.name, size=archive_record["size"], sha256=archive_record["sha256"], files=kept)
        artifact.pop("url", None)
        print(f"Cache {result['id']}: {archive_record['size'] / 1024**2:.1f} MB", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--desktop-build", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--variants", choices=("both", "full", "online"), default="both")
    parser.add_argument("--version", default="2026.10.07", help="Release and application component version")
    parser.add_argument("--runtime-assets", type=Path, default=ROOT / "build/network-runtime-assets")
    parser.add_argument("--component-cache", type=Path, help="Reuse verified payloads from package/components.json")
    parser.add_argument("--release-notices", type=Path, help="Additional exact-version legal texts and source references")
    parser.add_argument("--runtime-catalog", type=Path, default=ROOT / "src/utau_forge/data/runtime-catalog.json")
    parser.add_argument("--acceleration", choices=("directml", "cpu"), default="directml")
    args = parser.parse_args()
    runtimes = validate_catalog(json.loads(args.runtime_catalog.read_text(encoding="utf-8")))
    runtime = next(c for c in runtimes["components"] if c["id"] == "tifa-cpu")
    has_directml = any("torch_directml/" in f["path"] for f in component_files(runtime))
    if has_directml != (args.acceleration == "directml"):
        parser.error("Runtime catalog does not match --acceleration; generate a fresh catalog with prepare_runtime_catalog.py --backend directml (or cpu)")
    for component in runtimes["components"]:
        if component["id"] == "tifa-gpu" or any("torch_cuda.dll" in f["path"].lower() for f in component_files(component)):
            parser.error("CUDA components are not supported by the DirectML fork")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = (args.output or ROOT / "dist" / ("single_exe_" + stamp)).resolve()
    output.mkdir(parents=True, exist_ok=False)
    package = args.package.resolve()
    core = tree(package, "_internal")
    # Cached dependency files remain, but all application data come from source.
    core = [(path, name) for path, name in core if not name.startswith("_internal/utau_forge/data/")]
    core = [(path, name) for path, name in core if path.suffix not in {".onnx", ".jit", ".pt"}]
    core += [(path, "_internal/utau_forge/data/" + path.relative_to(ROOT / "src/utau_forge/data").as_posix())
             for path in sorted((ROOT / "src/utau_forge/data").rglob("*")) if path.is_file()
             and not any(part in {".history", "__pycache__"} for part in path.relative_to(ROOT / "src/utau_forge/data").parts)]
    core += [(args.desktop_build / name, name) for name in ("VoiceBankGen.exe", "VoiceBankGenWorker.exe")]
    core += [(ROOT / "packaging/tifa_entry.py", "_internal/utau_forge/data/runtime/tifa_entry.py")]
    # Use the same public documents and license scope as the source snapshot;
    # local development manuals may contain private paths or test records.
    core += [(ROOT / "LICENSE", "LICENSE"),
             (ROOT / "packaging/RELEASE_QUICK_START.txt", "使用说明.txt"),
             (ROOT / "packaging/RELEASE_NOTICES.txt", "第三方许可说明.txt"),
             (ROOT / "src/utau_forge/data/zh_inventory_LICENSE.txt", "LICENSES/zh_inventory_LICENSE.txt")]
    core += tree(ROOT, "LICENSES")
    core += tree(package, "licenses")
    if args.release_notices:
        core += [(path, "licenses/" + path.relative_to(args.release_notices).as_posix())
                 for path in sorted(args.release_notices.rglob("*")) if path.is_file()]
    catalog = json.loads((ROOT / "src/utau_forge/data/resource-catalog.json").read_text(encoding="utf-8"))
    app = bundle("app-core", "VoiceBankGen 主程序", core, output, version=args.version)
    editions = ("full", "online") if args.variants == "both" else (args.variants,)
    checksums = []
    for edition in editions:
        if edition == "full":
            if args.component_cache:
                cached = validate_catalog(json.loads((package / "components.json").read_text(encoding="utf-8")))
                models = {c["id"]: c for c in catalog["components"]}
                wanted = ("tifa-cpu", "ffmpeg", "vad-model", "tifa-model", "unidic", "whisper-small")
                by_id = {c["id"]: c for c in cached["components"]}
                cached_dml = any("torch_directml/" in f["path"] for f in component_files(by_id["tifa-cpu"]))
                if cached_dml != has_directml:
                    raise ValueError("Cached TIFA runtime does not match the requested CPU/DirectML profile")
                components = [app] + [cached_component(by_id[key], args.component_cache, output,
                                                      expected_model=models.get(key)) for key in wanted]
            else:
                components = full_components(app, package, output, runtimes, catalog, args.runtime_assets)
        else:
            components = [app] + runtimes["components"] + catalog["components"]
        build_installer(components, edition, args.version, output, stamp)
        executable = output / ("VoiceBankGen-" + ("Full" if edition == "full" else "Online") + "-Setup.exe")
        archive_record = file_record(executable, executable.name)
        checksums.append(archive_record["sha256"] + "  " + executable.name)
    (output / "SHA256SUMS.txt").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    shutil.copy2(ROOT / "packaging/RELEASE_QUICK_START.txt", output / "使用说明.txt")


def full_components(app, package, output, runtimes, catalog, runtime_assets):
    runtime = next(c for c in runtimes["components"] if c["id"] == "tifa-cpu")
    if not component_ready(package, runtime, verify=True):
        raise ValueError("Prepared TIFA runtime differs from the selected pinned runtime catalog")
    components = [app,
        bundle("tifa-cpu", runtime["name"], tree(package, "runtime/tifa-cpu") + tree(package, "resources/upstream/TIFA"), output, version=runtime["version"]),
        bundle("ffmpeg", "FFmpeg 解码环境", tree(package, "runtime/ffmpeg"), output)]
    components[1]["description"] = runtime["description"]
    components[1]["probe_files"] += [f["path"] for f in component_files(components[1])
                                      if "torch_directml/" in f["path"] or "directml" in f["path"].lower() and f["path"].endswith(".pyd")]
    vad = next(c for c in runtimes["components"] if c["id"] == "vad-model")
    vad_root = package if component_ready(package, vad, verify=True) else output / "verified-vad"
    if vad_root != package:
        for artifact in vad["artifacts"]:
            archive = runtime_assets / artifact["cache"]
            if digest(archive) != artifact["sha256"]:
                raise ValueError("VAD upstream archive checksum differs")
            unpack(artifact, archive, vad_root, None, lambda **_: None)
    assert component_ready(vad_root, vad, verify=True)
    for model in [vad, *catalog["components"]]:
        source_root = vad_root if model["id"] == "vad-model" else package
        for item in component_files(model):
            if "content" not in item and digest(source_root / item["path"]) != item["sha256"]:
                raise ValueError("Bundled model differs from official pinned file: " + item["path"])
        files = [(source_root / item["path"], item["path"]) for item in component_files(model)]
        for item in model.get("generated", []):
            path = output / "generated" / item["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(item["content"], encoding="utf-8", newline="")
            files = [(source, name) for source, name in files if name != item["path"]]
            files.append((path, item["path"]))
        components.append(bundle(model["id"], model["name"], files, output))
    return components


def build_installer(components, edition, version, output, stamp):
    selected = validate_catalog({"schema": 1, "version": version, "edition": edition, "components": components})
    directory = output / edition
    directory.mkdir()
    manifest = directory / "install-catalog.json"
    manifest.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    work = ROOT / "build" / ("single_exe_" + stamp) / edition
    name = "VoiceBankGen-" + ("Full" if edition == "full" else "Online") + "-Setup"
    subprocess.run([sys.executable, "-m", "PyInstaller", "--onefile", "--windowed", "--name", name,
                    "--paths", str(ROOT / "src"), "--exclude-module", "PySide6", "--exclude-module", "numpy",
                    "--add-data", str(manifest) + ";.", "--add-data", str(ROOT / "packaging/create_shortcut.ps1") + ";.",
                    "--icon", str(ROOT / "src/utau_forge/data/ui/voicebankgen.ico"), "--distpath", str(output),
                    "--workpath", str(work), "--specpath", str(work), str(ROOT / "packaging/bootstrap_entry.py")], check=True, cwd=ROOT)
    executable = output / (name + ".exe")
    with zipfile.ZipFile(executable, "a", compression=zipfile.ZIP_STORED, allowZip64=True) as payload:
        for component in components:
            for artifact in component["artifacts"]:
                if not artifact.get("url"):
                    payload.write(output / artifact["cache"], "payload/" + artifact["cache"])
    print(executable, flush=True)


if __name__ == "__main__":
    main()
