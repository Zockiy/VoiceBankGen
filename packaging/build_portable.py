"""Create an extract-and-run 7z/ZIP from fixed offline components."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from utau_forge.resource_download import component_files, destination, digest, unpack, validate_catalog
from prepare_download_catalog import generated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--payload-dir", type=Path)
    source.add_argument("--prepared-dir", type=Path, help="Reuse a prepared portable directory through hardlinks")
    parser.add_argument("--output", type=Path, required=True, help="New .7z or .zip file")
    parser.add_argument("--cpu-only", action="store_true", help="Exclude GPU runtime and its notices")
    parser.add_argument("--compression", choices=("balanced", "smallest"), default="balanced")
    parser.add_argument("--dictionary-mib", type=int, help="Override the 7z LZMA2 dictionary in MiB")
    parser.add_argument("--reuse-in-place", action="store_true", help="Compress an already prepared profile without rebuilding its directory")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or output.suffix.lower() not in {".7z", ".zip"}:
        raise ValueError("Output must be a new .7z or .zip file")
    dictionary_mib = args.dictionary_mib or (32 if args.compression == "balanced" else 384)
    if not 16 <= dictionary_mib <= 512:
        raise ValueError("Dictionary must be between 16 and 512 MiB")
    if args.reuse_in_place and not args.prepared_dir:
        parser.error("--reuse-in-place requires --prepared-dir")
    catalog = copy.deepcopy(validate_catalog(json.loads(args.catalog.read_text(encoding="utf-8"))))
    catalog["edition"] = "portable"
    if any(c["id"] == "tifa-gpu" for c in catalog["components"]):
        raise ValueError("Use a CPU/DirectML catalog from this fork; legacy CUDA components cannot be published")
    if args.cpu_only:
        if any("torch_directml/" in f["path"] for c in catalog["components"] for f in component_files(c)):
            raise ValueError("CPU-only requires a separate CPU runtime catalog; the DirectML runtime is shared")
    # Keep the original app archive checksum while replacing its installer
    # instructions through the catalog's supported generated-file mechanism.
    app = next(c for c in catalog["components"] if c["id"] == "app-core")
    quick_start = (ROOT / "packaging/PORTABLE_QUICK_START.txt").read_text(encoding="utf-8")
    replacements = {"使用说明.txt": quick_start}
    if args.cpu_only:
        start = quick_start.index("CPU / GPU 选择")
        end = quick_start.index("空格试听", start)
        replacements["使用说明.txt"] = quick_start[:start] + (
            "CPU 版\n本压缩包仅包含 CPU 运行环境及全部必要模型，不含 GPU / CUDA 运行库。\n"
            "无需另外安装 Python、Conda 或 MFA，首次使用不需要下载模型。\n"
            "请在“素材与切分”的计算设备中选择“CPU”。\n\n") + quick_start[end:]
        replacements["第三方许可说明.txt"] = (ROOT / "packaging/RELEASE_NOTICES.txt").read_text(encoding="utf-8")
    for artifact in app["artifacts"]:
        artifact["files"] = [f for f in artifact["files"] if f["path"] not in replacements
                             and not (args.cpu_only and f["path"].startswith("licenses/GPU/"))]
    app["generated"] = [f for f in app.get("generated", []) if f["path"] not in replacements]
    app["generated"] += [generated(name, content) for name, content in replacements.items()]
    validate_catalog(catalog)
    work = ROOT / "build" / ("portable_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    work.mkdir(parents=True, exist_ok=False)
    package = args.prepared_dir.resolve() if args.reuse_in_place else work / "VoiceBankGen"
    if args.reuse_in_place:
        if package.name != "VoiceBankGen" or not (package / "VoiceBankGen.exe").is_file():
            raise ValueError("Prepared directory must be the VoiceBankGen application folder")
        if (package / "runtime/tifa-gpu").exists() or (package / ".component-cache").exists():
            raise ValueError("In-place publication must exclude legacy CUDA runtimes and duplicate component caches")
    else:
        package.mkdir()
    for component in catalog["components"]:
        print("Preparing", component["id"], flush=True)
        if args.reuse_in_place:
            pass
        elif args.prepared_dir:
            for item in component_files(component):
                if "content" in item:
                    continue
                original = destination(args.prepared_dir, item["path"])
                target = destination(package, item["path"])
                if original.stat().st_size != item["size"]:
                    raise ValueError("Prepared file size differs: " + item["path"])
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(original, target)
                except OSError:
                    shutil.copy2(original, target)
        else:
            for artifact in component["artifacts"]:
                archive = destination(args.payload_dir, artifact["cache"])
                if archive.stat().st_size != artifact["size"] or digest(archive) != artifact["sha256"]:
                    raise ValueError("Component archive differs: " + archive.name)
                unpack(artifact, archive, package, None, lambda **_: None)
        for item in component.get("generated", []):
            path = destination(package, item["path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(item["content"].encode("utf-8"))
    (package / "components.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    sevenzip = shutil.which("7z") or shutil.which("7za")
    if not sevenzip:
        candidate = Path("C:/Program Files/7-Zip/7z.exe")
        sevenzip = str(candidate) if candidate.is_file() else None
    print("Compressing portable", output.suffix, flush=True)
    if sevenzip:
        if output.suffix.lower() == ".7z":
            settings = (["-t7z", "-mx=5", "-m0=LZMA2", f"-md={dictionary_mib}m", "-mfb=32",
                         "-ms=256m", "-mqs=on", "-mmt=4"] if args.compression == "balanced" else
                        ["-t7z", "-mx=9", "-m0=LZMA2", f"-md={dictionary_mib}m", "-mfb=273",
                         "-ms=on", "-mqs=on", "-mmt=2"])
        else:
            settings = ["-tzip", "-mx=5", "-mmt=on"]
        with (work / "compression.log").open("wb") as log:
            subprocess.run([sevenzip, "a", *settings, "-bsp0", str(output), "VoiceBankGen"],
                           cwd=package.parent, stdout=log, stderr=subprocess.STDOUT, check=True)
    else:
        if output.suffix.lower() == ".7z":
            raise FileNotFoundError("7-Zip is required to create a .7z archive")
        with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
            for path in sorted(package.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(package.parent).as_posix())
    checksum = digest(output)
    output.with_suffix(output.suffix + ".sha256").write_text(checksum + "  " + output.name + "\n", encoding="utf-8")
    report = {"archive": str(output), "bytes": output.stat().st_size, "sha256": checksum,
              "components": [c["id"] for c in catalog["components"]], "staging_directory": str(package),
              "cpu_only": args.cpu_only, "compression": args.compression,
              "dictionary_mib": dictionary_mib, "installation_or_runtime_tests_run": False}
    (work / "build-record.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
