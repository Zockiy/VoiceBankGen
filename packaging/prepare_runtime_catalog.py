"""Pin real upstream Windows runtime archives and their installed file mapping.

Run only when preparing a release. Users install verified files without pip,
dependency resolution, compilation, or access to a VoiceBankGen release server.
"""
from __future__ import annotations
import argparse
import calendar
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from prepare_download_catalog import file_record, generated

ROOT = Path(__file__).resolve().parents[1]
TIFA_COMMIT = "32a0a13aebce580844ad1dc281ea11ef1be94f14"
PYTHON_URL = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip"
RUNTIME = "runtime/tifa-cpu/"
METADATA_CACHE = None


def request(url):
    for attempt in range(4):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "VoiceBankGen-release-builder"}), timeout=60)
        except (OSError, urllib.error.URLError):
            if attempt == 3:
                raise
            time.sleep(1 + attempt)


def get_json(url):
    path = METADATA_CACHE / (hashlib.sha256(url.encode()).hexdigest() + ".json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    with request(url) as response:
        value = json.load(response)
    path.write_text(json.dumps(value), encoding="utf-8")
    return value


def acquire(url, path, expected=None):
    if not path.exists():
        print("Download", path.name, flush=True)
        with request(url) as response, path.open("wb") as target:
            while chunk := response.read(1024 * 1024):
                target.write(chunk)
    info = file_record(path, path.name)
    if expected and info["sha256"] != expected:
        raise ValueError("Upstream checksum differs: " + path.name)
    return info


def wheel_target(member):
    if member.startswith(("torch/include/", "torch/share/", "torch/bin/")) or Path(member).suffix.lower() in {".lib", ".pdb", ".exp"}:
        return None  # C++ extension build headers are unused by this inference environment.
    parts = member.split("/")
    if parts[0].endswith(".data"):
        if parts[1] in {"purelib", "platlib"}:
            return RUNTIME + "Lib/site-packages/" + "/".join(parts[2:])
        if parts[1] == "data":
            relative = "/".join(parts[2:])
            if relative.startswith("lib/"):
                relative = "Lib/" + relative[4:]
            return RUNTIME + relative
        # Console scripts and build headers are not used by our own entry point.
        return None
    return RUNTIME + "Lib/site-packages/" + member


def archive(url, path, mapper):
    result = acquire(url, path)
    files = []
    if path.name.endswith((".zip", ".whl")):
        format = "zip"
        with zipfile.ZipFile(path) as bundle:
            for item in bundle.infolist():
                target = mapper(item.filename) if not item.is_dir() else None
                if target:
                    content = bundle.read(item)
                    files.append({"path": target, "member": item.filename, "size": len(content), "sha256": hashlib.sha256(content).hexdigest()})
    else:
        format = "tar"
        with tarfile.open(path, "r:gz") as bundle:
            for item in bundle.getmembers():
                target = mapper(item.name) if item.isfile() else None
                if target:
                    with bundle.extractfile(item) as stream:
                        content = stream.read()
                    files.append({"path": target, "member": item.name, "size": len(content), "sha256": hashlib.sha256(content).hexdigest()})
    record = {"url": url, "cache": path.name, "format": format, "size": result["size"], "sha256": result["sha256"], "files": files}
    if result["size"] > 8 * 1024**2:
        record["range_size"] = 8 * 1024**2
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--pip-python", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "src/utau_forge/data/runtime-catalog.json")
    parser.add_argument("--skip-resolve", action="store_true", help="Reuse a completed pinned download directory")
    parser.add_argument("--backend", choices=("cpu", "directml"), default="directml")
    parser.add_argument("--support-catalog", type=Path, help="Keep pinned FFmpeg/VAD components from an existing catalog")
    args = parser.parse_args()
    cache = args.cache.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    global METADATA_CACHE
    METADATA_CACHE = cache / ".metadata"
    METADATA_CACHE.mkdir(exist_ok=True)
    if not args.skip_resolve:
        subprocess.run([str(args.pip_python), "-m", "pip", "download", "--no-cache-dir", "--timeout", "120", "--retries", "8", "--only-binary=:all:",
                        "--no-binary=textgrid,antlr4-python3-runtime,unidic", "--index-url", "https://pypi.org/simple",
                        "-r", str(ROOT / "packaging" / ("runtime_directml_requirements.txt" if args.backend == "directml" else "runtime_requirements.txt")), "-d", str(cache)], check=True)
    artifacts = [archive(PYTHON_URL, cache / "python-3.11.9-embed-amd64.zip",
                         lambda m: None if m == "python311._pth" else RUNTIME + m)]
    wheels = sorted(cache.glob("*.whl")) + sorted(cache.glob("*.tar.gz"))
    dependencies = []
    for path in wheels:
        print("Pin", path.name, flush=True)
        if path.name.startswith(("torch-", "torchvision-")) and "+cpu" in path.name:
            url = "https://download.pytorch.org/whl/cpu/" + path.name.replace("+", "%2B")
            name, version = path.name.split("-")[:2]
        else:
            # Filename's distribution and version are resolved to primary PyPI metadata.
            if path.suffix == ".whl":
                name, version = path.name.split("-")[:2]
            else:
                name, version = path.name.removesuffix(".tar.gz").rsplit("-", 1)
            metadata = get_json(f"https://pypi.org/pypi/{name}/{version}/json")
            info = next(item for item in metadata["urls"] if item["filename"] == path.name)
            url = info["url"]
            acquire(url, path, info["digests"]["sha256"])
        dependencies.append({"name": name, "version": version, "url": url})
        if path.suffix == ".whl":
            mapper = wheel_target
        elif name.lower() in {"textgrid", "unidic"}:
            package = name.lower()
            mapper = lambda m, package=package: RUNTIME + "Lib/site-packages/" + m.split("/", 1)[1] if "/" + package + "/" in m else None
        elif name.lower() == "antlr4-python3-runtime":
            mapper = lambda m: RUNTIME + "Lib/site-packages/" + m.split("/src/", 1)[1] if "/src/antlr4/" in m else None
        else:
            raise ValueError("Unmapped source distribution: " + path.name)
        artifacts.append(archive(url, path, mapper))
    by_name = {item["name"].replace("_", "-").lower(): item for item in dependencies}
    expected_torch = {"2.4.1", "2.4.1+cpu"} if args.backend == "directml" else {"2.6.0+cpu"}
    if by_name.get("torch", {}).get("version") not in expected_torch:
        raise ValueError("Wrong torch wheel; use a fresh cache for each backend")
    if args.backend == "directml":
        torch_wheel = next(p for p in wheels if p.name.startswith("torch-"))
        with zipfile.ZipFile(torch_wheel) as source:
            version_source = source.read("torch/version.py").decode("utf-8")
            if "'2.4.1+cpu'" not in version_source or any(name.lower().endswith("torch_cuda.dll") for name in source.namelist()):
                raise ValueError("DirectML requires the verified CPU-only torch build")
    if args.backend == "directml" and by_name.get("torch-directml", {}).get("version") != "0.2.5.dev240914":
        raise ValueError("DirectML wheel is missing or unpinned")
    if args.backend == "cpu" and "torch-directml" in by_name:
        raise ValueError("CPU-only cache contains DirectML; use a fresh cache")
    url = f"https://github.com/openvpi/TIFA/archive/{TIFA_COMMIT}.zip"
    artifacts.append(archive(url, cache / ("TIFA-source-" + TIFA_COMMIT + ".zip"),
                             lambda m: "resources/upstream/TIFA/" + m.split("/", 1)[1] if "/" in m else None))
    cpu = {"id": "tifa-cpu", "name": "TIFA CPU 运行环境", "required": True, "version": "python3.11.9-torch2.6.0",
           "description": "官方 Python、PyTorch/PyPI 依赖及固定版本 TIFA 源码；无需系统 Python。",
           "artifacts": artifacts, "generated": [generated(RUNTIME + "python311._pth", "python311.zip\n.\nLib/site-packages\nimport site\n"),
                generated(RUNTIME + "dependencies.json", json.dumps(dependencies, ensure_ascii=False, indent=2))],
           "probe_files": [RUNTIME + "python.exe", RUNTIME + "python311.dll", RUNTIME + "Lib/site-packages/torch/lib/torch_cpu.dll",
                           RUNTIME + "Lib/site-packages/unidic/__init__.py", "resources/upstream/TIFA/inference/api.py"]}
    if args.backend == "directml":
        cpu.update(name="TIFA CPU / DirectML 共享运行环境", version="python3.11.9-torch2.4.1-dml0.2.5",
                   description="共用一份 CPU PyTorch，DirectML 加速 TIFA 神经网络；不包含 CUDA/cuDNN。语音识别使用 CPU。")
        dml_files = [item["path"] for artifact in artifacts for item in artifact["files"]
                     if item["path"].endswith("torch_directml/__init__.py")
                     or ("directml" in item["path"].lower() and item["path"].endswith(".pyd"))]
        if not dml_files:
            raise ValueError("DirectML wheel has no usable Windows runtime files")
        cpu["probe_files"] += dml_files
    if args.support_catalog:
        sys.path.insert(0, str(ROOT / "src"))
        from utau_forge.resource_download import save_json, validate_catalog
        previous = validate_catalog(json.loads(args.support_catalog.read_text(encoding="utf-8")))
        support = {c["id"]: c for c in previous["components"]}
        result = {"schema": 1, "version": "2026.11.05-directml" if args.backend == "directml" else "2026.11.05-cpu",
                  "components": [cpu, support["ffmpeg"], support["vad-model"]]}
        save_json(args.output, validate_catalog(result))
        print(args.output, flush=True)
        return
    releases = get_json("https://api.github.com/repos/BtbN/FFmpeg-Builds/releases?per_page=100")
    # The maintainer prunes daily assets after 14 builds. Month-end snapshots
    # are retained for two years, so don't ship a soon-to-expire daily URL.
    def retained(item):
        parts = item["tag_name"].split("-")
        return (not item["draft"] and len(parts) >= 4 and parts[0] == "autobuild"
                and int(parts[3]) == calendar.monthrange(int(parts[1]), int(parts[2]))[1])
    release = next(item for item in releases if retained(item))
    asset = next(item for item in release["assets"] if item["name"].endswith("win64-lgpl.zip"))
    sys.path.insert(0, str(ROOT / "src"))
    from utau_forge.resource_download import fetch, preserve
    path = cache / asset["name"]
    if path.exists() and path.stat().st_size != asset["size"]:
        preserve(path)
    last = [0]
    def progress(**event):
        if event.get("current", 0) - last[0] >= 8 * 1024**2:
            print("FFmpeg", round(event["current"] / 1024**2), "MB", flush=True)
            last[0] = event["current"]
    fetch({"cache": path.name, "size": asset["size"], "sha256": (asset.get("digest") or "").removeprefix("sha256:"),
           "url": asset["browser_download_url"], "range_size": 8 * 1024**2}, cache, {"source": "official"}, None, progress)
    ffmpeg_asset = archive(asset["browser_download_url"], path,
                  lambda m: "runtime/ffmpeg/" + m.split("/", 1)[1] if "/" in m else None)
    ffmpeg_asset["range_size"] = 8 * 1024**2
    ffmpeg = {"id": "ffmpeg", "name": "FFmpeg 解码环境", "required": True, "version": release["tag_name"],
              "description": "BtbN 维护的 LGPL Windows 静态构建（FFmpeg 官网推荐来源之一）。",
              "artifacts": [ffmpeg_asset],
              "probe_files": ["runtime/ffmpeg/bin/ffmpeg.exe", "runtime/ffmpeg/bin/ffprobe.exe"]}
    vad_artifacts = []
    (cache / "vad").mkdir(exist_ok=True)
    for distribution in ("silero-vad", "faster-whisper"):
        version = importlib.metadata.version(distribution)
        metadata = get_json(f"https://pypi.org/pypi/{distribution}/{version}/json")
        info = next(item for item in metadata["urls"] if item["filename"].endswith(".whl"))
        path = cache / "vad" / info["filename"]
        acquire(info["url"], path, info["digests"]["sha256"])
        asset = archive(info["url"], path, lambda m: "_internal/" + m if m.endswith((".onnx", ".jit")) else None)
        asset["cache"] = "vad/" + path.name
        vad_artifacts.append(asset)
    vad = {"id": "vad-model", "name": "Silero 语音检测模型", "required": True, "version": "silero-" + importlib.metadata.version("silero-vad"),
           "description": "库附带的 Silero 模型也独立下载，联网 EXE 不内置模型权重。", "artifacts": vad_artifacts}
    result = {"schema": 1, "version": "2026.10.07", "components": [cpu, ffmpeg, vad]}
    sys.path.insert(0, str(ROOT / "src"))
    from utau_forge.resource_download import save_json, validate_catalog
    save_json(args.output, validate_catalog(result))
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
