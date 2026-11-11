"""Rebuild application code while retaining an existing shared portable runtime."""
from pathlib import Path
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone

from PyInstaller.archive.readers import CArchiveReader
from PyInstaller.archive.writers import ZlibArchiveWriter
from build_shared_portable import ROOT, write_exe


def refresh(base: Path, output: Path, resume: bool = False, version: str | None = None, in_place: bool = False):
    if not (base / "shared-runtime.json").is_file():
        raise ValueError("An existing shared portable runtime is required")
    if in_place:
        if output != (ROOT / "dist/VoiceBankGen").resolve():
            raise ValueError("In-place updates are restricted to dist/VoiceBankGen")
        package = output
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
        work = ROOT / "build" / ("desktop_update_" + stamp)
        work.mkdir(parents=True)
        package.mkdir(parents=True, exist_ok=True)
        backup = work / "previous"
        backup.mkdir()
        moved = []
        try:
            for name in ("VoiceBankGen.exe", "VoiceBankGenWorker.exe", "VoiceBankGen_UI_Fork.exe", "components.json", "shared-runtime.json", "_internal"):
                source, destination = package / name, backup / name
                if source.exists():
                    if source.resolve().parent != package.resolve() or not destination.resolve().is_relative_to(work.resolve()):
                        raise ValueError("Unexpected update path")
                    source.rename(destination)
                    moved.append((source, destination))
        except Exception:
            for source, destination in reversed(moved):
                destination.rename(source)
            raise
    else:
        output.mkdir(parents=True, exist_ok=resume)
        package = output / "VoiceBankGen"
        work = output / "build-records"
        package.mkdir(exist_ok=resume)
        work.mkdir(exist_ok=resume)
    for path in base.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(base)
        if any(part in {"__pycache__", ".component-cache", "backups"} for part in relative.parts):
            continue
        if relative.as_posix() in {"VoiceBankGen.exe", "VoiceBankGenWorker.exe", "components.json"}:
            continue
        destination = package / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            os.link(path, destination)
    archive = CArchiveReader(str(base / "VoiceBankGen.exe")).open_embedded_archive("PYZ.pyz")
    entries, code = {}, {}
    for name, row in archive.toc.items():
        kind = row[0]
        filename = None if kind == 3 else name.replace(".", "/") + ("/__init__.py" if kind == 1 else ".py")
        entries[name] = (name, filename, "PYMODULE")
        if kind != 3:
            code[name] = archive.extract(name)
    for path in (ROOT / "src/utau_forge").rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(ROOT / "src").with_suffix("")
        name = ".".join(relative.parts[:-1] if relative.name == "__init__" else relative.parts)
        entries[name] = (name, str(path), "PYMODULE")
        code[name] = compile(path.read_text(encoding="utf-8-sig"), str(path), "exec")
    pyz = work / "application.pyz"
    ZlibArchiveWriter(str(pyz), list(entries.values()), code_dict=code)
    for name in ("VoiceBankGen.exe", "VoiceBankGenWorker.exe"):
        write_exe(base / name, package / name, pyz, work)
    if in_place:
        os.link(package / "VoiceBankGen.exe", package / "VoiceBankGen_UI_Fork.exe")
    catalog = json.loads((base / "components.json").read_text(encoding="utf-8"))
    if version:
        catalog["version"] = version
    for component in catalog["components"]:
        if version and component["id"] == "app-core":
            component["version"] = version
        for record in component.get("generated", []):
            if record["path"] in {"VoiceBankGen.exe", "VoiceBankGenWorker.exe"}:
                path = package / record["path"]
                record.update(size=path.stat().st_size, sha256=hashlib.file_digest(path.open("rb"), "sha256").hexdigest())
    (package / "components.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    print(package, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--version")
    parser.add_argument("--in-place", action="store_true")
    args = parser.parse_args()
    refresh(args.base.resolve(), args.output.resolve(), args.resume, args.version, args.in_place)
