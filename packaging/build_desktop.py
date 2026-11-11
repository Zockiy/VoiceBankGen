"""Build a fresh Windows desktop core from public source; never copy local models.

The resulting portable directory is the input for build_single_exe.py. Models,
TIFA runtime and FFmpeg can subsequently be installed by its download dialog.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import importlib.metadata
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DOCS = ("README.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "docs/USAGE.md",
               "docs/UI_WORKFLOW_FORK.md", "docs/LICENSE_AUDIT.md")


def desktop_spec(root: Path) -> str:
    """A shared PYZ gives GUI and disposable worker the same application code."""
    return f'''from pathlib import Path
from PyInstaller.utils.hooks import collect_all
root = Path({str(root)!r})
datas = []
binaries = []
hidden = []
for name in ["silero_vad", "pyqtgraph", "faster_whisper", "pyworld", "soundfile"]:
    d, b, h = collect_all(name)
    datas += d
    binaries += b
    hidden += h
# The online core contains no neural network weights, including VAD assets
# otherwise automatically collected from installed wheels.
datas = [(s, t) for s, t in datas if Path(s).suffix.lower() not in
         {{".onnx", ".jit", ".pt", ".pth", ".ckpt", ".safetensors"}}]
for path in (root / "src/utau_forge/data").rglob("*"):
    if path.is_file() and not any(p in {{".history", "__pycache__"}} for p in path.parts):
        target = Path("utau_forge/data") / path.relative_to(root / "src/utau_forge/data").parent
        datas.append((str(path), str(target)))
datas.append((str(root / "packaging/tifa_entry.py"), "utau_forge/data/runtime"))
a = Analysis([str(root / "packaging/desktop_entry.py")], pathex=[str(root / "src")],
    binaries=binaries, datas=datas, hiddenimports=hidden,
    excludes=["gradio", "plotly", "lightning", "matplotlib", "IPython", "parselmouth"])
# Hooks can append their own model data after the initial collection.
a.datas = [(t, s, k) for t, s, k in a.datas if Path(t).suffix.lower() not in
           {{".onnx", ".jit", ".pt", ".pth", ".ckpt", ".safetensors"}}]
pyz = PYZ(a.pure)
icon = str(root / "src/utau_forge/data/ui/voicebankgen.ico")
gui = EXE(pyz, a.scripts, [], exclude_binaries=True, name="VoiceBankGen", console=False, icon=icon, upx=False)
worker = EXE(pyz, a.scripts, [], exclude_binaries=True, name="VoiceBankGenWorker", console=True, icon=icon, upx=False)
collection = COLLECT(gui, worker, a.binaries, a.datas, name="VoiceBankGen", upx=False)
'''


def copy_notices(package: Path) -> None:
    # Retain upstream texts from the build environment as a starting inventory.
    # This is not a replacement for inspecting native libraries and Qt modules.
    notices = package / "licenses"
    notices.mkdir()
    for name in PUBLIC_DOCS:
        target = package / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    shutil.copytree(ROOT / "LICENSES", package / "LICENSES")
    for dist in importlib.metadata.distributions():
        for item in dist.files or []:
            relative = Path(str(item))
            if not relative.parts or not relative.parts[0].endswith(".dist-info"):
                continue
            if not any(word in relative.name.lower() for word in ("license", "copying", "notice")):
                continue
            source = Path(dist.locate_file(item))
            if source.is_file() and ".." not in relative.parts:
                target = notices / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New build directory; defaults to dist/desktop_<timestamp>")
    args = parser.parse_args()
    if sys.platform != "win32":
        parser.error("This builder targets Windows x64")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = (args.output or ROOT / "dist" / ("desktop_" + stamp)).resolve()
    if output.exists():
        raise FileExistsError("Refusing to overwrite an existing desktop build")
    work = ROOT / "build" / ("desktop_" + stamp)
    work.mkdir(parents=True, exist_ok=False)
    spec = work / "VoiceBankGen.spec"
    spec.write_text(desktop_spec(ROOT), encoding="utf-8")
    subprocess.run([sys.executable, "-m", "PyInstaller", "--distpath", str(output),
                    "--workpath", str(work), str(spec)], cwd=ROOT, check=True)
    package = output / "VoiceBankGen"
    copy_notices(package)
    print("DESKTOP_CORE", package)


if __name__ == "__main__":
    main()
