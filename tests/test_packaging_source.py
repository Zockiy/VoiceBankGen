"""Source release includes usable installer code, without shipping resource data."""
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_desktop", ROOT / "packaging/build_desktop.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def test_public_manifest_covers_application_tests_and_installer_inputs():
    manifest = json.loads((ROOT / "packaging/source_release_manifest.json").read_text(encoding="utf-8"))
    names = set(manifest["files"])
    for directory in (ROOT / "src", ROOT / "tests"):
        for path in directory.rglob("*.py"):
            if any(part in {".history", "__pycache__"} for part in path.parts):
                continue
            assert path.relative_to(ROOT).as_posix() in names
    for name in (*builder.PUBLIC_DOCS, "packaging/build_desktop.py", "packaging/build_single_exe.py",
                 "packaging/bootstrap_entry.py", "packaging/create_shortcut.ps1", "packaging/tifa_entry.py",
                 "LICENSES/CC-BY-SA-4.0.txt", "LICENSES/sources.json", "docs/BUILDING.md"):
        assert name in names
        assert (ROOT / name).is_file()
    assert "docs/UI_WORKFLOW_VERIFICATION.md" not in names


def test_spec_filters_models_added_by_hooks_and_shares_gui_worker_code(tmp_path, monkeypatch):
    hooks = ModuleType("PyInstaller.utils.hooks")
    hooks.collect_all = lambda name: ([(str(tmp_path / "model.pt"), "weights")], [], [])
    for name in ("PyInstaller", "PyInstaller.utils"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, "PyInstaller.utils.hooks", hooks)
    data = tmp_path / "src/utau_forge/data"
    data.mkdir(parents=True)
    (data / "inventory.json").write_text("{}", encoding="utf-8")
    captured = []

    def analysis(*args, **kwargs):
        records = [(str(Path(target) / Path(source).name), source, "DATA") for source, target in kwargs["datas"]]
        records.append(("silero_vad/model.onnx", "from-hook.onnx", "DATA"))
        return SimpleNamespace(pure=["application-code"], scripts=["entry"], binaries=[], datas=records)

    def exe(*args, **kwargs):
        captured.append((args[0], kwargs))
        return kwargs["name"]

    scope = {"Analysis": analysis, "PYZ": lambda pure: pure, "EXE": exe,
             "COLLECT": lambda *args, **kwargs: args}
    exec(compile(builder.desktop_spec(tmp_path), "generated.spec", "exec"), scope)
    assert [v[1]["name"] for v in captured] == ["VoiceBankGen", "VoiceBankGenWorker"]
    assert captured[0][0] is captured[1][0]
    records = scope["collection"][3]
    assert all(Path(target).suffix not in {".onnx", ".pt"} for target, _, _ in records)
    assert any(target.endswith("inventory.json") for target, _, _ in records)
    assert any(target.endswith("runtime/tifa_entry.py") or target.endswith("runtime\\tifa_entry.py")
               for target, _, _ in records)
