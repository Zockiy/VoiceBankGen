"""Publication boundary tests; all fixtures are generated in a temporary directory."""
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest

spec = importlib.util.spec_from_file_location("source_export", Path(__file__).resolve().parents[1] / "packaging/export_source.py")
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


def fixture_root(tmp_path, files):
    (tmp_path / "packaging").mkdir()
    (tmp_path / exporter.MANIFEST).write_text(json.dumps({"files": files}), encoding="utf-8")
    return tmp_path


def test_export_is_an_allowlist_and_omits_history_and_runtime_data(tmp_path):
    root = fixture_root(tmp_path, ["main.py", "LICENSE"])
    (root / "main.py").write_text("print('hello')\n", encoding="utf-8")
    (root / "LICENSE").write_text("MIT\n", encoding="utf-8")
    (root / ".git").mkdir()
    (root / ".git/private-history").write_text("private")
    (root / "recording.wav").write_bytes(b"private recording")
    (root / "model.pt").write_bytes(b"private model")
    report = exporter.export_source(root, root / "release/public")
    with zipfile.ZipFile(report["archive"]) as archive:
        assert set(archive.namelist()) == {"VoiceBankGen/main.py", "VoiceBankGen/LICENSE", "VoiceBankGen/SOURCE_MANIFEST.json"}
    assert not (Path(report["directory"]) / ".git").exists()
    assert (root / "recording.wav").read_bytes() == b"private recording"
    assert (root / "model.pt").exists()
    with pytest.raises(FileExistsError):
        exporter.export_source(root, Path(report["directory"]))


@pytest.mark.parametrize("name", ["clip.wav", "model.pt", "app.exe", "credentials.pem", ".env", "../outside.py", "runtime/code.py"])
def test_export_rejects_asset_or_unsafe_paths(tmp_path, name):
    root = fixture_root(tmp_path, [name])
    if ".." not in name:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("data", encoding="utf-8")
    with pytest.raises((ValueError, FileNotFoundError)):
        exporter.export_source(root, root / "release/rejected")
    assert not (root / "release/rejected").exists()


def test_export_refuses_detected_credentials_without_printing_them(tmp_path):
    root = fixture_root(tmp_path, ["main.py"])
    token = "ghp_" + "x" * 36
    (root / "main.py").write_text("token = " + repr(token), encoding="utf-8")
    with pytest.raises(ValueError, match="github-token") as error:
        exporter.export_source(root, root / "release/rejected")
    assert token not in str(error.value)


def test_export_refuses_binary_disguised_as_text(tmp_path):
    root = fixture_root(tmp_path, ["data.json"])
    (root / "data.json").write_bytes(b"\0binary")
    with pytest.raises(ValueError, match="Binary content"):
        exporter.export_source(root, root / "release/rejected")


def test_negative_proxy_fixture_has_only_an_exact_dummy_exemption(tmp_path):
    root = fixture_root(tmp_path, ["main.py"])
    path = root / "main.py"
    path.write_text(repr("http://user" + ":pass@host:1234"), encoding="utf-8")
    assert len(exporter.checked_files(root)) == 1
    path.write_text(repr("http://realuser" + ":realpass@example.org"), encoding="utf-8")
    with pytest.raises(ValueError, match="credential-in-url"):
        exporter.checked_files(root)


def test_export_normalizes_text_for_stable_git_hashes(tmp_path):
    root = fixture_root(tmp_path, ["main.py"])
    (root / "main.py").write_bytes(b"print('hello')\r\n")
    report = exporter.export_source(root, root / "release/normalized")
    content = (Path(report["directory"]) / "main.py").read_bytes()
    assert content == b"print('hello')\n"
    import hashlib
    manifest = json.loads((Path(report["directory"]) / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["files"][0]["sha256"] == hashlib.sha256(content).hexdigest()
