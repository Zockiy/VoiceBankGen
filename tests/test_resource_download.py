"""Real HTTP and interrupted-install regression tests for pinned resources."""
import hashlib
import io
import json
import threading
import zipfile
import tarfile
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from utau_forge import resource_download as resources


def file(name, data, member=None):
    result = {"path": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    if member:
        result["member"] = member
    return result


def artifact(data, files, url="", format="raw"):
    return {"cache": "asset.bin", "size": len(data), "sha256": hashlib.sha256(data).hexdigest(),
            "files": files, "url": url, "format": format}


def catalog(item):
    return {"schema": 1, "components": [{"id": "fixture", "name": "测试组件", "required": True, "artifacts": [item]}]}


def test_proxy_and_other_mirror_preferences(tmp_path, monkeypatch):
    monkeypatch.setenv("UFORGE_DATA_DIR", str(tmp_path))
    prefs = resources.save_preferences({"source": "mirror", "other_mirror": "https://relay.example/{url}", "proxy": "http://127.0.0.1:7890"})
    assert resources.load_preferences() == prefs
    url = "https://download.pytorch.org/whl/cpu/pinned.whl"
    assert resources.download_url(url, prefs) == "https://relay.example/" + url
    assert resources.download_url(url, {"source": "mirror"}) == url
    for invalid in ("socks5://127.0.0.1:1080", "http://127.0.0.1", "http://127.0.0.1:bad", "http://user:pass@host:1234"):
        with pytest.raises(ValueError, match="代理"):
            resources.save_preferences({"source": "official", "proxy": invalid})


def test_explicit_http_proxy_is_used(monkeypatch):
    import urllib.request
    monkeypatch.setenv("no_proxy", "")
    data = b"proxy-response"
    with server(data) as (proxy, requests):
        with resources.open_download(urllib.request.Request("http://download.example/asset"), {"proxy": proxy.rsplit("/", 1)[0]}) as response:
            assert response.read() == data
    assert requests == [None]


def test_transient_connection_open_retries_and_cancellation(monkeypatch):
    import urllib.error
    calls = []
    def connect(request, timeout):
        calls.append(timeout)
        if len(calls) < 2:
            raise urllib.error.URLError("TLS handshake timeout")
        return "connected"
    monkeypatch.setattr(resources.urllib.request, "urlopen", connect)
    assert resources.open_download(None, {}) == "connected"
    assert len(calls) == 2
    cancel = threading.Event()
    def cancel_connect(request, timeout):
        cancel.set()
        raise urllib.error.URLError("temporary failure")
    monkeypatch.setattr(resources.urllib.request, "urlopen", cancel_connect)
    with pytest.raises(resources.DownloadCancelled):
        resources.open_download(None, {}, cancel)


def test_nontransient_http_error_is_not_retried(monkeypatch):
    import urllib.error
    calls = []
    def missing(request, timeout):
        calls.append(timeout)
        raise urllib.error.HTTPError("https://official.example", 404, "missing", {}, None)
    monkeypatch.setattr(resources.urllib.request, "urlopen", missing)
    with pytest.raises(urllib.error.HTTPError):
        resources.open_download(None, {})
    assert calls == [30]


def test_pinned_source_distribution_install(tmp_path):
    buffer = io.BytesIO()
    data = b"__version__='1.0'\n"
    with tarfile.open(fileobj=buffer, mode="w:gz") as bundle:
        item = tarfile.TarInfo("upstream/src/module.py")
        item.size = len(data)
        bundle.addfile(item, io.BytesIO(data))
    with server(buffer.getvalue()) as (url, _):
        item = artifact(buffer.getvalue(), [file("runtime/site-packages/module.py", data, "upstream/src/module.py")], url, "tar")
        resources.ComponentInstaller(tmp_path, catalog(item)).run(["fixture"])
    assert (tmp_path / "runtime/site-packages/module.py").read_bytes() == data


@pytest.mark.parametrize("kind", ["link", "traversal"])
def test_source_distribution_rejects_unsafe_entries(tmp_path, kind):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as bundle:
        item = tarfile.TarInfo("source/link" if kind == "link" else "../outside")
        if kind == "link":
            item.type = tarfile.SYMTYPE
            item.linkname = "../../outside"
        bundle.addfile(item)
    path = tmp_path / "archive.tar.gz"
    path.write_bytes(buffer.getvalue())
    with pytest.raises(ValueError):
        resources.unpack(artifact(buffer.getvalue(), [], format="tar"), path, tmp_path / "stage", None, lambda **_: None)


@contextmanager
def server(data, mode="range"):
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.headers.get("Range"))
            requested = self.headers.get("Range", "bytes=0-").split("=")[1].split("-")
            start = int(requested[0])
            end = int(requested[1]) if requested[1] else len(data) - 1
            if mode == "ignore":
                start = 0
                end = len(data) - 1
            code = 206 if self.headers.get("Range") and mode != "ignore" else 200
            self.send_response(code)
            if code == 206:
                offset = start + 1 if mode == "bad_range" else start
                self.send_header("Content-Range", f"bytes {offset}-{end}/{len(data)}")
            self.send_header("Content-Length", str(end + 1 - start))
            self.end_headers()
            try:
                self.wfile.write(data[start:end + 1])
            except (ConnectionResetError, BrokenPipeError):
                pass
        def log_message(self, *_):
            pass
    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{http.server_port}/asset", requests
    finally:
        http.shutdown()
        http.server_close()
        thread.join()


@pytest.mark.parametrize("mode", ["range", "ignore"])
def test_http_resume_and_ignored_range_keep_original_part(tmp_path, mode):
    data = b"pinned-data" * 1000
    cache = tmp_path / "cache"
    cache.mkdir()
    part = cache / "asset.bin.part"
    part.write_bytes(data[:123])
    with server(data, mode) as (url, requests):
        path = resources.fetch(artifact(data, [file("resource.bin", data)], url), cache, {}, threading.Event(), lambda **_: None)
    assert requests == ["bytes=123-"]
    assert path.read_bytes() == data
    if mode == "ignore":
        assert next(cache.glob("*.retained-*")).read_bytes() == data[:123]


@pytest.mark.parametrize("mode", ["range", "ignore"])
def test_segmented_http_range_download_and_resume(tmp_path, mode):
    data = b"pinned large archive" * 100
    (tmp_path / "asset.bin.part").write_bytes(data[:120])
    item = artifact(data, [file("data", data)], "")
    item["range_size"] = 400
    with server(data, mode) as (url, requests):
        item["url"] = url
        path = resources.fetch(item, tmp_path, {}, None, lambda **_: None)
    assert path.read_bytes() == data
    assert requests[0] == "bytes=120-519"
    assert len(requests) == (5 if mode == "range" else 1)


def test_reject_incorrect_resume_offset(tmp_path):
    data = b"abcdef" * 100
    (tmp_path / "asset.bin.part").write_bytes(data[:20])
    with server(data, "bad_range") as (url, _):
        with pytest.raises(ValueError, match="续传范围"):
            resources.fetch(artifact(data, [file("data", data)], url), tmp_path, {}, threading.Event(), lambda **_: None)
    assert not (tmp_path / "asset.bin").exists()
    assert next(tmp_path.glob("*.retained-*")).read_bytes() == data[:20]


def test_cancel_then_continue_http_download(tmp_path, monkeypatch):
    monkeypatch.setattr(resources, "CHUNK", 100)
    data = b"sample" * 500
    cancel = threading.Event()
    def interrupt(**event):
        if event["stage"] == "下载" and event["current"] >= 100:
            cancel.set()
    with server(data) as (url, requests):
        item = artifact(data, [file("data", data)], url)
        with pytest.raises(resources.DownloadCancelled):
            resources.fetch(item, tmp_path, {}, cancel, interrupt)
        assert (tmp_path / "asset.bin.part").stat().st_size == 100
        cancel.clear()
        resources.fetch(item, tmp_path, {}, cancel, lambda **_: None)
    assert requests == [None, "bytes=100-"]
    assert (tmp_path / "asset.bin").read_bytes() == data


def test_completed_partial_is_reused_without_network(tmp_path):
    data = b"finished before cancellation"
    (tmp_path / "asset.bin.part").write_bytes(data)
    item = artifact(data, [file("model", data)], "http://127.0.0.1:1/unreachable")
    result = resources.fetch(item, tmp_path, {}, threading.Event(), lambda **_: None)
    assert result.read_bytes() == data


def test_hash_failure_does_not_change_old_resource(tmp_path):
    data = b"new file"
    (tmp_path / "model").write_bytes(b"old file")
    with server(b"bad file") as (url, _):
        installer = resources.ComponentInstaller(tmp_path, catalog(artifact(data, [file("model", data)], url)))
        with pytest.raises(ValueError, match="SHA-256"):
            installer.run(["fixture"])
    assert (tmp_path / "model").read_bytes() == b"old file"
    assert list((tmp_path / ".component-cache").glob("*.retained-*"))
    assert not (tmp_path / ".component-lock").exists()


def test_zip_install_and_repair_preserve_old_file(tmp_path):
    data = b"model content"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        bundle.writestr("upstream/model.bin", data)
    payload = buffer.getvalue()
    (tmp_path / "resources").mkdir()
    (tmp_path / "resources/model.bin").write_bytes(b"old")
    with server(payload) as (url, requests):
        items = catalog(artifact(payload, [file("resources/model.bin", data, "model.bin")], url, "zip"))
        installer = resources.ComponentInstaller(tmp_path, items)
        assert installer.run(["fixture"]) == ["fixture"]
        assert installer.run(["fixture"]) == ["fixture"]
        assert len(requests) == 1
    assert (tmp_path / "resources/model.bin").read_bytes() == data
    assert next((tmp_path / ".component-backups").glob("*/resources/model.bin")).read_bytes() == b"old"


@pytest.mark.parametrize("name", ["../escaped", "/escaped", "C:/escaped", "a\\b", "a/../b", "CON", "a./b"])
def test_unsafe_paths_are_rejected(name):
    with pytest.raises(ValueError):
        resources.safe_relative(name)


def test_malicious_zip_and_bad_file_hash_never_commit(tmp_path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zipped:
        zipped.writestr("model.bin", b"model")
        zipped.writestr("../escape", b"bad")
    data = buffer.getvalue()
    with server(data) as (url, _):
        items = catalog(artifact(data, [file("model.bin", b"model")], url, "zip"))
        with pytest.raises(ValueError, match="路径不安全"):
            resources.ComponentInstaller(tmp_path, items).run(["fixture"])
    assert not (tmp_path / "model.bin").exists()
    assert not (tmp_path.parent / "escape").exists()


def test_partial_commit_rollback_and_stale_lock_recovery(tmp_path, monkeypatch):
    data = b"new"
    component = catalog(artifact(data, [file("first", data)]))
    component["components"][0]["generated"] = [{**file("second", data), "content": "new"}]
    cache = tmp_path / ".component-cache"
    cache.mkdir()
    (cache / "asset.bin").write_bytes(data)
    (tmp_path / "first").write_bytes(b"original first")
    (tmp_path / "second").write_bytes(b"original second")
    real_replace = Path.replace
    failure = []
    def fail_once(self, target):
        if Path(target) == tmp_path / "second" and ".component-staging" in str(self) and not failure:
            failure.append(True)
            raise PermissionError("controlled lock")
        return real_replace(self, target)
    monkeypatch.setattr(Path, "replace", fail_once)
    with pytest.raises(PermissionError):
        resources.ComponentInstaller(tmp_path, component).run(["fixture"])
    assert (tmp_path / "first").read_bytes() == b"original first"
    assert (tmp_path / "second").read_bytes() == b"original second"
    assert json.loads(next((tmp_path / ".component-backups").glob("*/installation.json")).read_text())["phase"] == "rolled_back"
    resources.ComponentInstaller(tmp_path, component).run(["fixture"])
    assert (tmp_path / "first").read_bytes() == data
    assert (tmp_path / "second").read_bytes() == data


def test_recover_power_loss_between_renames(tmp_path):
    data = b"new"
    component = catalog(artifact(data, [file("first", data)]))
    backup = tmp_path / ".component-backups/interrupted"
    stage = tmp_path / ".component-staging/interrupted"
    backup.mkdir(parents=True)
    stage.mkdir(parents=True)
    (backup / "first").write_bytes(b"old")
    (tmp_path / "first").write_bytes(data)
    (backup / "installation.json").write_text(json.dumps({"phase": "installing", "component": "fixture", "files": [file("first", data)]}))
    resources.ComponentInstaller(tmp_path, component).recover()
    assert (tmp_path / "first").read_bytes() == b"old"
    assert (stage / "first").read_bytes() == data
    assert json.loads((backup / "installation.json").read_text())["phase"] == "rolled_back"


def test_mirror_urls_and_remembered_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("UFORGE_DATA_DIR", str(tmp_path))
    preferences = resources.save_preferences({"source": "mirror", "github_mirror": "https://mirror.example/{url}", "hf_mirror": "https://hf.example"})
    assert resources.load_preferences() == preferences
    assert resources.download_url("https://github.com/openvpi/TIFA/releases/file.zip", preferences) == "https://mirror.example/https://github.com/openvpi/TIFA/releases/file.zip"
    assert resources.download_url("https://huggingface.co/Systran/revision/file", preferences) == "https://hf.example/Systran/revision/file"
    with pytest.raises(ValueError, match="填写"):
        resources.download_url("https://github.com/repo/file", {"source": "mirror"})
    with pytest.raises(ValueError):
        resources.save_preferences({"source": "mirror", "github_mirror": "http://bad.example"})


def test_real_mirror_transport_keeps_pinned_hash(tmp_path):
    data = b"mirrored pinned bytes"
    with server(data) as (url, requests):
        preferences = {"source": "mirror", "github_mirror": url.rsplit("/", 1)[0]}
        item = artifact(data, [file("model", data)], "https://github.com/upstream/release/asset")
        path = resources.fetch(item, tmp_path, preferences, threading.Event(), lambda **_: None)
    assert requests == [None] and path.read_bytes() == data


def test_zip_without_archive_hash_checks_each_file_before_commit(tmp_path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        bundle.writestr("model.bin", b"wrong")
    payload = buffer.getvalue()
    with server(payload) as (url, _):
        item = artifact(payload, [file("model.bin", b"right")], url, "zip")
        item.pop("sha256")
        with pytest.raises(ValueError, match="组件文件校验失败"):
            resources.ComponentInstaller(tmp_path, catalog(item)).run(["fixture"])
    assert not (tmp_path / "model.bin").exists()
    assert list((tmp_path / ".component-cache").glob("*.retained-*"))


def test_cross_process_lock_prevents_concurrent_installs(tmp_path):
    import os
    (tmp_path / ".component-lock").write_text(json.dumps({"pid": os.getpid()}))
    with pytest.raises(ValueError, match="已有"):
        resources.ComponentInstaller(tmp_path, catalog(artifact(b"x", [file("file", b"x")]))).run(["fixture"])


def test_quick_runtime_detection_and_full_verification_are_distinct(tmp_path):
    data = b"binary"
    component = catalog(artifact(data, [file("worker.exe", data)]))["components"][0]
    component["generated"] = [{**file("dependency", b"data"), "content": "data"}]
    component["probe_files"] = ["worker.exe"]
    (tmp_path / "worker.exe").write_bytes(data)
    assert resources.component_ready(tmp_path, component)
    assert not resources.component_ready(tmp_path, component, verify=True)
