"""Pinned component downloads and recoverable installation, without GUI dependencies."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tarfile
import threading
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from uuid import uuid4

CHUNK = 1024 * 1024


class DownloadCancelled(Exception):
    pass


def data_directory() -> Path:
    return Path(os.environ.get("UFORGE_DATA_DIR") or Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "UTAUForge")


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + "." + uuid4().hex + ".pending")
    with pending.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2))
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists():
        history = path.parent / ".history"
        history.mkdir(exist_ok=True)
        path.replace(history / (path.name + "." + uuid4().hex))
    pending.replace(path)


def load_preferences() -> dict:
    path = data_directory() / "downloads.json"
    current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return {"source": current.get("source", "official"), "github_mirror": current.get("github_mirror", ""),
            "hf_mirror": current.get("hf_mirror", ""), "other_mirror": current.get("other_mirror", ""),
            "proxy": current.get("proxy", "")}


def save_preferences(values: dict) -> dict:
    result = {key: str(values.get(key, "")).strip() for key in ("source", "github_mirror", "hf_mirror", "other_mirror", "proxy")}
    if result["source"] not in {"official", "mirror"}:
        raise ValueError("请选择官方上游或镜像站")
    for key in ("github_mirror", "hf_mirror", "other_mirror"):
        if result[key]:
            validate_url(result[key].replace("{url}", "https://github.com/test"))
    if result["proxy"]:
        validate_proxy(result["proxy"])
    save_json(data_directory() / "downloads.json", result)
    return result


def validate_proxy(url: str) -> None:
    parts = urllib.parse.urlsplit(url)
    try:
        port = parts.port
    except ValueError as error:
        raise ValueError("代理端口无效") from error
    if (parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password
            or parts.path not in {"", "/"} or parts.query or parts.fragment or port is None or port < 1):
        raise ValueError("代理须为 HTTP/HTTPS 地址和端口，例如 http://127.0.0.1:7890；SOCKS 地址请改用代理软件的 HTTP 端口")


def open_download(request, preferences, cancel=None):
    proxy = preferences.get("proxy", "").strip()
    if proxy:
        validate_proxy(proxy)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
        open_request = opener.open
    else:
        open_request = urllib.request.urlopen
    for attempt in range(3):
        checkpoint(cancel)
        try:
            return open_request(request, timeout=30)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            if isinstance(error, urllib.error.HTTPError) and error.code not in {429, 500, 502, 503, 504}:
                raise
            if attempt == 2:
                raise
            if cancel is not None:
                cancel.wait(.5 * (attempt + 1))
            else:
                threading.Event().wait(.5 * (attempt + 1))


def validate_url(url: str) -> None:
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in {"https", "http"} or not parts.hostname or parts.username or parts.password:
        raise ValueError("下载地址须为 HTTP(S) URL，不能包含用户名或密码")
    if parts.scheme == "http" and parts.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("远程下载源必须使用 HTTPS")


def download_url(url: str, preferences: dict) -> str:
    validate_url(url)
    if preferences.get("source") != "mirror":
        return url
    parts = urllib.parse.urlsplit(url)
    if parts.hostname == "github.com":
        mirror = preferences.get("github_mirror", "").strip()
        if not mirror:
            raise ValueError("请填写 GitHub 镜像地址，或切换官方上游")
        result = mirror.replace("{url}", url) if "{url}" in mirror else mirror.rstrip("/") + "/" + url
    elif parts.hostname == "huggingface.co":
        mirror = preferences.get("hf_mirror", "").strip()
        if not mirror:
            raise ValueError("Whisper 来自 Hugging Face；请填写其镜像地址，或切换官方上游")
        result = mirror.rstrip("/") + parts.path + ("?" + parts.query if parts.query else "")
    else:
        mirror = preferences.get("other_mirror", "").strip()
        # Some users mirror GitHub only; Python/PyPI/PyTorch can stay upstream.
        result = (mirror.replace("{url}", url) if "{url}" in mirror else mirror.rstrip("/") + "/" + url) if mirror else url
    validate_url(result)
    return result


def safe_relative(name: str) -> str:
    path = PurePosixPath(name)
    if (not name or "\\" in name or ":" in name or path.is_absolute()
            or any(part in {"", ".", ".."} or part.endswith((".", " ")) for part in name.split("/"))
            or any(re.fullmatch(r"(?i)(con|prn|aux|nul|com[1-9]|lpt[1-9])(\..*)?", part) for part in path.parts)):
        raise ValueError(f"组件路径不安全：{name}")
    return path.as_posix()


def destination(root: Path, name: str) -> Path:
    path = root / safe_relative(name)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("组件路径越出安装目录")
    return path


def digest(path: Path, cancel: threading.Event | None = None) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK):
            checkpoint(cancel)
            result.update(chunk)
    return result.hexdigest()


def checkpoint(cancel) -> None:
    if cancel is not None and cancel.is_set():
        raise DownloadCancelled("已取消，下载缓存保留；再次点击可继续")


def component_files(component: dict) -> list[dict]:
    return [item for artifact in component.get("artifacts", []) for item in artifact["files"]] + component.get("generated", [])


def validate_catalog(catalog: dict) -> dict:
    if catalog.get("schema") != 1 or not isinstance(catalog.get("components"), list):
        raise ValueError("不支持的组件清单")
    paths, identifiers = set(), set()
    for component in catalog["components"]:
        key = component["id"]
        if not re.fullmatch(r"[a-z0-9_-]+", key) or key in identifiers:
            raise ValueError("组件标识重复或无效")
        identifiers.add(key)
        for artifact in component.get("artifacts", []):
            if artifact["format"] not in {"zip", "tar", "raw"} or artifact["size"] <= 0:
                raise ValueError("组件格式或大小无效")
            if "range_size" in artifact and (not isinstance(artifact["range_size"], int) or artifact["range_size"] <= 0):
                raise ValueError("下载分段大小无效")
            if artifact.get("url"):
                validate_url(artifact["url"])
            safe_relative(artifact["cache"])
            if artifact.get("sha256") and not re.fullmatch("[a-f0-9]{64}", artifact["sha256"]):
                raise ValueError("组件归档哈希无效")
            if artifact["format"] == "raw" and len(artifact["files"]) != 1:
                raise ValueError("原始文件组件须只有一个目标文件")
        for item in component_files(component):
            path = safe_relative(item["path"])
            if path.casefold() in paths or path.startswith(".component-"):
                raise ValueError("组件目标路径重复或占用内部目录")
            paths.add(path.casefold())
            if item["size"] < 0 or not re.fullmatch("[a-f0-9]{64}", item["sha256"]):
                raise ValueError("组件文件缺少固定大小或 SHA-256")
            if "content" in item and (len(item["content"].encode("utf-8")) != item["size"]
                                      or hashlib.sha256(item["content"].encode("utf-8")).hexdigest() != item["sha256"]):
                raise ValueError("生成文件校验失败")
        available = {item["path"] for item in component_files(component)}
        if any(name not in available for name in component.get("probe_files", [])):
            raise ValueError("环境检测文件不在固定清单中")
    return catalog


def load_catalog(root: Path) -> dict:
    local = root / "components.json"
    if not local.is_file():
        local = Path(__file__).parent / "data/resource-catalog.json"
    catalog = validate_catalog(json.loads(local.read_text(encoding="utf-8")))
    if not (root / "components.json").is_file():
        runtime = Path(__file__).parent / "data/runtime-catalog.json"
        if runtime.is_file():
            runtimes = validate_catalog(json.loads(runtime.read_text(encoding="utf-8")))
            identifiers = {c["id"] for c in catalog["components"]}
            catalog["components"] += [c for c in runtimes["components"] if c["id"] not in identifiers]
            validate_catalog(catalog)
    # Existing CUDA installations can still provide shared model files. Do not
    # offer their obsolete CUDA runtime for download/repair in this fork.
    catalog["components"] = [c for c in catalog["components"] if c["id"] != "tifa-gpu"]
    return catalog


def component_ready(root: Path, component: dict, *, verify=False, cancel=None) -> bool:
    files = component_files(component)
    # Thousands of runtime files must not block opening the settings window.
    # Installation and explicit repair always perform full cryptographic checks.
    if not verify and component.get("probe_files"):
        probes = set(component["probe_files"])
        files = [item for item in files if item["path"] in probes]
    for item in files:
        path = destination(root, item["path"])
        if not path.is_file() or path.stat().st_size != item["size"]:
            return False
        if verify and digest(path, cancel) != item["sha256"]:
            return False
    return bool(files)


def preserve(path: Path) -> None:
    if path.exists():
        path.replace(path.with_name(path.name + ".retained-" + uuid4().hex))


def fetch(artifact: dict, cache: Path, preferences: dict, cancel, emit) -> Path:
    """Range resume is accepted only if the server confirms the exact offset."""
    path = destination(cache, artifact["cache"])
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = artifact["size"]
    if path.is_file() and path.stat().st_size == expected:
        if not artifact.get("sha256") or digest(path, cancel) == artifact["sha256"]:
            emit(stage="缓存", current=expected, total=expected)
            return path
        preserve(path)
    if not artifact.get("url"):
        raise ValueError("此环境由单文件安装器提供；请重新运行安装器修复环境")
    url = download_url(artifact["url"], preferences)
    pending = path.with_name(path.name + ".part")
    offset = pending.stat().st_size if pending.exists() else 0
    if offset == expected and (not artifact.get("sha256") or digest(pending, cancel) == artifact["sha256"]):
        pending.replace(path)
        emit(stage="缓存", current=expected, total=expected)
        return path
    if offset >= expected:
        preserve(pending)
        offset = 0
    try:
        while offset < expected:
            checkpoint(cancel)
            block = artifact.get("range_size", 0)
            end = min(expected - 1, offset + block - 1) if block else expected - 1
            headers = {"User-Agent": "VoiceBankGen/1.0", "Accept-Encoding": "identity"}
            if offset or block:
                headers["Range"] = f"bytes={offset}-{end}" if block else f"bytes={offset}-"
            with open_download(urllib.request.Request(url, headers=headers), preferences, cancel) as response:
                validate_url(response.url)
                code = response.status
                if code == 200:
                    if offset:
                        preserve(pending)
                        offset = 0
                    end = expected - 1
                elif code == 206:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                    if not match or int(match[1]) != offset or int(match[3]) != expected or int(match[2]) != end:
                        preserve(pending)
                        raise ValueError("下载源返回错误的续传范围；缓存已保留，请重试")
                else:
                    raise ValueError(f"下载源返回 HTTP {code}")
                length = response.headers.get("Content-Length")
                if length is not None and int(length) != end + 1 - offset:
                    raise ValueError("下载文件大小与固定版本不符")
                with pending.open("ab" if offset else "wb") as stream:
                    emit(stage="下载", current=offset, total=expected)
                    while True:
                        checkpoint(cancel)
                        chunk = response.read(CHUNK)
                        if not chunk:
                            break
                        stream.write(chunk)
                        offset += len(chunk)
                        if offset > end + 1:
                            raise ValueError("下载文件超出预期大小")
                        emit(stage="下载", current=offset, total=expected)
                if offset != end + 1:
                    raise OSError("下载未完成，缓存已保留，可继续下载")
        if offset != expected:
            raise OSError("下载未完成，缓存已保留，可继续下载")
        emit(stage="校验下载", current=0, total=expected)
        if artifact.get("sha256") and digest(pending, cancel) != artifact["sha256"]:
            preserve(pending)
            raise ValueError("下载 SHA-256 不匹配；不会安装，请检查下载源")
        pending.replace(path)
        return path
    except urllib.error.URLError as error:
        raise OSError(f"连接下载源失败：{error.reason}；可改镜像或重试，缓存保留") from error


def unpack(artifact: dict, archive: Path, stage: Path, cancel, emit) -> None:
    if artifact["format"] == "raw":
        item = artifact["files"][0]
        path = destination(stage, item["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        with archive.open("rb") as source, path.open("wb") as output:
            while chunk := source.read(CHUNK):
                checkpoint(cancel)
                output.write(chunk)
        return
    if artifact["format"] == "tar":
        with tarfile.open(archive, "r:gz") as bundle:
            entries = bundle.getmembers()
            names = set()
            for info in entries:
                safe_relative(info.name.rstrip("/"))
                if info.name.casefold() in names or not (info.isfile() or info.isdir()):
                    raise ValueError("TAR 包含重复文件、链接或特殊文件")
                names.add(info.name.casefold())
            by_name = {info.name: info for info in entries}
            for index, item in enumerate(artifact["files"]):
                checkpoint(cancel)
                info = by_name.get(item.get("member", item["path"]))
                if info is None or not info.isfile() or info.size != item["size"]:
                    raise ValueError("归档内容不符合固定版本")
                path = destination(stage, item["path"])
                path.parent.mkdir(parents=True, exist_ok=True)
                with bundle.extractfile(info) as source, path.open("wb") as output:
                    while chunk := source.read(CHUNK):
                        checkpoint(cancel)
                        output.write(chunk)
                emit(stage="解压", current=index + 1, total=len(artifact["files"]))
        return
    with zipfile.ZipFile(archive) as bundle:
        entries = bundle.infolist()
        by_name = {info.filename: info for info in entries}
        names = set()
        for info in entries:
            safe_relative(info.filename.rstrip("/"))
            if info.filename.casefold() in names or stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError("ZIP 包含重复文件或符号链接")
            names.add(info.filename.casefold())
        for index, item in enumerate(artifact["files"]):
            checkpoint(cancel)
            suffix = item.get("member", item["path"])
            exact = by_name.get(suffix)
            matches = ([exact] if exact and not exact.is_dir() else
                       [info for info in entries if not info.is_dir() and info.filename.endswith("/" + suffix)])
            if len(matches) != 1 or matches[0].file_size != item["size"]:
                raise ValueError(f"归档内容不符合固定版本：{suffix}")
            path = destination(stage, item["path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(matches[0]) as source, path.open("wb") as output:
                while chunk := source.read(CHUNK):
                    checkpoint(cancel)
                    output.write(chunk)
            emit(stage="解压", current=index + 1, total=len(artifact["files"]))


def _pid_alive(pid: int) -> bool:
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.OpenProcess.restype = ctypes.c_void_p
        ctypes.windll.kernel32.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
        ctypes.windll.kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return ctypes.windll.kernel32.GetLastError() == 5
        code = ctypes.c_ulong()
        try:
            return not ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value == 259
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


class ComponentInstaller:
    def __init__(self, root: Path, catalog: dict, preferences=None, cancel=None, emit=lambda **event: None):
        self.root = root.resolve()
        self.catalog = validate_catalog(catalog)
        self.preferences = preferences or load_preferences()
        self.cancel = cancel or threading.Event()
        self.emit = emit
        self.cache = self.root / ".component-cache"

    def run(self, identifiers: list[str], *, repair=False, prepare=None) -> list[str]:
        components = {c["id"]: c for c in self.catalog["components"]}
        if set(identifiers) - components.keys():
            raise ValueError("未知组件")
        self.root.mkdir(parents=True, exist_ok=True)
        lock = self.root / ".component-lock"
        if lock.exists():
            state = json.loads(lock.read_text(encoding="utf-8"))
            if _pid_alive(state["pid"]):
                raise ValueError("此目录已有下载/安装任务，请等待完成")
            preserve(lock)
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid()}, stream)
        completed = []
        try:
            self.recover()
            if prepare is not None:
                prepare()
            for identifier in identifiers:
                checkpoint(self.cancel)
                component = components[identifier]
                self.emit(component=identifier, name=component["name"], stage="检测", current=0, total=1)
                if not repair and component_ready(self.root, component, verify=True, cancel=self.cancel):
                    completed.append(identifier)
                    self.emit(stage="已就绪", current=1, total=1)
                    continue
                self.install(component)
                completed.append(identifier)
            return completed
        finally:
            preserve(lock)

    def recover(self) -> None:
        """Recover a process interrupted between file renames before retrying."""
        for journal in (self.root / ".component-backups").glob("*/installation.json"):
            state = json.loads(journal.read_text(encoding="utf-8"))
            if state.get("phase") != "installing":
                continue
            backup = journal.parent
            stage = self.root / ".component-staging" / backup.name
            for item in reversed(state["files"]):
                target = destination(self.root, item["path"])
                old = destination(backup, item["path"])
                pending = destination(stage, item["path"])
                if old.exists() or not pending.exists():
                    if target.exists():
                        if pending.exists():
                            preserve(pending)
                        pending.parent.mkdir(parents=True, exist_ok=True)
                        target.replace(pending)
                    if old.exists():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        old.replace(target)
            save_json(journal, {"phase": "rolled_back", "component": state["component"]})

    def install(self, component: dict) -> None:
        stage = self.root / ".component-staging" / (component["id"] + "-" + uuid4().hex)
        stage.mkdir(parents=True)
        files = component_files(component)
        size = sum(item["size"] for item in files)
        download_size = sum(a["size"] for a in component["artifacts"] if not destination(self.cache, a["cache"]).is_file())
        if shutil.disk_usage(self.root).free < size + download_size + 64 * CHUNK:
            raise OSError("安装磁盘空间不足，需容纳下载缓存和解压后的组件")
        for artifact in component["artifacts"]:
            archive = fetch(artifact, self.cache, self.preferences, self.cancel, self.emit)
            try:
                unpack(artifact, archive, stage, self.cancel, self.emit)
            except (ValueError, zipfile.BadZipFile):
                if not artifact.get("sha256"):
                    preserve(archive)
                raise
        for item in component.get("generated", []):
            path = destination(stage, item["path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(item["content"].encode("utf-8"))
        for index, item in enumerate(files):
            checkpoint(self.cancel)
            path = destination(stage, item["path"])
            if path.stat().st_size != item["size"] or digest(path, self.cancel) != item["sha256"]:
                for artifact in component["artifacts"]:
                    if item in artifact["files"]:
                        preserve(destination(self.cache, artifact["cache"]))
                raise ValueError(f"组件文件校验失败：{item['path']}；原有文件未改变")
            self.emit(stage="校验文件", current=index + 1, total=len(files))
        checkpoint(self.cancel)
        # Commit is deliberately non-cancellable; rollback restores all old files.
        backup = self.root / ".component-backups" / stage.name
        backup.mkdir(parents=True)
        journal = backup / "installation.json"
        installed, moved = [], []
        try:
            save_json(journal, {"component": component["id"], "phase": "installing", "files": files})
            for item in files:
                name = item["path"]
                target = destination(self.root, name)
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() and not target.is_file():
                    raise ValueError(f"安装目标被目录占用：{name}")
                if target.exists():
                    old = destination(backup, name)
                    old.parent.mkdir(parents=True, exist_ok=True)
                    target.replace(old)
                    moved.append(name)
                destination(stage, name).replace(target)
                installed.append(name)
                self.emit(stage="安装（请等待）", current=len(installed), total=len(files))
            save_json(journal, {"component": component["id"], "phase": "complete"})
        except BaseException:
            for name in reversed(installed):
                target = destination(self.root, name)
                retained = destination(stage, name)
                target.replace(retained)
            for name in reversed(moved):
                destination(backup, name).replace(destination(self.root, name))
            save_json(journal, {"component": component["id"], "phase": "rolled_back"})
            raise
        self.emit(stage="已就绪", current=1, total=1)
