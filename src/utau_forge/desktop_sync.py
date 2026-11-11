"""Manifest-owned installation updates with conflict checks and a recovery journal."""

from __future__ import annotations

import json
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from .desktop_service import load_manifest, safe_path
from .desktop_store import ProjectStore, recent_projects, user_data_dir
from .oto import encode_oto, parse_oto
from .storage import atomic_bytes, sha256_file, write_json


class ExternalConflict(ValueError):
    pass


def sync_history_dir(store: ProjectStore) -> Path:
    directory=store.directory/"backups"/"sync"
    if any(parent.name.lower()=="singers" for parent in directory.parents):
        directory=user_data_dir()/"sync"/store.project_id
    return directory


def _recovery_staging(store,original):
    path=Path(original)
    if path.is_dir():
        return path
    archive=sync_history_dir(store).parent/"legacy_staging"
    for record in archive.glob("*.json"):
        migration=json.loads(record.read_text(encoding="utf-8"))
        old=Path(migration["old_root"])
        if path.is_relative_to(old):
            candidate=safe_path(archive,migration["new_directory"])/path.relative_to(old)
            if candidate.is_dir():
                return candidate
    raise ValueError("中断同步的暂存文件未找到；请保留工程同步备份并恢复暂存文件")


def relocate_legacy_staging(store,installation_id,root):
    """Preserve legacy scratch files outside the singer tree and retarget journals.

    Caller must hold installation_lock. Same-volume rename is atomic; never
    fall back to removing files if moving is unavailable.
    """
    root=Path(root).resolve()
    source=root/".uforge-staging"
    if not source.exists():
        return None
    if source.is_symlink() or source.resolve().parent!=root:
        raise ValueError("旧暂存目录是链接或越出安装目录，不能自动迁移")
    archive=(sync_history_dir(store).parent/"legacy_staging").resolve()
    if archive.is_relative_to(root) or any(parent.name.lower()=="singers" for parent in (archive,*archive.parents)):
        raise ValueError("同步恢复目录必须位于 OpenUtau Singers 之外")
    archive.mkdir(parents=True,exist_ok=True)
    destination=archive/(installation_id+"_"+uuid4().hex[:8])
    if destination.parent!=archive or destination.exists():
        raise ValueError("旧暂存迁移目标无效")
    write_json(destination.with_suffix(".json"),{"old_root":str(source),"new_directory":destination.name})
    os.rename(source,destination)
    with store.connection() as connection:
        rows=connection.execute("SELECT id,payload FROM sync_transactions WHERE installation_id=?",(installation_id,)).fetchall()
        for row in rows:
            journal=json.loads(row["payload"])
            staging=Path(journal["staging"])
            if not staging.is_relative_to(source):
                continue
            journal["staging"]=str(destination/staging.relative_to(source))
            connection.execute("UPDATE sync_transactions SET payload=? WHERE id=?",(json.dumps(journal),row["id"]))
            saved=sync_history_dir(store)/row["id"]/"journal.json"
            if saved.exists():
                write_json(saved,journal)
    return str(destination)


@contextmanager
def installation_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    stream = (root / ".uforge.lock").open("a+b")
    try:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        stream.close()


def resolve_installation(root: Path, relink: Path | None = None) -> tuple[ProjectStore, str]:
    marker, manifest = load_manifest(root)
    directory = relink or Path(recent_projects().get(marker["project_id"], ""))
    if not (directory / "project.json").is_file():
        raise FileNotFoundError("工程未找到；请选择原工程重新关联，或导入为恢复工程")
    store = ProjectStore(directory)
    if store.project_id != marker["project_id"]:
        raise ValueError("所选工程 UUID 不匹配")
    store.bank(marker["bank_id"])
    identifier = bind(store, marker["bank_id"], root, applied_revision=marker["applied_revision"])
    return store, identifier


def bind(store: ProjectStore, bank_id: str, root: Path, applied_revision: int = -1) -> str:
    root = root.resolve()
    if root.name.casefold()=="singers":
        raise ValueError("不能安装到 Singers 总目录，请选择其中独立的声库子文件夹")
    with store.connection() as connection:
        existing = connection.execute("SELECT * FROM installations WHERE path=?", (str(root),)).fetchone()
        if existing:
            if existing["bank_id"] != bank_id:
                raise ValueError("安装目录已绑定其他音源；请另选文件夹")
            return existing["id"]
        identifier = str(uuid4())
        connection.execute("INSERT INTO installations VALUES(?,?,?,?,?,?,?)", (identifier, bank_id, str(root), store.revision, applied_revision, "pending", None))
        return identifier


def installations(store: ProjectStore, bank_id: str) -> list[dict]:
    with store.connection() as connection:
        return [dict(row) for row in connection.execute("SELECT * FROM installations WHERE bank_id=?", (bank_id,))]


def _merge_oto(current: list[str], baseline: list[dict], new_entries: list[dict], force: bool) -> list[str]:
    baseline_lookup = {(entry["alias"], entry["filename"]): entry["line"] for entry in baseline}
    new_keys = {(entry["alias"], entry["filename"]): entry["line"] for entry in new_entries}
    new_aliases = {entry["alias"] for entry in new_entries}
    found: dict[tuple[str, str], int] = {}
    unknown: list[str] = []
    for line in current:
        try:
            parsed = parse_oto(line)
            key = (parsed.alias, parsed.filename)
        except ValueError:
            unknown.append(line)
            continue
        if key in baseline_lookup:
            found[key] = found.get(key, 0) + 1
            if line != baseline_lookup[key] and line not in new_keys.values() and not force:
                raise ExternalConflict(f"oto 的 {parsed.alias} 已被外部修改")
        elif key in new_keys and line == new_keys[key]:
            continue
        elif parsed.alias in new_aliases:
            if not force:
                raise ExternalConflict(f"外部 oto 与工程别名重名：{parsed.alias}")
        else:
            unknown.append(line)
    if baseline_lookup and not force and any(found.get(key, 0) != 1 for key in baseline_lookup):
        raise ExternalConflict("受管理 oto 行缺失或重复；同步已暂停")
    return [entry["line"] for entry in new_entries] + unknown


def sync_installation(store: ProjectStore, installation_id: str, *, force: bool = False) -> dict:
    with store.connection() as connection:
        row = connection.execute("SELECT * FROM installations WHERE id=?", (installation_id,)).fetchone()
        if row is None:
            raise ValueError("安装绑定不存在")
        installation = dict(row)
    bank = store.bank(installation["bank_id"])
    if not bank["build_path"]:
        raise ValueError("请先制作音源，再同步安装目录")
    from .languages import profile
    from .desktop_service import bank_encoding
    encoding=bank_encoding(bank)
    source_root = safe_path(store.directory, bank["build_path"])
    marker, new_manifest = load_manifest(source_root)
    captured_revision = store.revision
    if new_manifest["revision"] != captured_revision:
        raise ValueError("工程已有新修改；请先准备最新音源")
    root = Path(installation["path"])
    if root.name.casefold()=="singers":
        raise ValueError("不能同步到 Singers 总目录，请选择独立声库子文件夹重新绑定")
    transaction_id = str(uuid4())
    journal_dir = sync_history_dir(store) / transaction_id
    journal_dir.mkdir(parents=True)
    with installation_lock(root):
        try:
            relocated=relocate_legacy_staging(store,installation_id,root)
            adoption_path=store.directory/"analyses"/("adoption_"+installation_id+".json")
            if not force and adoption_path.exists():
                adoption=json.loads(adoption_path.read_text(encoding="utf-8"))
                if adoption["revision"] > installation["applied_revision"] and Path(adoption["root"]).resolve()==root.resolve():
                    for name,digest in adoption["hashes"].items():
                        target=safe_path(root,name)
                        if not target.is_file() or sha256_file(target)!=digest:
                            raise ExternalConflict("采用外部修改后安装目录又有变化，请重新采用或处理冲突")
                    force=True
            old_manifest = {"files": {}, "entries": []}
            recoverable_hashes: dict[str, str] = {}
            with store.connection() as connection:
                pending = connection.execute("SELECT payload FROM sync_transactions WHERE installation_id=? AND status='pending' ORDER BY rowid DESC LIMIT 1", (installation_id,)).fetchone()
            if pending:
                recovery = json.loads(pending[0])
                old_manifest = recovery["old_manifest"]
                recoverable_hashes = recovery["new_hashes"]
            elif (root / "uforge.project-link.json").exists():
                old_marker, old_manifest = load_manifest(root)
                if old_marker["project_id"] != store.project_id or old_marker["bank_id"] != bank["id"]:
                    raise ExternalConflict("此目录属于另一个工程或音源")
            actual_hashes = {}
            for relative, digest in old_manifest["files"].items():
                target = safe_path(root, relative)
                if relative == "oto.ini":
                    continue
                actual_hashes[relative] = sha256_file(target) if target.is_file() else None
                if not force and (not target.is_file() or actual_hashes[relative] not in {digest, recoverable_hashes.get(relative)}):
                    raise ExternalConflict(f"受管理文件已被外部修改：{relative}")
            current_lines = (root / "oto.ini").read_bytes().decode(old_manifest.get("oto_encoding",encoding)).splitlines() if (root / "oto.ini").exists() else []
            baseline_entries = old_manifest["entries"]
            if pending and (root / "oto.ini").is_file() and sha256_file(root / "oto.ini") == recoverable_hashes.get("oto.ini"):
                previous_staging = _recovery_staging(store,recovery["staging"])
                previous_manifest = json.loads((previous_staging / "uforge.manifest.json").read_text(encoding="utf-8"))
                baseline_entries = previous_manifest["entries"]
            merged_lines = _merge_oto(current_lines, baseline_entries, new_manifest["entries"], force)
            # Legacy recovery may contain unknown rows in the authoritative
            # build even on its first installation. Retain those as well.
            managed={(entry["alias"],entry["filename"]) for entry in new_manifest["entries"]}
            for line in (source_root/"oto.ini").read_bytes().decode(encoding).splitlines():
                try:
                    parsed=parse_oto(line)
                    unknown=(parsed.alias,parsed.filename) not in managed
                except ValueError:
                    unknown=True
                if unknown and line not in merged_lines:
                    merged_lines.append(line)
            staging = journal_dir / "staging"
            staging.mkdir(parents=True)
            changed_files = []
            for relative, digest in new_manifest["files"].items():
                if relative == "oto.ini":
                    continue
                source = safe_path(source_root, relative)
                target = safe_path(root, relative)
                actual = actual_hashes.get(relative) if relative in actual_hashes else (sha256_file(target) if target.is_file() else None)
                if actual == digest:
                    continue
                if sha256_file(source) != digest:
                    raise ValueError(f"构建文件已改变：{relative}")
                if target.exists() and relative not in old_manifest["files"] and not force:
                    raise ExternalConflict(f"目标目录有不受管理的同名文件：{relative}")
                if relative.endswith(".wav") and target.exists() and not force:
                    raise ExternalConflict(f"同名 WAV 内容不同：{relative}")
                staged = safe_path(staging, relative)
                staged.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, staged)
                changed_files.append(relative)
            atomic_bytes(staging / "oto.ini", encode_oto(merged_lines, encoding))
            merged_manifest = {**new_manifest, "files": {**new_manifest["files"], "oto.ini": sha256_file(staging / "oto.ini")}}
            write_json(staging / "uforge.manifest.json", merged_manifest)
            installed_marker = {**marker, "installation_id": installation_id, "desired_revision": captured_revision, "applied_revision": captured_revision,
                                "manifest_sha256": sha256_file(staging / "uforge.manifest.json")}
            write_json(staging / "uforge.project-link.json", installed_marker)
            for relative in (*changed_files, "oto.ini", "uforge.manifest.json", "uforge.project-link.json"):
                target = safe_path(root, relative)
                if target.is_file():
                    recovery_path = safe_path(journal_dir, relative)
                    recovery_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(target, recovery_path)
            journal = {"root": str(root), "staging": str(staging), "old_manifest": old_manifest, "new_hashes": merged_manifest["files"], "revision": captured_revision}
            write_json(journal_dir / "journal.json", journal)
            with store.connection() as connection:
                connection.execute("INSERT INTO sync_transactions VALUES(?,?,?,?,?)", (transaction_id, installation_id, captured_revision, "pending", json.dumps(journal)))
                connection.execute("UPDATE installations SET state='syncing' WHERE id=?", (installation_id,))
            if store.revision != captured_revision:
                raise ValueError("同步期间有新编辑，最新版本仍待同步")
            for relative in changed_files:
                target = safe_path(root, relative)
                atomic_bytes(target, safe_path(staging, relative).read_bytes())
            atomic_bytes(root / "oto.ini", (staging / "oto.ini").read_bytes())
            atomic_bytes(root / "uforge.manifest.json", (staging / "uforge.manifest.json").read_bytes())
            atomic_bytes(root / "uforge.project-link.json", (staging / "uforge.project-link.json").read_bytes())
            with store.connection() as connection:
                connection.execute("UPDATE sync_transactions SET status='committed' WHERE installation_id=? AND status='pending'", (installation_id,))
                connection.execute("UPDATE installations SET applied_revision=?,state=CASE WHEN desired_revision<=? THEN 'synced' ELSE 'pending' END,desired_revision=max(desired_revision,?),error=NULL WHERE id=?", (captured_revision, captured_revision,captured_revision, installation_id))
            return {"state": "synced", "revision": captured_revision, "path": str(root), "refresh_required": True,
                    "relocated_legacy_staging":relocated,
                    "copied_wavs": sum(name.endswith(".wav") for name in changed_files), "changed_files": changed_files}
        except (OSError, ValueError) as error:
            state = "conflict" if isinstance(error, ExternalConflict) else "failed"
            with store.connection() as connection:
                connection.execute("UPDATE installations SET state=?,error=? WHERE id=?", (state, str(error), installation_id))
            raise
