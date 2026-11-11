"""Detach imported sources with a database backup and recoverable file removal."""
from datetime import datetime, timezone
from pathlib import Path
import json
import sqlite3
from uuid import uuid4

from .models import SourceAsset, Candidate
from .storage import write_json


def removal_plan(store, source_ids: list[str]) -> dict:
    wanted = set(source_ids)
    assets = store.records("assets", SourceAsset)
    sources = [a for a in assets if a.source_id in wanted]
    if not wanted or len(sources) != len(wanted):
        raise ValueError("所选素材不存在，请刷新列表后重试")
    candidates = store.records("candidates", Candidate)
    removed = {c.candidate_id for c in candidates if c.source_id in wanted}
    def references(value):
        if isinstance(value, dict):
            return any(references(v) for v in value.values())
        if isinstance(value, list):
            return any(references(v) for v in value)
        return isinstance(value, str) and (value in wanted or value in removed)
    while True:
        dependent = {c.candidate_id for c in candidates if set(c.parent_candidate_ids) & removed or references(c.phonetic_context)}
        if dependent <= removed:
            break
        removed |= dependent
    with store.connection() as connection:
        selections = [dict(r) for r in connection.execute("SELECT * FROM bank_selections") if r["candidate_id"] in removed]
    retained_files = {(store.directory / p).resolve() for a in assets if a.source_id not in wanted for p in (a.original_path, a.canonical_path)}
    files = []
    root = store.directory.resolve()
    for asset in sources:
        for relative in (asset.original_path, asset.canonical_path):
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or path == root:
                raise ValueError("素材路径超出工程范围，已停止")
            if path not in retained_files and path.is_file() and path not in files:
                files.append(path)
    return {"source_ids": sorted(wanted), "candidate_ids": sorted(removed), "selections": len(selections),
            "sources": [{"id": a.source_id, "name": Path(a.original_path).name, "kind": a.source_kind} for a in sources],
            "files": [str(p) for p in files]}


def remove_sources(store, source_ids, bank_id, generation, emit):
    from .recycle import recycle_paths
    plan = removal_plan(store, source_ids)
    if store.bank(bank_id)["generation"] != generation:
        raise ValueError("工程已发生修改，请刷新后重新删除")
    backup = store.directory / "backups" / ("removed_sources_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid4().hex[:8])
    backup.mkdir(parents=True)
    emit(stage="备份素材与相关工程记录", current=0, total=1)
    with store.connection() as connection, sqlite3.connect(backup / "project.sqlite") as target:
        connection.backup(target)
    write_json(backup / "removal.json", plan)
    # Commit the logical removal atomically. Audio files can remain if Windows
    # refuses recycling; the backup retains all deleted database records.
    with store.connection() as connection:
        if connection.execute("SELECT generation FROM banks WHERE id=?", (bank_id,)).fetchone()[0] != generation:
            raise ValueError("备份期间工程发生修改，未删除素材")
        for identifier in plan["candidate_ids"]:
            connection.execute("DELETE FROM bank_selections WHERE candidate_id=?", (identifier,))
            connection.execute("DELETE FROM candidate_timings WHERE candidate_id=?", (identifier,))
            connection.execute("DELETE FROM candidates WHERE id=?", (identifier,))
        for identifier in plan["source_ids"]:
            connection.execute("DELETE FROM alignments WHERE json_extract(payload,'$.source_id')=?", (identifier,))
            connection.execute("DELETE FROM utterances WHERE json_extract(payload,'$.source_id')=?", (identifier,))
            connection.execute("DELETE FROM assets WHERE id=?", (identifier,))
        # Old undo entries may reference removed candidates. Retain the audit
        # records, but prevent undo from restoring dangling selections.
        identifiers = set(plan["source_ids"]) | set(plan["candidate_ids"])
        def referenced(value):
            if isinstance(value, dict):
                return any(referenced(v) for v in value.values())
            if isinstance(value, list):
                return any(referenced(v) for v in value)
            return isinstance(value, str) and value in identifiers
        for row in connection.execute("SELECT id,target,before_value,after_value FROM edit_events").fetchall():
            if row["target"] in identifiers or referenced(json.loads(row["before_value"])) or referenced(json.loads(row["after_value"])):
                connection.execute("UPDATE edit_events SET active=0 WHERE id=?", (row["id"],))
        for row in connection.execute("SELECT id FROM banks").fetchall():
            store._bump(connection, row[0])
    warning = None
    if plan["files"]:
        emit(stage="将项目内独占音频副本移入回收站", current=0, total=len(plan["files"]))
        try:
            recycle_paths([Path(p) for p in plan["files"]])
        except Exception as error:
            warning = str(error) + "；素材已移出工程，无法回收的副本保留在原处。未尝试其他删除方式。"
    emit(stage="素材已从工程移除", current=1, total=1)
    return {"removed_sources": len(plan["source_ids"]), "removed_candidates": len(plan["candidate_ids"]),
            "cleared_selections": plan["selections"], "backup": str(backup), "warning": warning}
