"""SQLite working state, legacy import and recoverable edit history."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4, uuid5, NAMESPACE_URL

from .languages import language, profile
from .models import Candidate, Landmarks, Phone, SourceAsset, Utterance
from .storage import read_jsonl, write_json
from .storage import json_hash
from .bank_aliases import BREATH_ALIAS, split_variant, export_aliases, DEFAULT_ALIAS_FORMAT, ALIAS_FORMATS, to_katakana
from .shared_banks import CV_BASE_KEY,base_id,effective_bank,extra_aliases,history_ids,related_ids,raw_bank,link_cvvc,migrate_links


SCHEMA_VERSION = 3
UNDO_LIMIT = 8


def user_data_dir() -> Path:
    directory = Path(os.environ.get("UFORGE_DATA_DIR") or Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "UTAUForge")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def register_project(project_id: str, directory: Path) -> None:
    registry_path = user_data_dir() / "registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.exists() else {}
    registry[project_id] = str(directory.resolve())
    write_json(registry_path, registry)
    hidden_path = user_data_dir() / "recent_hidden.json"
    if hidden_path.exists():
        hidden = json.loads(hidden_path.read_text(encoding="utf-8"))
        if project_id in hidden:
            write_json(hidden_path, [key for key in hidden if key != project_id])


def recent_projects() -> dict[str, str]:
    registry_path = user_data_dir() / "registry.json"
    return json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.exists() else {}


def forget_recent_project(project_id: str) -> None:
    """Remove a recent-list entry while retaining project/location records."""
    registry = recent_projects()
    if project_id in registry:
        # Installation links still need the location even after hiding history.
        hidden_path = user_data_dir() / "recent_hidden.json"
        hidden = json.loads(hidden_path.read_text(encoding="utf-8")) if hidden_path.exists() else []
        if project_id not in hidden:
            write_json(hidden_path, [*hidden, project_id])


def visible_recent_projects() -> dict[str, str]:
    hidden_path = user_data_dir() / "recent_hidden.json"
    hidden = set(json.loads(hidden_path.read_text(encoding="utf-8"))) if hidden_path.exists() else set()
    return {key: path for key, path in recent_projects().items() if key not in hidden}


class ProjectStore:
    def __init__(self, directory: Path):
        self.directory = directory.resolve()
        self.database = self.directory / "project.sqlite"
        if not self.database.exists():
            self._migrate_legacy()
        with self.connection() as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version != SCHEMA_VERSION:
                raise ValueError(f"工程数据库版本 {version} 不受支持；请使用匹配的软件版本")
            connection.execute("CREATE TABLE IF NOT EXISTS bank_units(bank_id TEXT NOT NULL,alias TEXT NOT NULL,label TEXT NOT NULL,filename_stem TEXT NOT NULL,PRIMARY KEY(bank_id,alias),UNIQUE(bank_id,label))")
            columns=connection.execute("PRAGMA table_info(candidate_timings)").fetchall()
            if next(row for row in columns if row["name"]=="preset_hash")["pk"]==0:
                connection.execute("ALTER TABLE candidate_timings RENAME TO candidate_timings_legacy")
                connection.execute("CREATE TABLE candidate_timings(candidate_id TEXT NOT NULL,marks TEXT NOT NULL,locks TEXT NOT NULL,preset_hash TEXT NOT NULL DEFAULT '',wav_hash TEXT,PRIMARY KEY(candidate_id,preset_hash))")
                connection.execute("INSERT INTO candidate_timings SELECT * FROM candidate_timings_legacy")
        migrate_links(self)
        register_project(self.project_id, self.directory)

    @contextmanager
    def connection(self):
        connection = sqlite3.connect(self.database, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @classmethod
    def create(cls, directory: Path, display_name: str, *, profile_id: str = "ja-cv", source_language: str | None = None) -> "ProjectStore":
        from .project import initialize

        if not display_name.strip():
            raise ValueError("请输入音源名称")
        target=profile(profile_id).language_id
        source_language=source_language or target
        language(source_language)
        initialize(directory, language_id=target, profile_id=profile_id, speaker_id=str(uuid4()))
        identity=json.loads((directory/"project.json").read_text(encoding="utf-8"))
        identity["source_languages"]=[source_language]
        write_json(directory/"project.json",identity)
        from .config import read_config,write_config
        config=read_config(directory/"config.yaml")
        config.source_languages=[source_language]
        write_config(directory/"config.yaml",config)
        store = cls(directory)
        store.set_meta("display_name", display_name.strip())
        with store.connection() as connection:
            connection.execute("UPDATE banks SET name=?", (display_name.strip(),))
        store._write_identity()
        return store

    def _migrate_legacy(self) -> None:
        identity_path = self.directory / "project.json"
        if not identity_path.is_file():
            raise ValueError("此目录没有可打开的工程；请选择工程文件夹或导入已安装音源")
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        backup = self.directory / "backups" / ("legacy_" + uuid4().hex)
        backup.mkdir(parents=True)
        for filename in ("project.json", "config.yaml", "candidates.jsonl", "phones.jsonl", "utterances.jsonl", "selection.json", "edits.jsonl", "sources/manifest.jsonl"):
            source = self.directory / filename
            if source.is_file():
                target = backup / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        connection = sqlite3.connect(self.database)
        try:
            connection.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
                CREATE TABLE assets(id TEXT PRIMARY KEY,payload TEXT NOT NULL);
                CREATE TABLE utterances(id TEXT PRIMARY KEY,payload TEXT NOT NULL);
                CREATE TABLE alignments(id TEXT PRIMARY KEY,payload TEXT NOT NULL);
                CREATE TABLE candidates(id TEXT PRIMARY KEY,alias TEXT NOT NULL,unit_type TEXT NOT NULL,payload TEXT NOT NULL);
                CREATE INDEX candidates_alias ON candidates(alias);
                CREATE TABLE candidate_timings(candidate_id TEXT NOT NULL,marks TEXT NOT NULL,locks TEXT NOT NULL,preset_hash TEXT NOT NULL DEFAULT '',wav_hash TEXT,PRIMARY KEY(candidate_id,preset_hash));
                CREATE TABLE banks(id TEXT PRIMARY KEY,profile_id TEXT NOT NULL,name TEXT NOT NULL,preset TEXT NOT NULL,generation INTEGER NOT NULL DEFAULT 0,build_path TEXT);
                CREATE TABLE bank_selections(bank_id TEXT NOT NULL,alias TEXT NOT NULL,candidate_id TEXT NOT NULL,entry_id TEXT NOT NULL,manual INTEGER NOT NULL DEFAULT 0,revision INTEGER NOT NULL,PRIMARY KEY(bank_id,alias));
                CREATE TABLE bank_units(bank_id TEXT NOT NULL,alias TEXT NOT NULL,label TEXT NOT NULL,filename_stem TEXT NOT NULL,PRIMARY KEY(bank_id,alias),UNIQUE(bank_id,label));
                CREATE TABLE installations(id TEXT PRIMARY KEY,bank_id TEXT NOT NULL,path TEXT NOT NULL UNIQUE,desired_revision INTEGER NOT NULL,applied_revision INTEGER NOT NULL DEFAULT -1,state TEXT NOT NULL,error TEXT);
                CREATE TABLE jobs(id TEXT PRIMARY KEY,bank_id TEXT,generation INTEGER,stage TEXT,status TEXT,payload TEXT NOT NULL);
                CREATE TABLE edit_events(id INTEGER PRIMARY KEY AUTOINCREMENT,bank_id TEXT NOT NULL,kind TEXT NOT NULL,target TEXT NOT NULL,before_value TEXT NOT NULL,after_value TEXT NOT NULL,applied INTEGER NOT NULL DEFAULT 1,active INTEGER NOT NULL DEFAULT 1,revision INTEGER NOT NULL);
                CREATE TABLE sync_transactions(id TEXT PRIMARY KEY,installation_id TEXT NOT NULL,revision INTEGER NOT NULL,status TEXT NOT NULL,payload TEXT NOT NULL);
                PRAGMA user_version=3;
            """)
            metadata = {"project_id": identity["project_id"], "speaker_id": identity["speaker_id"], "display_name": identity.get("display_name", self.directory.name), "revision": 0, "source_languages": identity.get("source_languages", ["ja"]), "target_languages": identity.get("target_languages", ["ja"])}
            for key, value in metadata.items():
                connection.execute("INSERT INTO metadata VALUES(?,?)", (key, json.dumps(value, ensure_ascii=False)))
            for table, filename, model, key in (("assets", "sources/manifest.jsonl", SourceAsset, "source_id"), ("utterances", "utterances.jsonl", Utterance, "utterance_id"), ("alignments", "phones.jsonl", Phone, "phone_id")):
                for item in read_jsonl(self.directory / filename, model):
                    connection.execute(f"INSERT INTO {table} VALUES(?,?)", (getattr(item, key), item.model_dump_json()))
            for candidate in read_jsonl(self.directory / "candidates.jsonl", Candidate):
                connection.execute("INSERT INTO candidates VALUES(?,?,?,?)", (candidate.candidate_id, candidate.alias, self.unit_type(candidate), candidate.model_dump_json()))
                if candidate.landmarks is not None:
                    connection.execute("INSERT INTO candidate_timings(candidate_id,marks,locks) VALUES(?,?,?)", (candidate.candidate_id, candidate.landmarks.model_dump_json(), json.dumps(candidate.locked_fields)))
            connection.commit()
        finally:
            connection.close()
        bank = self.create_bank(identity.get("export_profiles",["ja-cv"])[0])
        selection_path = self.directory / "selection.json"
        if selection_path.exists():
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            with self.connection() as connection:
                for alias, candidate_id in selection.get("chosen", {}).items():
                    if connection.execute("SELECT 1 FROM candidates WHERE id=?", (candidate_id,)).fetchone():
                        self._put_selection(connection, bank["id"], alias, candidate_id, alias in selection.get("manual_chosen", {}))
        self.recommend(bank["id"])
        self._write_identity()

    @staticmethod
    def unit_type(candidate: Candidate) -> str:
        if candidate.unit_role == "custom":
            return "自定义"
        if candidate.unit_role == "breath":
            return "呼吸"
        if candidate.unit_role in {"VC", "VV"}:
            return candidate.unit_role
        return "CV" if len(candidate.phones) > 1 else "V"

    def _write_identity(self) -> None:
        write_json(self.directory / "project.json", {"schema_version": SCHEMA_VERSION, "project_id": self.project_id, "speaker_id": self.meta("speaker_id"), "display_name": self.meta("display_name"), "source_languages": self.meta("source_languages"), "target_languages": self.meta("target_languages"), "database": "project.sqlite"})

    @property
    def project_id(self) -> str:
        return self.meta("project_id")

    @property
    def revision(self) -> int:
        return self.meta("revision")

    def meta(self, key: str):
        with self.connection() as connection:
            row = connection.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
            return json.loads(row[0]) if row else None

    def set_meta(self, key: str, value) -> None:
        with self.connection() as connection:
            connection.execute("INSERT OR REPLACE INTO metadata VALUES(?,?)", (key, json.dumps(value, ensure_ascii=False)))

    def banks(self) -> list[dict]:
        with self.connection() as connection:
            identifiers=[row[0] for row in connection.execute("SELECT id FROM banks ORDER BY rowid")]
        return [self.bank(identifier) for identifier in identifiers]

    @staticmethod
    def _bank(row) -> dict:
        result = dict(row)
        result["preset"] = json.loads(result["preset"])
        return result

    def bank(self, bank_id: str) -> dict:
        return effective_bank(self,bank_id)

    def cv_base_id(self, bank_id: str) -> str:
        return base_id(self,bank_id)

    def edit_owner(self, bank_id: str, candidate: Candidate) -> str:
        return bank_id if candidate.unit_role in {"VC","VV"} else base_id(self,bank_id)

    def link_cvvc(self, bank_id: str, cv_id: str) -> None:
        link_cvvc(self,bank_id,cv_id)

    def create_bank(self, profile_id: str, preset: dict | None = None, *, copy_units_from: str | None = None) -> dict:
        export_profile=profile(profile_id)
        identifier = str(uuid4())
        if profile_id.endswith("-cvvc"):
            existing=self.banks()
            parents=[b for b in existing if b["profile_id"]==profile_id.replace("-cvvc","-cv")]
            if existing and not parents:
                parent=self.create_bank(profile_id.replace("-cvvc","-cv"),copy_units_from=copy_units_from)
                copy_units_from=parent["id"]
        if profile_id.endswith("-cvvc") and not copy_units_from:
            parents=[b for b in self.banks() if b["profile_id"]==profile_id.replace("-cvvc","-cv")]
            if len(parents)==1:
                copy_units_from=parents[0]["id"]
        source_bank = self.bank(copy_units_from) if copy_units_from else None
        inherit = source_bank is not None and profile(source_bank["profile_id"]).language_id == export_profile.language_id
        source_preset = source_bank["preset"] if inherit else {}
        preset = {"pitch_mode": "preserve", "target_note": "auto", "repair": False,
                  "alias_format": "romaji" if export_profile.language_id=="zh" else DEFAULT_ALIAS_FORMAT, "include_breaths": False, **source_preset, **(preset or {})}
        preset.pop(CV_BASE_KEY,None)
        if inherit and profile_id.endswith("-cvvc"):
            root=base_id(self,copy_units_from)
            if self.bank(root)["profile_id"].endswith("-cv"):
                preset[CV_BASE_KEY]=root
        source_timings = {}
        source_selections = self.selections(copy_units_from) if inherit else {}
        if inherit:
            manual_ids = self.manual_timing_ids(copy_units_from)
            source_timings = {c.candidate_id: c for c in self.candidates(bank_id=copy_units_from) if c.candidate_id in manual_ids and c.landmarks}
        valid_aliases = {export_profile.aliases(u)[0] for u in export_profile.required_units(language(export_profile.language_id).inventory, "core")}
        if inherit:
            valid_aliases.update(u["alias"] for u in self.custom_units(copy_units_from))
            if preset.get("include_breaths"):
                valid_aliases.add(BREATH_ALIAS)
        name = self.meta("display_name") + (" 普通话" if export_profile.language_id=="zh" else "") + (" CVVC" if profile_id.endswith("-cvvc") else "")
        with self.connection() as connection:
            connection.execute("INSERT INTO banks(id,profile_id,name,preset) VALUES(?,?,?,?)", (identifier, profile_id, name, json.dumps(preset)))
            if copy_units_from:
                connection.execute("INSERT INTO bank_units SELECT ?,alias,label,filename_stem FROM bank_units WHERE bank_id=? ORDER BY rowid", (identifier, copy_units_from))
            scope = "vc-bank:"+identifier if profile_id.endswith("-cvvc") else self._bank_timing_key(identifier, preset)
            for candidate in source_timings.values():
                if preset.get(CV_BASE_KEY) and candidate.unit_role not in {"VC","VV"}:
                    continue
                connection.execute("INSERT INTO candidate_timings(candidate_id,marks,locks,preset_hash) VALUES(?,?,?,?)", (candidate.candidate_id, candidate.landmarks.model_dump_json(), json.dumps(candidate.locked_fields), scope))
            for alias, selection in source_selections.items():
                if preset.get(CV_BASE_KEY) and split_variant(alias)[0] not in extra_aliases({"profile_id":profile_id}):
                    continue
                if split_variant(alias)[0] in valid_aliases:
                    self._put_selection(connection, identifier, alias, selection["candidate_id"], True)
        targets=list(dict.fromkeys([*(self.meta("target_languages") or []),export_profile.language_id]))
        self.set_meta("target_languages",targets)
        return self.bank(identifier)

    def update_bank(self, bank_id: str, name: str, preset: dict) -> None:
        if not name.strip() or any(character in name for character in "\n\r=\x00"):
            raise ValueError("音源名称不能为空或包含换行/等号")
        if preset.get("alias_format", DEFAULT_ALIAS_FORMAT) not in ALIAS_FORMATS:
            raise ValueError("未知 oto 发音格式")
        if preset.get("include_breaths"):
            self.extract_breaths()
        with self.connection() as connection:
            owner=base_id(self,bank_id)
            settings={key:value for key,value in preset.items() if key!=CV_BASE_KEY}
            connection.execute("UPDATE banks SET preset=? WHERE id=?",(json.dumps(settings),owner))
            if owner!=bank_id:
                connection.execute("UPDATE banks SET preset=? WHERE id=?",(json.dumps({CV_BASE_KEY:owner}),bank_id))
            connection.execute("UPDATE banks SET name=? WHERE id=?",(name.strip(),bank_id))
            self._bump(connection,owner)

    def extract_breaths(self) -> int:
        from .breaths import breath_candidates
        items = breath_candidates(self.records("alignments", Phone), {utt.utterance_id: utt for utt in self.records("utterances", Utterance)})
        with self.connection() as connection:
            existing = {row[0] for row in connection.execute("SELECT id FROM candidates WHERE unit_type='呼吸'")}
        self.ingest("candidates", [item for item in items if item.candidate_id not in existing])
        return len(items)

    def targets(self, bank_id: str, *, include_variants: bool = True) -> list[tuple[str, str]]:
        bank = self.bank(bank_id)
        export_profile = profile(bank["profile_id"])
        result = [(export_profile.aliases(unit)[0], "V" if unit.unit_type=="V_or_N" else unit.unit_type) for unit in export_profile.required_units(language(export_profile.language_id).inventory, "core")]
        if bank["preset"].get("include_breaths"):
            result.append((BREATH_ALIAS, "呼吸"))
        result.extend((unit["alias"], "自定义") for unit in self.custom_units(bank_id))
        kinds = dict(result)
        if include_variants:
            variants = []
            for alias in self.selections(bank_id):
                base, suffix = split_variant(alias)
                if suffix and base in kinds:
                    variants.append((alias, kinds[base]))
            # Keep numbered alternatives next to their primary sound.
            ordered = []
            for primary in result:
                ordered.append(primary)
                children = [item for item in variants if split_variant(item[0])[0] == primary[0]]
                ordered.extend(sorted(children, key=lambda item: int(split_variant(item[0])[1][1:])))
            result = ordered
        return result

    def custom_units(self, bank_id: str) -> list[dict]:
        bank_id=base_id(self,bank_id)
        with self.connection() as connection:
            return [dict(row) for row in connection.execute("SELECT alias,label,filename_stem FROM bank_units WHERE bank_id=? ORDER BY rowid", (bank_id,))]

    def display_alias(self, bank_id: str, alias: str) -> str:
        base, suffix = split_variant(alias)
        unit = next((unit for unit in self.custom_units(bank_id) if unit["alias"] == base), None)
        if unit:
            return unit["label"] + suffix
        return to_katakana(alias) if self.bank(bank_id)["preset"].get("alias_format", DEFAULT_ALIAS_FORMAT)=="katakana" else alias

    def create_custom_unit(self, bank_id: str, label: str, stem: str | None = None) -> dict:
        bank_id=base_id(self,bank_id)
        from .custom_units import validate_unit
        label, stem = validate_unit(label, stem)
        bank = self.bank(bank_id)
        units = self.custom_units(bank_id)
        # Reserve all notation forms, breath and existing numbered aliases.
        aliases = {BREATH_ALIAS}
        for alias, kind in self.targets(bank_id):
            for notation in ALIAS_FORMATS:
                aliases.update(export_aliases(bank["profile_id"], alias, notation, units))
        if label in aliases:
            raise ValueError("此标签与已有音或导出别名重复；请直接选择已有音导入，或使用另一个标签")
        unit = {"alias": "custom:" + str(uuid4()), "label": label, "filename_stem": stem}
        with self.connection() as connection:
            self._event(connection, bank_id, "custom_unit", unit["alias"], None, unit)
            connection.execute("INSERT INTO bank_units VALUES(?,?,?,?)", (bank_id, unit["alias"], label, stem))
        return unit

    def records(self, table: str, model) -> list:
        if table not in {"assets", "utterances", "alignments", "candidates"}:
            raise ValueError("invalid record table")
        with self.connection() as connection:
            values = [model.model_validate_json(row[0]) for row in connection.execute(f"SELECT payload FROM {table}")]
            if table == "candidates":
                timings = {row["candidate_id"]: row for row in connection.execute("SELECT * FROM candidate_timings WHERE preset_hash=''")}
                for value in values:
                    if value.candidate_id in timings:
                        value.landmarks = Landmarks.model_validate_json(timings[value.candidate_id]["marks"])
                        value.locked_fields = json.loads(timings[value.candidate_id]["locks"])
            return values

    def preset_key(self,bank_id: str) -> str:
        preset=self.bank(bank_id)["preset"]
        return json_hash({key:preset.get(key) for key in ("pitch_mode","target_note")})

    @staticmethod
    def _bank_timing_key(bank_id: str, preset: dict) -> str:
        return "bank:" + bank_id + ":" + json_hash({key:preset.get(key) for key in ("pitch_mode","target_note")})

    def timing_key(self, bank_id: str) -> str:
        bank=self.bank(bank_id)
        return "vc-bank:"+bank_id if bank["profile_id"].endswith("-cvvc") else self._bank_timing_key(bank_id,bank["preset"])

    def _timing_context(self, bank_id: str):
        # One fresh bank snapshot per read; do not reopen SQLite for every
        # fallback key, and do not retain stale settings across worker edits.
        bank=self.bank(bank_id)
        owner=bank["preset"].get(CV_BASE_KEY) or bank_id
        preset=json_hash({key:bank["preset"].get(key) for key in ("pitch_mode","target_note")})
        legacy="bank:"+bank_id+":"+preset
        scope="vc-bank:"+bank_id if bank["profile_id"].endswith("-cvvc") else legacy
        body_scope="bank:"+owner+":"+preset if owner!=bank_id else scope
        return bank,preset,scope,body_scope,legacy,(body_scope if owner!=bank_id else legacy)

    def candidate(self, candidate_id: str, bank_id: str | None = None) -> Candidate:
        context=self._timing_context(bank_id) if bank_id else None
        with self.connection() as connection:
            row = connection.execute("SELECT payload FROM candidates WHERE id=?", (candidate_id,)).fetchone()
            if row is None:
                raise ValueError("找不到候选")
            item = Candidate.model_validate_json(row[0])
            if context:
                _,preset,transition,body,old_transition,old_body=context
                scope,legacy=(transition,old_transition) if item.unit_role in {"VC","VV"} else (body,old_body)
            else:
                scope=legacy=preset=""
            timing = connection.execute("SELECT marks,locks FROM candidate_timings WHERE candidate_id=? AND preset_hash IN (?, ?, ?, '') ORDER BY (preset_hash=?) DESC,(preset_hash=?) DESC,(preset_hash=?) DESC LIMIT 1", (candidate_id,scope,legacy,preset,scope,legacy,preset)).fetchone()
            if timing:
                item.landmarks = Landmarks.model_validate_json(timing[0])
                item.locked_fields = json.loads(timing[1])
            return item

    def candidates(self, alias: str | None = None, bank_id: str | None = None, *, selected_only: bool = False) -> list[Candidate]:
        if alias is not None:
            alias = split_variant(alias)[0]
        if bank_id:
            bank,preset,scope,body_scope,legacy,legacy_body=self._timing_context(bank_id)
            owner=bank["preset"].get(CV_BASE_KEY) or bank_id
        else:
            preset=scope=body_scope=legacy=legacy_body=""
            owner=None
        query="SELECT c.payload,coalesce(s.marks,l.marks,p.marks,b.marks) AS marks,coalesce(s.locks,l.locks,p.locks,b.locks) AS locks FROM candidates c LEFT JOIN candidate_timings s ON s.candidate_id=c.id AND s.preset_hash=CASE WHEN json_extract(c.payload,'$.unit_role') IN ('VC','VV') THEN ? ELSE ? END LEFT JOIN candidate_timings l ON l.candidate_id=c.id AND l.preset_hash=CASE WHEN json_extract(c.payload,'$.unit_role') IN ('VC','VV') THEN ? ELSE ? END LEFT JOIN candidate_timings p ON p.candidate_id=c.id AND p.preset_hash=? LEFT JOIN candidate_timings b ON b.candidate_id=c.id AND b.preset_hash=''"
        arguments=[scope,body_scope,legacy,legacy_body,preset]
        if alias is not None:
            query+=" WHERE c.alias=?"
            arguments.append(alias)
        if selected_only:
            if not bank_id:
                raise ValueError("读取已选候选需要音源版本")
            query += (" AND " if alias is not None else " WHERE ") + "c.id IN (SELECT candidate_id FROM bank_selections WHERE bank_id IN (?,?))"
            arguments.extend([bank_id,owner])
        items=[]
        with self.connection() as connection:
            for row in connection.execute(query,arguments):
                item=Candidate.model_validate_json(row["payload"])
                if row["marks"]:
                    item.landmarks=Landmarks.model_validate_json(row["marks"])
                    item.locked_fields=json.loads(row["locks"])
                items.append(item)
        if bank_id:
            target=profile(bank["profile_id"]).language_id
            items=[item for item in items if (item.target_language or item.source_language)==target]
            if selected_only:
                selected_ids={v["candidate_id"] for v in self.selections(bank_id).values()}
                items=[item for item in items if item.candidate_id in selected_ids]
        provenance_order={"original":0,"vowel_extended":1,"cv_spliced":2}
        return sorted((item for item in items if alias is None or item.alias == alias), key=lambda item: (item.review_status == "rejected", provenance_order.get(item.provenance_kind.value,3), -(item.quality_score or 0), item.span.start))

    def selections(self, bank_id: str) -> dict[str, dict]:
        with self.connection() as connection:
            selected={row["alias"]: dict(row) for row in connection.execute("SELECT * FROM bank_selections WHERE bank_id=?", (bank_id,))}
        owner=base_id(self,bank_id)
        if owner!=bank_id:
            extra=extra_aliases(self.bank(bank_id))
            selected={alias:value for alias,value in selected.items() if split_variant(alias)[0] in extra}
            for alias,value in self.selections(owner).items():
                selected[alias]={**value,"bank_id":bank_id,"entry_id":str(uuid5(NAMESPACE_URL,bank_id+":"+alias))}
        return selected

    def manual_timing_ids(self, bank_id: str) -> set[str]:
        """Current manual oto overrides, including shared legacy timings."""
        _,preset,scope,body_scope,legacy,legacy_body=self._timing_context(bank_id)
        with self.connection() as connection:
            return {row[0] for row in connection.execute(
                "SELECT DISTINCT t.candidate_id FROM candidate_timings t JOIN candidates c ON c.id=t.candidate_id WHERE t.preset_hash IN (CASE WHEN json_extract(c.payload,'$.unit_role') IN ('VC','VV') THEN ? ELSE ? END, CASE WHEN json_extract(c.payload,'$.unit_role') IN ('VC','VV') THEN ? ELSE ? END, ?, '')",
                (scope,body_scope,legacy,legacy_body,preset),
            )}

    def splice_source_index(self) -> list[dict]:
        """Read only list metadata; resolve full candidates/audio on selection."""
        with self.connection() as connection:
            return [dict(row) for row in connection.execute(
                "SELECT id AS candidate_id, alias, "
                "json_extract(payload,'$.source_language') AS source_language, "
                "json_extract(payload,'$.source_id') AS source_id, "
                "json_extract(payload,'$.span.start') AS start, "
                "json_extract(payload,'$.span.end') AS end, "
                "json_extract(payload,'$.unit_role') AS unit_role, "
                "json_extract(payload,'$.quality_flags') AS quality_flags "
                "FROM candidates WHERE json_extract(payload,'$.provenance_kind')='original'"
            )]

    def _put_selection(self, connection, bank_id: str, alias: str, candidate_id: str, manual: bool) -> None:
        row=connection.execute("SELECT * FROM banks WHERE id=?",(bank_id,)).fetchone()
        bank=self._bank(row)
        root=bank["preset"].get(CV_BASE_KEY)
        if root and split_variant(alias)[0] not in extra_aliases(bank):
            bank_id=root
        entry_id = str(uuid5(NAMESPACE_URL, bank_id + ":" + alias))
        revision = json.loads(connection.execute("SELECT value FROM metadata WHERE key='revision'").fetchone()[0])
        connection.execute("INSERT OR REPLACE INTO bank_selections VALUES(?,?,?,?,?,?)", (bank_id, alias, candidate_id, entry_id, int(manual), revision))

    def recommend(self, bank_id: str, *, preserve_shared_cv: bool = False) -> None:
        owner=base_id(self,bank_id)
        if owner!=bank_id and not preserve_shared_cv:
            self.recommend(owner)
        bank = self.bank(bank_id)
        selected = self.selections(bank_id)
        candidates = self.candidates(bank_id=bank_id)
        by_id={item.candidate_id:item for item in candidates}
        changed=False
        with self.connection() as connection:
            for alias, unit_type in self.targets(bank_id, include_variants=False):
                if owner!=bank_id and unit_type not in {"VC","VV"}:
                    continue
                if alias in selected and selected[alias]["manual"]:
                    continue
                usable = [item for item in candidates if item.alias == alias and item.landmarks is not None and item.review_status != "rejected" and not set(item.quality_flags) - {"COARTICULATION_UNCERTAIN", "REPAIR_REVIEW_REQUIRED", "CROSS_LANGUAGE_APPROXIMATE", "ZH_TIMING_REVIEW"} and (item.provenance_kind == "original" or bank["preset"].get("repair") or item.review_status == "accepted")]
                if alias in selected:
                    current=by_id.get(selected[alias]["candidate_id"])
                    if usable and current and current.candidate_id==usable[0].candidate_id:
                        continue
                    connection.execute("DELETE FROM bank_selections WHERE bank_id=? AND alias=?",(bank_id,alias))
                    changed=True
                if usable:
                    self._put_selection(connection, bank_id, alias, usable[0].candidate_id, False)
                    changed=True
            if changed:
                self._bump(connection,bank_id)

    def _bump(self, connection, bank_id: str, *, bump_generation: bool = True) -> int:
        revision = json.loads(connection.execute("SELECT value FROM metadata WHERE key='revision'").fetchone()[0]) + 1
        connection.execute("UPDATE metadata SET value=? WHERE key='revision'", (json.dumps(revision),))
        children=[r[0] for r in connection.execute("SELECT id FROM banks WHERE json_extract(preset,?)=?",("$."+CV_BASE_KEY,bank_id))]
        affected=[bank_id,*children]
        for identifier in affected:
            if bump_generation or identifier!=bank_id:
                connection.execute("UPDATE banks SET generation=generation+1 WHERE id=?", (identifier,))
            connection.execute("UPDATE installations SET desired_revision=?,state='pending',error=NULL WHERE bank_id=?", (revision,identifier))
        return revision

    def _event(self, connection, bank_id: str, kind: str, target: str, before, after) -> None:
        owner=base_id(self,bank_id)
        if owner!=bank_id and kind in {"timing","timing_batch","forced_missing"}:
            values=after if isinstance(after,list) else [after]
            if any((item.get("timing",item) or {}).get("preset_hash","").startswith("bank:"+owner+":") for item in values):
                bank_id=owner
        group=related_ids(connection,bank_id)
        for identifier in group:
            connection.execute("UPDATE edit_events SET active=0 WHERE bank_id=? AND applied=0", (identifier,))
        revision = self._bump(connection, bank_id)
        connection.execute("INSERT INTO edit_events(bank_id,kind,target,before_value,after_value,revision) VALUES(?,?,?,?,?,?)", (bank_id, kind, target, json.dumps(before), json.dumps(after), revision))
        # Bound the undo/redo chain; retain inactive records as project audit
        # data, without allowing older edits back into the visible history.
        self._limit_history(connection,bank_id)
        for identifier in group:
            if identifier!=bank_id:
                self._limit_history(connection,identifier)

    @staticmethod
    def _limit_history(connection,bank_id):
        identifiers=history_ids(connection,bank_id)
        placeholders=",".join("?" for _ in identifiers)
        connection.execute(f"UPDATE edit_events SET active=0 WHERE bank_id IN ({placeholders}) AND active=1 AND id NOT IN (SELECT id FROM edit_events WHERE bank_id IN ({placeholders}) AND active=1 ORDER BY id DESC LIMIT ?)", (*identifiers,*identifiers,UNDO_LIMIT))

    def choose(self, bank_id: str, candidate_id: str, selection_alias: str | None = None) -> None:
        candidate = self.candidate(candidate_id,bank_id)
        owner=self.edit_owner(bank_id,candidate)
        if owner!=bank_id:
            return self.choose(owner,candidate_id,selection_alias)
        if (candidate.target_language or candidate.source_language)!=profile(self.bank(bank_id)["profile_id"]).language_id:
            raise ValueError("候选的目标语言不属于当前版本；请先执行跨语种拼接")
        if candidate.landmarks is None or candidate.review_status == "rejected":
            raise ValueError("此候选没有可用标线或已被排除")
        alias = selection_alias or candidate.alias
        if alias not in dict(self.targets(bank_id)) or split_variant(alias)[0] != candidate.alias:
            raise ValueError("此候选不属于当前音源格式")
        before = self.selections(bank_id).get(alias)
        if before and before["candidate_id"] == candidate_id and before["manual"]:
            return
        with self.connection() as connection:
            self._event(connection, bank_id, "selection", alias, before, {"candidate_id": candidate_id, "manual": 1})
            self._put_selection(connection, bank_id, alias, candidate_id, True)

    def add_variant(self, bank_id: str, candidate_id: str) -> str:
        candidate = self.candidate(candidate_id, bank_id)
        owner=self.edit_owner(bank_id,candidate)
        if owner!=bank_id:
            return self.add_variant(owner,candidate_id)
        if (candidate.target_language or candidate.source_language)!=profile(self.bank(bank_id)["profile_id"]).language_id:
            raise ValueError("候选的目标语言不属于当前版本；请先执行跨语种拼接")
        if candidate.landmarks is None or candidate.review_status == "rejected":
            raise ValueError("此候选没有可用标线或已被排除")
        if candidate.alias not in dict(self.targets(bank_id, include_variants=False)):
            raise ValueError("请先在音源设置中启用此音型")
        with self.connection() as connection:
            existing = {row[0] for row in connection.execute("SELECT alias FROM bank_selections WHERE bank_id=?", (bank_id,))}
            units = self.custom_units(bank_id)
            bank = self.bank(bank_id)
            exported = {name for alias, kind in self.targets(bank_id) for notation in ALIAS_FORMATS for name in export_aliases(bank["profile_id"], alias, notation, units)}
            number = 1
            while f"{candidate.alias}_{number}" in existing or any(name in exported for notation in ALIAS_FORMATS for name in export_aliases(bank["profile_id"], f"{candidate.alias}_{number}", notation, units)):
                number += 1
            alias = f"{candidate.alias}_{number}"
            self._event(connection, bank_id, "selection", alias, None, {"candidate_id": candidate_id, "manual": 1})
            self._put_selection(connection, bank_id, alias, candidate_id, True)
        return alias

    def remove_variant(self, bank_id: str, alias: str) -> None:
        owner=base_id(self,bank_id)
        if owner!=bank_id and split_variant(alias)[0] not in extra_aliases(self.bank(bank_id)):
            return self.remove_variant(owner,alias)
        before = self.selections(bank_id).get(alias)
        if not split_variant(alias)[1] or before is None:
            raise ValueError("请选择一个编号备选音；主音不能用此操作移除")
        with self.connection() as connection:
            self._event(connection, bank_id, "selection", alias, before, None)
            connection.execute("DELETE FROM bank_selections WHERE bank_id=? AND alias=?", (bank_id, alias))

    def edit_timing(self, bank_id: str, candidate_id: str, marks: Landmarks, locks: list[str]) -> None:
        before, after = self._timing_edit_values(bank_id, candidate_id, marks, locks)
        if before == after:
            return
        with self.connection() as connection:
            self._event(connection, bank_id, "timing", candidate_id, before, after)
            self._restore_timing(connection, candidate_id, after)

    def _timing_edit_values(self, bank_id: str, candidate_id: str, marks: Landmarks, locks: list[str], *, candidate=None, preset=None, assets=None) -> tuple[dict, dict]:
        candidate = candidate if candidate is not None else self.candidate(candidate_id,bank_id)
        preset = preset if preset is not None else self.timing_key(self.edit_owner(bank_id,candidate))
        if not set(locks) <= set(Landmarks.model_fields):
            raise ValueError("未知锁定字段")
        for field in candidate.locked_fields:
            if field in locks and candidate.landmarks and getattr(candidate.landmarks, field) != getattr(marks, field):
                raise ValueError(f"{field} 已锁定；先取消锁定再修改")
        total = candidate.phonetic_context.get("audio_samples")
        if total is None:
            derived = candidate.phonetic_context.get("derived_audio_path")
            if derived:
                import soundfile as sf
                total = sf.info(self.directory / derived).frames
            else:
                total = assets[candidate.source_id] if assets is not None else next(item.canonical_samples for item in self.records("assets", SourceAsset) if item.source_id == candidate.source_id)
        marks.validate_oto(total)
        if not candidate.phonetic_context.get("derived_audio_path"):
            automatic=candidate.automatic_landmarks or candidate.landmarks
            lower=max(0,min(candidate.span.start,automatic.start if automatic else candidate.span.start)-4410)
            upper=min(total,max(candidate.span.end,automatic.end if automatic else candidate.span.end)+4410)
            if marks.start<lower or marks.end>upper:
                raise ValueError(f"标线超出此 WAV 的可编辑范围 {lower/44.1:.3f}–{upper/44.1:.3f} ms（源坐标）")
        before = {"marks": candidate.landmarks.model_dump() if candidate.landmarks else None, "locks": candidate.locked_fields,"preset_hash":preset}
        after = {"marks": marks.model_dump(), "locks": locks,"preset_hash":preset}
        return before, after

    @staticmethod
    def _restore_timing(connection, candidate_id: str, value: dict) -> None:
        if value["marks"] is None:
            connection.execute("DELETE FROM candidate_timings WHERE candidate_id=? AND preset_hash=?", (candidate_id, value.get("preset_hash", "")))
        else:
            connection.execute("INSERT OR REPLACE INTO candidate_timings(candidate_id,marks,locks,preset_hash) VALUES(?,?,?,?)", (candidate_id, json.dumps(value["marks"]), json.dumps(value["locks"]), value.get("preset_hash", "")))

    def apply_forced_missing(self, bank_id: str, proposals: list[dict], expected_generation: int) -> int:
        """Publish missing oto and selections atomically as one undo step."""
        targets = dict(self.targets(bank_id, include_variants=False))
        selected = self.selections(bank_id)
        before, after = [], []
        aliases = set()
        for proposal in proposals:
            alias, identifier = proposal["alias"], proposal["candidate_id"]
            candidate = self.candidate(identifier, bank_id)
            if (alias in aliases or alias in selected or alias not in targets or candidate.alias != alias
                    or (candidate.target_language or candidate.source_language) != profile(self.bank(bank_id)["profile_id"]).language_id):
                raise ValueError("批量标注的音节已选用、重复或不属于当前版本")
            if candidate.landmarks is not None or candidate.review_status == "rejected" or "NO_SUSTAIN_REGION" not in candidate.quality_flags:
                raise ValueError("批量标注只处理因未找到稳定元音而缺失、没有 oto 的候选")
            aliases.add(alias)
            old, new = self._timing_edit_values(bank_id, identifier, Landmarks.model_validate(proposal["marks"]), [])
            owner=self.edit_owner(bank_id,candidate)
            before.append({"bank_id":owner,"alias": alias, "candidate_id": identifier, "timing": old, "selection": None})
            after.append({"bank_id":owner,"alias": alias, "candidate_id": identifier, "timing": new, "selection": {"candidate_id": identifier, "manual": 1}})
        with self.connection() as connection:
            if connection.execute("SELECT generation FROM banks WHERE id=?", (bank_id,)).fetchone()[0] != expected_generation:
                raise ValueError("分析期间工程发生修改，未覆盖当前编辑；可再次强制标注 oto")
            if not after:
                return 0
            self._event(connection, bank_id, "forced_missing", "missing_sustain", before, after)
            for item in after:
                self._restore_timing(connection, item["candidate_id"], item["timing"])
                self._put_selection(connection, item["bank_id"], item["alias"], item["candidate_id"], True)
        return len(after)

    def apply_timing_batch(self, bank_id: str, changes: list[dict], expected_generation: int) -> int:
        before, after, identifiers = [], [], set()
        code = profile(self.bank(bank_id)["profile_id"]).language_id
        # Read the batch snapshot once instead of reopening SQLite and parsing
        # every source record for each candidate in the GUI result callback.
        candidates = {c.candidate_id:c for c in self.candidates(bank_id=bank_id)} if changes else {}
        assets = {a.source_id:a.canonical_samples for a in self.records("assets", SourceAsset)} if changes else {}
        preset = self.timing_key(bank_id)
        owner=base_id(self,bank_id)
        body_preset=self.timing_key(owner) if owner!=bank_id else preset
        for change in changes:
            identifier = change["candidate_id"]
            candidate = candidates.get(identifier)
            if candidate is None:
                raise ValueError("找不到候选")
            if identifier in identifiers or (candidate.target_language or candidate.source_language) != code:
                raise ValueError("批量标线含重复候选或其他语言候选")
            identifiers.add(identifier)
            key=preset if candidate.unit_role in {"VC","VV"} else body_preset
            old, new = self._timing_edit_values(bank_id, identifier, Landmarks.model_validate(change["marks"]), change["locks"], candidate=candidate, preset=key, assets=assets)
            if old != new:
                before.append({"candidate_id": identifier, "timing": old})
                after.append({"candidate_id": identifier, "timing": new})
        with self.connection() as connection:
            if connection.execute("SELECT generation FROM banks WHERE id=?", (bank_id,)).fetchone()[0] != expected_generation:
                raise ValueError("分析期间工程发生修改，请重试；当前编辑保留")
            if after:
                self._event(connection, bank_id, "timing_batch", "cvvc_crossfade", before, after)
                for item in after:
                    self._restore_timing(connection, item["candidate_id"], item["timing"])
        return len(after)

    def history(self, bank_id: str, redo: bool = False) -> bool:
        with self.connection() as connection:
            self._limit_history(connection,bank_id)
            identifiers=history_ids(connection,bank_id)
            placeholders=",".join("?" for _ in identifiers)
            row = connection.execute(f"SELECT * FROM edit_events WHERE bank_id IN ({placeholders}) AND active=1 AND applied=? ORDER BY id " + ("ASC" if redo else "DESC") + " LIMIT 1", (*identifiers, 0 if redo else 1)).fetchone()
            if row is None:
                return False
            value = json.loads(row["after_value" if redo else "before_value"])
            bank_id=row["bank_id"]
            self._bump(connection, bank_id)
            if row["kind"] == "selection":
                if value is None:
                    connection.execute("DELETE FROM bank_selections WHERE bank_id=? AND alias=?", (bank_id, row["target"]))
                else:
                    self._put_selection(connection, bank_id, row["target"], value["candidate_id"], bool(value["manual"]))
            elif row["kind"] == "custom_unit":
                if value is None:
                    connection.execute("DELETE FROM bank_units WHERE bank_id=? AND alias=?", (bank_id, row["target"]))
                else:
                    connection.execute("INSERT OR REPLACE INTO bank_units VALUES(?,?,?,?)", (bank_id, value["alias"], value["label"], value["filename_stem"]))
            elif row["kind"] == "timing":
                self._restore_timing(connection, row["target"], value)
            elif row["kind"] == "timing_batch":
                for item in value:
                    self._restore_timing(connection, item["candidate_id"], item["timing"])
            elif row["kind"] == "forced_missing":
                for item in value:
                    self._restore_timing(connection, item["candidate_id"], item["timing"])
                    if item["selection"] is None:
                        connection.execute("DELETE FROM bank_selections WHERE bank_id=? AND alias=?", (item.get("bank_id",bank_id), item["alias"]))
                    else:
                        self._put_selection(connection, item.get("bank_id",bank_id), item["alias"], item["selection"]["candidate_id"], bool(item["selection"]["manual"]))
            elif row["kind"]=="review":
                candidate=Candidate.model_validate_json(connection.execute("SELECT payload FROM candidates WHERE id=?",(row["target"],)).fetchone()[0])
                candidate.review_status=value
                connection.execute("UPDATE candidates SET payload=? WHERE id=?",(candidate.model_dump_json(),candidate.candidate_id))
            connection.execute("UPDATE edit_events SET applied=? WHERE id=?", (int(redo), row["id"]))
            return True

    def ingest(self, table: str, items: list) -> None:
        key = {"assets": "source_id", "utterances": "utterance_id", "alignments": "phone_id", "candidates": "candidate_id"}[table]
        with self.connection() as connection:
            for item in items:
                if table == "candidates":
                    previous = connection.execute("SELECT payload FROM candidates WHERE id=?", (item.candidate_id,)).fetchone()
                    if previous:
                        existing = Candidate.model_validate_json(previous[0])
                        item.review_status = existing.review_status
                    connection.execute("INSERT OR REPLACE INTO candidates VALUES(?,?,?,?)", (item.candidate_id, item.alias, self.unit_type(item), item.model_dump_json()))
                else:
                    connection.execute(f"INSERT OR REPLACE INTO {table} VALUES(?,?)", (getattr(item, key), item.model_dump_json()))

    def exclude(self, bank_id: str, candidate_id: str) -> None:
        self.review(bank_id,candidate_id,"rejected")

    def review(self,bank_id: str,candidate_id: str,status: str):
        if status not in {"accepted","rejected","unreviewed"}:
            raise ValueError("未知校对状态")
        candidate = self.candidate(candidate_id)
        bank_id=self.edit_owner(bank_id,candidate)
        before=candidate.review_status
        if before == status:
            return
        candidate.review_status = status
        with self.connection() as connection:
            self._event(connection,bank_id,"review",candidate_id,before,status)
            connection.execute("UPDATE candidates SET payload=? WHERE id=?", (candidate.model_dump_json(), candidate_id))

    def backup(self, destination: Path) -> Path:
        destination.mkdir(parents=True, exist_ok=False)
        with self.connection() as source:
            target = sqlite3.connect(destination / "project.sqlite")
            try:
                source.backup(target)
            finally:
                target.close()
        shutil.copy2(self.directory / "project.json", destination / "project.json")
        for name in ("sources", "audio", "derived", "config.yaml", "models.lock.json"):
            source = self.directory / name
            if source.is_dir():
                shutil.copytree(source, destination / name)
            elif source.is_file():
                shutil.copy2(source, destination / name)
        return destination
