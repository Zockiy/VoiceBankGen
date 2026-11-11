"""A CVVC bank is a view of one CV foundation plus its own transitions."""
from __future__ import annotations

import json

from .languages import language, profile

CV_BASE_KEY = "cv_base_bank_id"


def raw_bank(store, bank_id):
    with store.connection() as connection:
        row=connection.execute("SELECT * FROM banks WHERE id=?",(bank_id,)).fetchone()
        if row is None:
            raise ValueError("找不到该音源版本")
        return store._bank(row)


def base_id(store, bank_id):
    bank=raw_bank(store,bank_id)
    root=bank["preset"].get(CV_BASE_KEY)
    if not root:
        return bank_id
    parent=raw_bank(store,root)
    if not bank["profile_id"].endswith("-cvvc") or parent["profile_id"]!=bank["profile_id"].replace("-cvvc","-cv"):
        raise ValueError("共用 CV 版本关联无效，请在音源设置重新选择")
    return root


def effective_bank(store, bank_id):
    bank=raw_bank(store,bank_id)
    root=bank["preset"].get(CV_BASE_KEY)
    if root:
        parent=raw_bank(store,root)
        if not bank["profile_id"].endswith("-cvvc") or parent["profile_id"]!=bank["profile_id"].replace("-cvvc","-cv"):
            raise ValueError("共用 CV 版本关联无效，请在音源设置重新选择")
        bank["preset"]={**parent["preset"],CV_BASE_KEY:root}
    return bank


def extra_aliases(bank):
    prof=profile(bank["profile_id"])
    return {prof.aliases(unit)[0] for unit in prof.required_units(language(prof.language_id).inventory,"core") if unit.unit_type in {"VC","VV"}}


def related_ids(connection, bank_id):
    row=connection.execute("SELECT preset FROM banks WHERE id=?",(bank_id,)).fetchone()
    root=json.loads(row[0]).get(CV_BASE_KEY) or bank_id
    return [root,*[r[0] for r in connection.execute("SELECT id FROM banks WHERE json_extract(preset,?)=?",("$."+CV_BASE_KEY,root))]]


def history_ids(connection, bank_id):
    row=connection.execute("SELECT preset FROM banks WHERE id=?",(bank_id,)).fetchone()
    root=json.loads(row[0]).get(CV_BASE_KEY)
    return [root,bank_id] if root else [bank_id]


def link_cvvc(store, bank_id, root_id):
    bank,parent=raw_bank(store,bank_id),raw_bank(store,root_id)
    if not bank["profile_id"].endswith("-cvvc") or parent["profile_id"]!=bank["profile_id"].replace("-cvvc","-cv"):
        raise ValueError("CVVC 只能关联同语言的 CV 版本")
    if bank["preset"].get(CV_BASE_KEY)==root_id:
        return
    transitions=[c for c in store.candidates(bank_id=bank_id) if c.unit_role in {"VC","VV"} and c.landmarks is not None]
    with store.connection() as connection:
        snapshot={"preset":bank["preset"],"selections":[dict(r) for r in connection.execute("SELECT * FROM bank_selections WHERE bank_id=?",(bank_id,))],
                  "note":"原标线与选音行保留；关联后基础 CV 采用主版本，VC/VV 保留本版本。"}
        connection.execute("INSERT OR IGNORE INTO metadata VALUES(?,?)",("before_shared_cv:"+bank_id,json.dumps(snapshot)))
        connection.execute("UPDATE banks SET preset=? WHERE id=?",(json.dumps({**bank["preset"],CV_BASE_KEY:root_id}),bank_id))
        for candidate in transitions:
            connection.execute("INSERT OR REPLACE INTO candidate_timings(candidate_id,marks,locks,preset_hash) VALUES(?,?,?,?)",
                               (candidate.candidate_id,candidate.landmarks.model_dump_json(),json.dumps(candidate.locked_fields),"vc-bank:"+bank_id))
        connection.execute("UPDATE edit_events SET active=0 WHERE bank_id=?",(bank_id,))
        store._bump(connection,bank_id)


def migrate_links(store):
    with store.connection() as connection:
        banks=[store._bank(r) for r in connection.execute("SELECT * FROM banks ORDER BY rowid")]
    for bank in banks:
        if not bank["profile_id"].endswith("-cvvc") or bank["preset"].get(CV_BASE_KEY):
            continue
        cv_profile=bank["profile_id"].replace("-cvvc","-cv")
        parents=[b for b in banks if b["profile_id"]==cv_profile]
        if not parents:
            parent=store.create_bank(cv_profile,copy_units_from=bank["id"])
            banks.append(parent)
            parents=[parent]
        if len(parents)==1:
            link_cvvc(store,bank["id"],parents[0]["id"])
        # Several CV foundations require an explicit choice in bank settings.
