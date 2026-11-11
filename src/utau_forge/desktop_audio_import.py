"""Import ready-made sounds as candidates without ASR or forced alignment."""
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from .audio import import_audio
from .bank_aliases import split_variant, units_by_alias
from .languages import profile
from .models import Candidate, Landmarks, SampleSpan, SourceAsset
from .storage import write_jsonl


def import_candidates(store, bank_id, generation, selection_alias, files, emit=lambda **event: None):
    bank = store.bank(bank_id)
    target_language=profile(bank["profile_id"]).language_id
    if bank["generation"] != generation:
        return {"stale": True}
    if selection_alias not in dict(store.targets(bank_id)):
        raise ValueError("请先选择左侧的目标音")
    alias, suffix = split_variant(selection_alias)
    unit = units_by_alias(bank["profile_id"]).get(alias)
    custom = next((item for item in store.custom_units(bank_id) if item["alias"] == alias), None)
    if custom:
        role, key, phones = "custom", alias, []
    elif alias == "br":
        role, key, phones = "breath", "br", ["br"]
    elif unit:
        role = unit.unit_type if unit.unit_type in {"VC", "VV"} else "sustain_vowel"
        key, phones = unit.unit_key, unit.unit_key.split("+")
    else:
        raise ValueError("此音没有可用的目标定义")
    # Share the existing source importer and immutable original/canonical files.
    # SQLite remains authoritative; synchronize the legacy asset index first.
    write_jsonl(store.directory / "sources/manifest.jsonl", store.records("assets", SourceAsset))
    imported, failures = [], []
    for index, filename in enumerate(files):
        emit(stage="导入音频候选", current=index, total=len(files), reason=Path(filename).name)
        try:
            asset = import_audio(store.directory, Path(filename), store.meta("speaker_id"), target_language)
            size = asset.canonical_samples
            fixed = size-1 if role in {"breath", "custom"} else min(size-1, 882)
            marks = Landmarks(start=0, overlap=0, preutter=0, fixed_end=fixed, end=size)
            candidate = Candidate(
                candidate_id=str(uuid5(NAMESPACE_URL, "manual-import:" + asset.sha256 + ":" + alias)),
                speaker_id=asset.speaker_id, source_id=asset.source_id, utterance_id="manual-import:" + asset.source_id,
                alignment_revision="manual-import-v1:" + asset.sha256, source_language=target_language, target_language=target_language, unit_role=role,
                unit_key=key, alias=alias, phones=phones, raw_phones=[], span=SampleSpan(start=0,end=size),
                landmarks=marks, automatic_landmarks=marks,
                phonetic_context={"manual_import": True, "manual_import_name": Path(filename).name,
                                  "ranking_reason": "手工导入音频；标线为初始估计，请试听校对"},
            )
            imported.append((asset, candidate))
        except (ValueError, OSError, RuntimeError) as error:
            failures.append({"file": str(filename), "reason": str(error)})
    if not imported:
        raise ValueError("没有导入成功的音频：" + "；".join(item["reason"] for item in failures))
    added = []
    with store.connection() as connection:
        if connection.execute("SELECT generation FROM banks WHERE id=?", (bank_id,)).fetchone()[0] != generation:
            return {"stale": True}
        for asset, candidate in imported:
            connection.execute("INSERT OR IGNORE INTO assets VALUES(?,?)", (asset.source_id, asset.model_dump_json()))
            # Reimporting the same bytes for the same sound keeps edited marks,
            # locks and review state, rather than making a duplicate row.
            if connection.execute("SELECT 1 FROM candidates WHERE id=?", (candidate.candidate_id,)).fetchone():
                continue
            connection.execute("INSERT INTO candidates VALUES(?,?,?,?)", (candidate.candidate_id, alias, store.unit_type(candidate), candidate.model_dump_json()))
            added.append(candidate.candidate_id)
        if added:
            store._bump(connection, bank_id)
    emit(stage="导入音频候选", current=len(files), total=len(files))
    return {"imported": len(added), "existing": len(imported)-len(added),
            "candidate_ids": [candidate.candidate_id for asset,candidate in imported],
            "selection_alias": selection_alias, "failures": failures}
