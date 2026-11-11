"""CVVC crossfade timing; envelopes are rendered by UTAU, not baked into WAVs."""
from __future__ import annotations

from .models import Landmarks, Phone, Utterance

DEFAULT_VC_CROSSFADE_MS = 30.0


def crossfade_marks(marks: Landmarks, fade_ms: float = DEFAULT_VC_CROSSFADE_MS) -> Landmarks:
    if not 5 <= fade_ms <= 100:
        raise ValueError("衔接淡化时长须为 5–100 ms")
    # Keep the incoming consonant/vowel boundary aligned. A positive overlap
    # makes OpenUtau crossfade both the previous tail and the current attack.
    available = marks.preutter - marks.start
    overlap = min(round(fade_ms * 44.1), max(0, available // 2))
    return marks.model_copy(update={"overlap": max(marks.overlap, marks.start + overlap)})


def propose_vc_optimization(store, bank_id: str, fade_ms: float = DEFAULT_VC_CROSSFADE_MS, emit=lambda **event: None) -> dict:
    if not store.bank(bank_id)["profile_id"].endswith("-cvvc"):
        raise ValueError("请先切换到 CVVC 音源版本")
    if not 5 <= fade_ms <= 100:
        raise ValueError("衔接淡化时长须为 5–100 ms")
    candidates = store.candidates(bank_id=bank_id)
    targets = dict(store.targets(bank_id, include_variants=False))
    transitions = [c for c in candidates if c.unit_role in {"VC","VV"} and c.alias in targets and c.review_status != "rejected"]
    onsets = {c.unit_key.split("+")[-1] for c in transitions if c.unit_role=="VC"}
    paired = [c for c in candidates if c.alias in targets and c.unit_role in {"sustain_vowel", "nasal"}
              and len(c.phones) >= 2 and c.unit_key.split("+")[0] in onsets and c.review_status != "rejected"]
    changes = []
    counts = {"vc": 0, "vv": 0, "cv": 0, "locked": 0, "insufficient": 0, "unchanged": 0}
    for index, candidate in enumerate([*transitions, *paired]):
        emit(stage="优化 VC / VV 衔接参数", current=index, total=len(transitions)+len(paired))
        if candidate.landmarks is None:
            counts["insufficient"] += 1
            continue
        if "overlap" in candidate.locked_fields:
            counts["locked"] += 1
            continue
        marks = crossfade_marks(candidate.landmarks, fade_ms)
        if marks.overlap <= marks.start:
            counts["insufficient"] += 1
            continue
        if marks == candidate.landmarks:
            counts["unchanged"] += 1
            continue
        changes.append({"candidate_id": candidate.candidate_id, "marks": marks.model_dump(), "locks": candidate.locked_fields})
        counts[candidate.unit_role.lower() if candidate.unit_role in {"VC","VV"} else "cv"] += 1
    emit(stage="优化 VC / VV 衔接参数", current=len(transitions)+len(paired), total=len(transitions)+len(paired))
    return {"changes": changes, **counts, "fade_ms": fade_ms}


def prepare_cvvc_version(store, bank_id: str, generation: int, emit=lambda **event: None) -> dict:
    """Add absent transitions from existing phones without models or reanalysis."""
    from .desktop_engine import all_transitions
    from .languages import profile
    bank = store.bank(bank_id)
    if not bank["profile_id"].endswith("-cvvc"):
        raise ValueError("连接音提取需要 CVVC 音源版本")
    if bank["generation"] != generation:
        return {"stale": True}
    code = profile(bank["profile_id"]).language_id
    phones = [p for p in store.records("alignments", Phone) if p.language_id == code]
    transitions = all_transitions(phones, {u.utterance_id: u for u in store.records("utterances", Utterance)})
    added = 0
    with store.connection() as connection:
        if connection.execute("SELECT generation FROM banks WHERE id=?", (bank_id,)).fetchone()[0] != generation:
            return {"stale": True}
        for item in transitions:
            added += connection.execute("INSERT OR IGNORE INTO candidates VALUES(?,?,?,?)",
                                        (item.candidate_id, item.alias, store.unit_type(item), item.model_dump_json())).rowcount
    store.recommend(bank_id,preserve_shared_cv=True)
    expected = store.bank(bank_id)["generation"]
    result = propose_vc_optimization(store, bank_id, emit=emit)
    transition_ids={c.candidate_id for c in store.candidates(bank_id=bank_id) if c.unit_role in {"VC","VV"}}
    changes=[change for change in result["changes"] if change["candidate_id"] in transition_ids]
    optimized = store.apply_timing_batch(bank_id, changes, expected)
    return {"transitions": len(transitions), "added": added, "optimized": optimized,
            "vc": result["vc"], "vv": result["vv"], "cv": 0, "no_alignments": not phones}
