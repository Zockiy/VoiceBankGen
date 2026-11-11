"""Japanese CV pack; no model-specific phone guesses are hidden in core."""

from __future__ import annotations

from importlib.resources import files
from functools import lru_cache
from typing import Any, Sequence

import yaml

from utau_forge.models import Candidate, Phone, SampleSpan, TargetUnit
from utau_forge.oto import compile_oto


@lru_cache(maxsize=16)
def _load_yaml(name: str) -> dict[str, Any]:
    inventory=yaml.safe_load((files("utau_forge") / "data" / name).read_text(encoding="utf-8"))
    if not isinstance(inventory,dict):
        raise ValueError(f"语言资源为空或格式无效：{name}")
    return inventory


class JapanesePack:
    language_id = "ja"
    version = "2"

    def __init__(self) -> None:
        self.inventory = _load_yaml("ja_cv_inventory.yaml")
        self.phone_map = _load_yaml("mfa_ja_phone_map.yaml")
        self.sequence_alias = {tuple(v): k for k, v in self.inventory["units"].items()}

    def normalize_text(self, text: str, context: dict[str, Any]) -> str:
        return text.strip()

    def asr_spec(self, model_lock: dict[str, Any]) -> dict[str, Any]:
        return {"language": "ja"}

    def g2p(self, text: str, lexicon: dict[str, str]) -> list[str]:
        raise NotImplementedError("Japanese pronunciation is delegated to the locked MFA dictionary")

    def alignment_spec(self, model_lock: dict[str, Any]) -> dict[str, Any]:
        return {"backend": "mfa", "acoustic": "japanese_mfa", "dictionary": "japanese_mfa"}

    def normalize_phones(self, raw_phones: Sequence[str]) -> list[str | None]:
        labels: dict[str, str] = self.phone_map["labels"]
        return [labels.get(raw) for raw in raw_phones]

    def propose_units(self, phone_intervals: Sequence[Phone], profile: Any) -> list[Candidate]:
        if profile.language_id != self.language_id:
            raise ValueError("profile language mismatch")
        result: list[Candidate] = []
        ordered = sorted(phone_intervals, key=lambda p: p.span.start)
        i = 0
        while i < len(ordered):
            phone = ordered[i]
            canonical = phone.phoneme_id.rsplit(":", 1)[-1] if phone.phoneme_id else None
            if phone.unknown or canonical is None or canonical in {"sil", "sp", "spn", "q", "cl", "ー"}:
                i += 1
                continue
            if canonical in {"a", "e", "i", "o", "u"} and i and ordered[i - 1].unknown and phone.span.start <= ordered[i - 1].span.end + 1:
                i += 1  # Do not mistake the vowel after an unknown onset for a pure-vowel unit.
                continue
            match: tuple[str, list[Phone]] | None = None
            for size in (3, 2, 1):
                batch = ordered[i:i + size]
                if len(batch) != size or any(p.utterance_id != phone.utterance_id or p.unknown for p in batch):
                    continue
                if any(batch[j].span.end < batch[j + 1].span.start - 1 for j in range(size - 1)):
                    continue
                sequence = tuple(p.phoneme_id.rsplit(":", 1)[-1] if p.phoneme_id else "" for p in batch)
                alias = self.sequence_alias.get(sequence)
                if alias:
                    match = (alias, batch)
                    break
            if match is None:
                i += 1
                continue
            alias, batch = match
            sequence = [p.phoneme_id.rsplit(":", 1)[-1] for p in batch if p.phoneme_id]
            result.append(Candidate(
                speaker_id="",  # Filled from utterance by the pipeline.
                source_id=phone.source_id,
                utterance_id=phone.utterance_id,
                alignment_revision=phone.alignment_revision,
                source_language=self.language_id,
                unit_role="nasal" if alias == "ん" else "sustain_vowel",
                unit_key="+".join(sequence),
                alias=alias,
                phones=sequence,
                raw_phones=[p.raw_model_label for p in batch],
                span=SampleSpan(start=batch[0].span.start, end=batch[-1].span.end),
                quality_flags=["TIMING_PENDING"],
                phonetic_context={"vowel_span": batch[-1].span.model_dump(), "phone_ids": [p.phone_id for p in batch], "devoiced_vowel": "̥" in batch[-1].raw_model_label},
            ))
            i += len(batch)
        return result

    def timing_policy(self, unit: Candidate) -> dict[str, Any]:
        return {"unit_role": unit.unit_role, "cv_order": True, "max_boundary_adjust_ms": 40}

    def phonotactic_constraints(self, target_sequence: Sequence[TargetUnit]) -> list[str]:
        return ["UNSUPPORTED_UNIT" for u in target_sequence if u.unit_key not in {"+".join(v) for v in self.inventory["units"].values()}]


class JapaneseCVProfile:
    profile_id = "ja-cv"
    version = "2"
    language_id = "ja"
    oto_encoding = "cp932"

    def required_units(self, inventory: dict[str, Any], scope: str) -> list[TargetUnit]:
        if scope != "core":
            raise ValueError("only ja-cv core inventory is implemented")
        return [TargetUnit(target_language="ja", inventory_version=inventory["version"], unit_type="CV" if len(seq) > 1 else "V_or_N", unit_key="+".join(seq)) for seq in inventory["units"].values()]

    def aliases(self, unit: TargetUnit) -> list[str]:
        inventory = _load_yaml("ja_cv_inventory.yaml")
        primary = [alias for alias, seq in inventory["units"].items() if "+".join(seq) == unit.unit_key]
        return primary + [alias for alias, equivalent in inventory["alias_equivalents"].items() if equivalent in primary]

    def filename_stem(self, unit: TargetUnit) -> str:
        if not self.aliases(unit):
            raise ValueError(f"unit has no Japanese CV alias: {unit.unit_key}")
        phones = unit.unit_key.split("+")
        if phones == ["N"]:
            return "n"
        if len(phones) == 1 and phones[0] in {"a", "i", "u", "e", "o"}:
            return phones[0]
        if len(phones) == 2 and phones[1] in {"a", "i", "u", "e", "o"}:
            onset, vowel = phones
            if vowel == "i" and onset in {"ky", "gy", "ny", "hy", "by", "py", "my", "ry"}:
                onset = onset[0]
            return onset + vowel
        raise ValueError(f"unsupported Japanese CV filename unit: {unit.unit_key}")

    def compile_timing(self, landmarks: Any, total_samples: int, filename: str, alias: str) -> str:
        return compile_oto(filename, alias, landmarks, total_samples)

    def renderer_contract(self) -> dict[str, Any]:
        return {"format": "UTAU", "sample_rate": 44100, "channels": 1, "subtype": "PCM_16", "oto_encoding": self.oto_encoding}
