"""Language and export plugins are separate contracts."""

from __future__ import annotations

from typing import Any, Protocol, Sequence

from utau_forge.models import Candidate, Landmarks, MappingProposal, Phone, TargetUnit


class LanguagePack(Protocol):
    language_id: str
    version: str

    def normalize_text(self, text: str, context: dict[str, Any]) -> str: ...
    def asr_spec(self, model_lock: dict[str, Any]) -> dict[str, Any]: ...
    def g2p(self, text: str, lexicon: dict[str, str]) -> list[str]: ...
    def alignment_spec(self, model_lock: dict[str, Any]) -> dict[str, Any]: ...
    def normalize_phones(self, raw_phones: Sequence[str]) -> list[str | None]: ...
    def propose_units(self, phone_intervals: Sequence[Phone], profile: "ExportProfile") -> list[Candidate]: ...
    def timing_policy(self, unit: Candidate) -> dict[str, Any]: ...
    def phonotactic_constraints(self, target_sequence: Sequence[TargetUnit]) -> list[str]: ...


class ExportProfile(Protocol):
    profile_id: str
    version: str
    language_id: str
    oto_encoding: str

    def required_units(self, inventory: dict[str, Any], scope: str) -> list[TargetUnit]: ...
    def aliases(self, unit: TargetUnit) -> list[str]: ...
    def filename_stem(self, unit: TargetUnit) -> str: ...
    def compile_timing(self, landmarks: Landmarks, total_samples: int, filename: str, alias: str) -> str: ...
    def renderer_contract(self) -> dict[str, Any]: ...


class CrossLanguageMapper(Protocol):
    def propose(self, source_candidates: Sequence[Candidate], target_unit: TargetUnit, context: dict[str, Any], policy: dict[str, Any]) -> list[MappingProposal]: ...
    def explain(self, proposal: MappingProposal) -> str: ...


LANGUAGES: dict[str, LanguagePack] = {}
PROFILES: dict[str, ExportProfile] = {}


def register_language(pack: LanguagePack) -> None:
    if pack.language_id in LANGUAGES:
        raise ValueError(f"duplicate language pack: {pack.language_id}")
    LANGUAGES[pack.language_id] = pack


def register_profile(profile: ExportProfile) -> None:
    if profile.profile_id in PROFILES:
        raise ValueError(f"duplicate export profile: {profile.profile_id}")
    PROFILES[profile.profile_id] = profile


def language(language_id: str) -> LanguagePack:
    try:
        return LANGUAGES[language_id]
    except KeyError as exc:
        raise ValueError(f"unsupported source language: {language_id}") from exc


def profile(profile_id: str) -> ExportProfile:
    try:
        return PROFILES[profile_id]
    except KeyError as exc:
        raise ValueError(f"unsupported export profile: {profile_id}") from exc
