"""Versioned project records. All stored intervals use canonical sample coordinates."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: int = 1


class SampleSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start: int = Field(ge=0)
    end: int = Field(gt=0)

    @model_validator(mode="after")
    def ordered(self) -> "SampleSpan":
        if self.end <= self.start:
            raise ValueError("span.end must exceed span.start")
        return self


class Prosody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tone: str | None = None
    stress: str | None = None
    length: str | None = None
    syllable_position: str | None = None


class Project(Record):
    project_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    speaker_id: str
    language: str = "ja"  # Compatibility only; source_languages is authoritative.
    source_languages: list[str] = Field(default_factory=lambda: ["ja"])
    target_languages: list[str] = Field(default_factory=lambda: ["ja"])
    export_profiles: list[str] = Field(default_factory=lambda: ["ja-cv"])


class SourceAsset(Record):
    source_id: str = Field(default_factory=lambda: str(uuid4()))
    original_path: str
    sha256: str
    original_sample_rate: int = Field(gt=0)
    original_channels: int = Field(gt=0)
    original_duration_s: float = Field(gt=0)
    canonical_path: str
    canonical_samples: int = Field(gt=0)
    canonical_sample_rate: int = 44100
    speaker_id: str
    recording_session: str | None = None
    language_id: str


class Utterance(Record):
    utterance_id: str = Field(default_factory=lambda: str(uuid4()))
    source_id: str
    speaker_id: str
    span: SampleSpan
    vad_span: SampleSpan
    corpus_offset_samples: int = Field(ge=0)
    language_id: str
    locale: str | None = None
    language_origin: Literal["user", "detector"] = "user"
    language_confidence: float | None = Field(default=None, ge=0, le=1)
    asr_text: str | None = None
    revised_text: str | None = None
    text_revision: int = Field(default=0, ge=0)
    quality_flags: list[str] = Field(default_factory=list)


class Phone(Record):
    phone_id: str = Field(default_factory=lambda: str(uuid4()))
    utterance_id: str
    source_id: str
    language_id: str
    phoneme_id: str | None = None
    ipa: str | None = None
    allophone: str | None = None
    features: dict[str, Any] = Field(default_factory=dict)
    raw_model_label: str
    span: SampleSpan
    alignment_revision: str
    unknown: bool = False


class Landmarks(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start: int = Field(ge=0)
    overlap: int = Field(ge=0)
    preutter: int = Field(ge=0)
    fixed_end: int = Field(ge=0)
    end: int = Field(gt=0)

    def validate_oto(self, total_samples: int) -> None:
        """Manual oto anchors are independent positions within the used span."""
        if not (0 <= self.start < self.end <= total_samples) or any(
            not self.start <= getattr(self, field) <= self.end
            for field in ("overlap", "preutter", "fixed_end")
        ):
            raise ValueError("oto 标线超出使用区间：起点必须早于终点，重叠、对拍点、固定区结束须位于起点与终点之间")

    def validate_cv(self, total_samples: int) -> None:
        if not (0 <= self.start <= self.overlap <= self.preutter <= self.fixed_end < self.end <= total_samples):
            raise ValueError("UNSUPPORTED_TIMING_STYLE: CV landmarks out of range/order")


class Metrics(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stable_vowel_ms: float | None = None
    f0_median_hz: float | None = None
    f0_mad_cents: float | None = None
    voiced_ratio: float | None = None
    periodicity: float | None = None
    rms_spread_db: float | None = None
    spectral_stability: float | None = None
    alignment_score: float | None = None
    evidence_completeness: float = Field(default=0, ge=0, le=1)
    components: dict[str, float | None] = Field(default_factory=dict)


class ProvenanceKind(str, Enum):
    original = "original"
    vowel_extended = "vowel_extended"
    cv_spliced = "cv_spliced"
    tts_generated = "tts_generated"


class Candidate(Record):
    candidate_id: str = Field(default_factory=lambda: str(uuid4()))
    speaker_id: str
    source_id: str
    utterance_id: str
    alignment_revision: str
    source_language: str
    target_language: str | None = None
    unit_role: str
    unit_key: str
    alias: str
    phones: list[str]
    raw_phones: list[str]
    canonical_sample_rate: int = 44100
    span: SampleSpan
    landmarks: Landmarks | None = None
    automatic_landmarks: Landmarks | None = None
    metrics: Metrics = Field(default_factory=Metrics)
    quality_score: float | None = Field(default=None, ge=0, le=1)
    quality_flags: list[str] = Field(default_factory=list)
    provenance_kind: ProvenanceKind = ProvenanceKind.original
    parent_candidate_ids: list[str] = Field(default_factory=list)
    review_status: Literal["unreviewed", "accepted", "rejected"] = "unreviewed"
    locked_fields: list[str] = Field(default_factory=list)
    phonetic_context: dict[str, Any] = Field(default_factory=dict)
    recording_session: str | None = None


class TargetUnit(Record):
    target_language: str
    inventory_version: str
    unit_type: str
    unit_key: str
    required_context: dict[str, Any] = Field(default_factory=dict)
    prosody: Prosody = Field(default_factory=Prosody)


class MappingProposal(Record):
    mapping_id: str = Field(default_factory=lambda: str(uuid4()))
    direction: str
    mapping_version: str
    source_candidate_ids: list[str]
    target_unit: TargetUnit
    evidence: dict[str, Any] = Field(default_factory=dict)
    equivalence_tier: Literal["L0", "L1", "L2", "L3", "L4", "missing"]
    losses: list[str] = Field(default_factory=list)
    review_status: Literal["unreviewed", "accepted", "rejected"] = "unreviewed"


class ExportEntry(Record):
    filename: str
    target_language: str
    target_alias: str
    source_candidate_ids: list[str]
    mapping_id: str | None = None
    landmarks: Landmarks
    output_samples: int
    source_to_output: dict[str, Any] = Field(default_factory=dict)
    provenance_kind: ProvenanceKind


class ReviewEvent(Record):
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    kind: Literal["edit", "choose", "undo", "reset", "transcript", "gold"]
    candidate_id: str | None = None
    utterance_id: str | None = None
    before: dict[str, Any] = Field(default_factory=dict)
    after: dict[str, Any] = Field(default_factory=dict)
