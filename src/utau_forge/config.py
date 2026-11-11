"""Strict, versioned configuration. Values here are initial engineering settings."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .storage import atomic_bytes


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AudioConfig(StrictModel):
    canonical_sample_rate: int = 44100
    export_subtype: Literal["PCM_16"] = "PCM_16"
    denoise: bool = False


class VadConfig(StrictModel):
    padding_ms: int = Field(default=150, ge=0)
    merge_gap_ms: int = Field(default=250, ge=0)
    max_utterance_s: float = Field(default=20, gt=0)


class AsrConfig(StrictModel):
    backend: Literal["faster_whisper"] = "faster_whisper"
    model: str = "large-v3"
    device: str = "cpu"
    compute_type: str = "int8"


class AlignmentConfig(StrictModel):
    backend: Literal["mfa"] = "mfa"
    acoustic: str = "japanese_mfa"
    dictionary: str = "japanese_mfa"
    max_boundary_adjust_ms: int = Field(default=40, ge=0)
    executable: str = "mfa"
    root_directory: str | None = None
    timeout_s: int = Field(default=3600, gt=0)


class FeaturesConfig(StrictModel):
    hop_ms: int = Field(default=5, gt=0)
    f0_backend: str = "world_harvest"
    f0_min_hz: int = Field(default=60, gt=0)
    f0_max_hz: int = Field(default=1000, gt=0)


class SustainConfig(StrictModel):
    min_ms: int = Field(default=80, gt=0)
    preferred_ms: int = Field(default=150, gt=0)
    min_voiced_ratio: float = Field(default=0.85, ge=0, le=1)
    min_periodicity: float = Field(default=0.60, ge=0, le=1)
    max_f0_mad_cents: float = Field(default=45, gt=0)
    max_rms_robust_spread_db: float = Field(default=4, gt=0)


class SelectionConfig(StrictModel):
    top_k: int = Field(default=3, gt=0)


class RepairConfig(StrictModel):
    vowel_tail: bool = True
    cv_splice: bool = False
    min_donor_ms: int = Field(default=150, gt=0)
    target_sustain_ms: int = Field(default=250, gt=0)
    crossfade_ms: float | None = Field(default=None, ge=1, le=200)


class PitchConfig(StrictModel):
    mode: Literal["flatten", "preserve"] = "preserve"
    target_note: str = "auto"
    warn_shift_semitones: float = 5


class LevelConfig(StrictModel):
    target_vowel_rms_dbfs: float = -20
    max_gain_db: float = 9
    max_peak_dbfs: float = -1


class ExportConfig(StrictModel):
    oto_encoding: str = "cp932"
    cutoff_style: Literal["negative"] = "negative"
    release_mode: Literal["draft", "strict"] = "draft"


class RuntimeConfig(StrictModel):
    offline: bool = False
    gpu_jobs: int = Field(default=1, ge=1)


class CrossLanguageConfig(StrictModel):
    enabled: bool = False
    mapping_policy: Literal["strict", "approximate"] = "strict"
    generation_policy: Literal["disabled", "missing_only"] = "disabled"
    require_same_speaker: bool = True
    mapping_pack: str | None = None


class Config(StrictModel):
    schema_version: Literal[1] = 1
    language: str = "ja"
    source_languages: list[str] = Field(default_factory=lambda: ["ja"])
    target_languages: list[str] = Field(default_factory=lambda: ["ja"])
    export_profiles: list[str] = Field(default_factory=lambda: ["ja-cv"])
    inventory: str = "ja-cv-core-v1"
    audio: AudioConfig = Field(default_factory=AudioConfig)
    vad: VadConfig = Field(default_factory=VadConfig)
    asr: AsrConfig = Field(default_factory=AsrConfig)
    alignment: AlignmentConfig = Field(default_factory=AlignmentConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    sustain: SustainConfig = Field(default_factory=SustainConfig)
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    repair: RepairConfig = Field(default_factory=RepairConfig)
    pitch: PitchConfig = Field(default_factory=PitchConfig)
    level: LevelConfig = Field(default_factory=LevelConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    cross_language: CrossLanguageConfig = Field(default_factory=CrossLanguageConfig)


def read_config(path: Path) -> Config:
    return Config.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def write_config(path: Path, config: Config) -> None:
    atomic_bytes(path, yaml.safe_dump(config.model_dump(mode="json"), allow_unicode=True, sort_keys=False).encode("utf-8"))
