"""Silero VAD segmentation in source canonical sample coordinates."""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
from scipy.signal import resample_poly

from .config import Config
from .models import SampleSpan, SourceAsset, Utterance
from .audio import read_region
from .storage import read_jsonl, sha256_file, write_jsonl


def _merge(timestamps: list[dict[str, int]], gap_samples: int) -> list[dict[str, int]]:
    output: list[dict[str, int]] = []
    for item in timestamps:
        if output and item["start"] - output[-1]["end"] <= gap_samples:
            output[-1]["end"] = max(output[-1]["end"], item["end"])
        else:
            output.append(dict(item))
    return output


def _split_long(project_dir: Path, asset: SourceAsset, span: SampleSpan, max_samples: int) -> list[SampleSpan]:
    if span.end - span.start <= max_samples:
        return [span]
    output: list[SampleSpan] = []
    cursor = span.start
    step = 441  # 10 ms analysis frames
    while span.end - cursor > max_samples:
        center = cursor + max_samples
        search_start = max(cursor + round(0.5 * 44100), center - round(2 * 44100))
        search_end = min(span.end - round(0.5 * 44100), center + round(2 * 44100))
        audio = read_region(project_dir, asset, search_start, search_end)
        usable = len(audio) // step
        if usable:
            frames = audio[: usable * step].reshape(usable, step)
            energies = np.mean(frames.astype("float64") ** 2, axis=1)
            cut = search_start + (int(np.argmin(energies)) + 1) * step
        else:
            cut = center
        if cut <= cursor or cut >= span.end:
            cut = center
        output.append(SampleSpan(start=cursor, end=cut))
        cursor = cut
    output.append(SampleSpan(start=cursor, end=span.end))
    return output


def segment_source(project_dir: Path, asset: SourceAsset, config: Config, manual_span: SampleSpan | None = None) -> list[Utterance]:
    existing = read_jsonl(project_dir / "utterances.jsonl", Utterance)
    if any(utt.source_id == asset.source_id for utt in existing):
        return [utt for utt in existing if utt.source_id == asset.source_id]
    if manual_span is None:
        try:
            import torch
            from silero_vad import get_speech_timestamps, load_silero_vad
        except ImportError as exc:
            raise RuntimeError("Silero VAD missing; install with uv sync --extra vad") from exc
        lock_path = project_dir / "models.lock.json"
        if lock_path.exists():
            records = json.loads(lock_path.read_text(encoding="utf-8")).get("models", [])
            locked = next((item for item in records if item["model_id"] == "silero_vad_onnx"), None)
            if locked:
                import silero_vad
                installed = Path(silero_vad.__file__).parent / "data" / "silero_vad.onnx"
                if not installed.is_file() or sha256_file(installed) != locked["files"].get("silero_vad.onnx"):
                    raise RuntimeError("installed Silero ONNX model differs from models.lock.json")
        audio = read_region(project_dir, asset, 0, asset.canonical_samples)
        analysis = resample_poly(audio, 160, 441).astype("float32")
        model = load_silero_vad(onnx=True)
        stamps = get_speech_timestamps(torch.from_numpy(analysis), model, sampling_rate=16000, speech_pad_ms=0)
        spans = _merge(stamps, round(config.vad.merge_gap_ms * 16))
        raw = [SampleSpan(start=max(0, round(s["start"] * 44100 / 16000)), end=min(asset.canonical_samples, round(s["end"] * 44100 / 16000))) for s in spans]
    else:
        if manual_span.end > asset.canonical_samples:
            raise ValueError("manual span exceeds source duration")
        raw = [manual_span]
    padding = round(config.vad.padding_ms * 44.1)
    utterances: list[Utterance] = []
    max_samples = round(config.vad.max_utterance_s * 44100)
    for initial in raw:
        for span in _split_long(project_dir, asset, initial, max(44100, max_samples - 2 * padding)):
            padded = SampleSpan(start=max(0, span.start - padding), end=min(asset.canonical_samples, span.end + padding))
            flags = ["USER_SPAN"] if manual_span else []
            if span != initial:
                flags.append("LONG_SPEECH_SPLIT_LOW_ENERGY")
            if padded.end - padded.start > config.vad.max_utterance_s * 44100:
                flags.append("LONG_SPEECH_UNSPLIT")
            utterances.append(Utterance(source_id=asset.source_id, speaker_id=asset.speaker_id, span=padded, vad_span=span, corpus_offset_samples=padded.start, language_id=asset.language_id, language_origin="user", quality_flags=flags))
    write_jsonl(project_dir / "utterances.jsonl", [*existing, *utterances])
    return utterances
