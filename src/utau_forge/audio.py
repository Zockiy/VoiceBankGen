"""Source import and canonical audio handling."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path
from uuid import uuid4

import numpy as np
import soundfile as sf

from .models import SourceAsset
from .storage import read_jsonl, sha256_file, write_jsonl


def _run(argv: list[str], timeout_s: int = 300) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout_s, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"missing executable: {argv[0]}") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"external command failed ({exc.returncode}): {exc.stderr[-4000:]}") from exc


def probe_audio(path: Path) -> dict:
    result = _run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=index,codec_name,sample_rate,channels", "-of", "json", str(path)])
    data = json.loads(result.stdout)
    streams = [stream for stream in data.get("streams", []) if "sample_rate" in stream]
    if len(streams) != 1:
        raise ValueError("expected one audio stream")
    stream = streams[0]
    return {"duration_s": float(data["format"]["duration"]), "sample_rate": int(stream["sample_rate"]), "channels": int(stream["channels"]), "codec": stream["codec_name"]}


def _check_stereo_phase(path: Path, info: dict) -> None:
    if info["channels"] != 2:
        return
    result = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-t", "10", "-ac", "2", "-ar", str(info["sample_rate"]), "-f", "f32le", "-"], capture_output=True, check=True, timeout=60)
    samples = np.frombuffer(result.stdout, dtype="<f4")
    if samples.size < 200:
        return
    stereo = samples[:samples.size // 2 * 2].reshape(-1, 2)
    left, right = stereo[:, 0], stereo[:, 1]
    rms_l = float(np.sqrt(np.mean(left * left)))
    rms_r = float(np.sqrt(np.mean(right * right)))
    if min(rms_l, rms_r) < 1e-5:
        return
    correlation = float(np.corrcoef(left, right)[0, 1])
    mix_rms = float(np.sqrt(np.mean(((left + right) / 2) ** 2)))
    if math.isfinite(correlation) and correlation < -0.8 and mix_rms < 0.25 * max(rms_l, rms_r):
        raise ValueError("STEREO_PHASE_CANCELLATION: choose a single channel before import")


def import_audio(project_dir: Path, input_path: Path, speaker_id: str, language_id: str, recording_session: str | None = None) -> SourceAsset:
    input_path = input_path.resolve(strict=True)
    info = probe_audio(input_path)
    if not math.isfinite(info["duration_s"]) or info["duration_s"] <= 0:
        raise ValueError("empty or invalid audio duration")
    _check_stereo_phase(input_path, info)
    digest = sha256_file(input_path)
    existing = read_jsonl(project_dir / "sources" / "manifest.jsonl", SourceAsset)
    for asset in existing:
        if asset.sha256 == digest and asset.speaker_id == speaker_id and asset.language_id == language_id:
            return asset
    original_dir = project_dir / "sources" / "original"
    original_dir.mkdir(parents=True, exist_ok=True)
    original = original_dir / (digest + input_path.suffix.lower())
    if not original.exists():
        shutil.copy2(input_path, original)
    canonical_dir = project_dir / "audio" / "canonical"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    canonical = canonical_dir / f"{digest}.wav"
    if not canonical.exists():
        temporary = canonical_dir / f"{digest}.{uuid4().hex}.pending.wav"
        _run(["ffmpeg", "-v", "error", "-n", "-i", str(original), "-map", "0:a:0", "-ac", "1", "-ar", "44100", "-c:a", "pcm_f32le", str(temporary)], timeout_s=max(300, int(info["duration_s"] * 3)))
        audio, sr = sf.read(temporary, dtype="float32", always_2d=False)
        if sr != 44100 or audio.ndim != 1 or audio.size == 0 or not bool(np.isfinite(audio).all()):
            raise ValueError("canonical decode failed validation")
        temporary.replace(canonical)
    sf_info = sf.info(canonical)
    asset = SourceAsset(original_path=str(original.relative_to(project_dir)), sha256=digest, original_sample_rate=info["sample_rate"], original_channels=info["channels"], original_duration_s=info["duration_s"], canonical_path=str(canonical.relative_to(project_dir)), canonical_samples=sf_info.frames, speaker_id=speaker_id, recording_session=recording_session, language_id=language_id)
    write_jsonl(project_dir / "sources" / "manifest.jsonl", [*existing, asset])
    return asset


def read_region(project_dir: Path, asset: SourceAsset, start: int, end: int) -> np.ndarray:
    if not (0 <= start < end <= asset.canonical_samples):
        raise ValueError("source sample range out of bounds")
    with sf.SoundFile(project_dir / asset.canonical_path) as audio:
        audio.seek(start)
        values = audio.read(end - start, dtype="float32")
    if len(values) != end - start or not bool(np.isfinite(values).all()):
        raise ValueError("source audio read failed")
    return values
