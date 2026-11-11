"""Prepare and validate time-stamped lyrics without guessing lyric text."""
from __future__ import annotations

import re
from pathlib import Path
from uuid import uuid4

from .models import SampleSpan, Utterance
from .storage import read_jsonl, write_jsonl

_STAMP = re.compile(r"\[(\d+):([0-5]\d(?:\.\d{1,3})?)\]")
_META = re.compile(r"\[(?:ar|ti|al|by|re|ve|offset):[^\]]*\]", re.I)
_CREDIT = re.compile(r"^(?:作[词詞曲]|编曲|編曲|演唱|歌手|制作|製作|混音|录音|錄音|和声|和聲|吉他|贝斯|貝斯|母带|母帶|词|詞|曲|Vocal|Lyrics|Composer|Arranger)\s*[:：]", re.I)


def parse_lrc(text: str, duration: float, *, allow_outside: bool = False) -> list[dict]:
    entries = []
    for line in text.lstrip("\ufeff").splitlines():
        line = line.strip()
        if not line:
            continue
        if _META.fullmatch(line):
            if line.lower().startswith("[offset:") and line[8:-1].strip() not in {"0", "+0", "-0"}:
                raise ValueError("暂不支持 LRC 全局 offset，请先将偏移应用到时间戳")
            continue
        matches = list(_STAMP.finditer(line))
        if not matches or "".join(m.group(0) for m in matches) != line[:matches[-1].end()]:
            raise ValueError("歌词文件需为逐行带时间戳的 LRC；请检查：" + line[:80])
        lyric = line[matches[-1].end():].strip()
        if "[" in lyric or "]" in lyric:
            raise ValueError("暂不支持逐字时间戳，请使用逐行 LRC")
        for match in matches:
            entries.append((int(match.group(1)) * 60 + float(match.group(2)), lyric))
    entries.sort(key=lambda item: item[0])
    if not entries or len({t for t, _ in entries}) != len(entries):
        raise ValueError("LRC 为空或存在重复时间戳")
    if not allow_outside and (entries[-1][0] > duration or (entries[-1][0] == duration and entries[-1][1])):
        raise ValueError("歌词时间戳超出音频长度")
    rows = []
    for index, (start, lyric) in enumerate(entries):
        end = entries[index + 1][0] if index + 1 < len(entries) else duration
        if lyric:
            rows.append({"start": start, "end": end, "text": lyric})
            if allow_outside and index == len(entries) - 1:
                rows[-1]["open_end"] = True
    if not rows:
        raise ValueError("LRC 中没有可对齐的歌词")
    return rows


def energy_spans(path: Path) -> list[tuple[float, float]]:
    """A dry-vocal fallback when speech VAD misses sustained singing."""
    import numpy as np
    import soundfile as sf
    energies = []
    with sf.SoundFile(path) as stream:
        frame = round(stream.samplerate * .02)
        for block in stream.blocks(blocksize=frame * 500, dtype="float32"):
            usable = len(block) // frame * frame
            if usable:
                energies.extend(np.sqrt(np.mean(block[:usable].reshape(-1, frame) ** 2, axis=1)))
    if not energies:
        return []
    threshold = max(float(np.percentile(energies, 95)) * .025, .0001)
    active = np.asarray(energies) > threshold
    spans, start, last = [], None, 0
    for index, voiced in enumerate(active):
        if voiced:
            if start is None:
                start = index
            last = index + 1
        elif start is not None and index - last >= 12:
            if last - start >= 6:
                spans.append((start * .02, last * .02))
            start = None
    if start is not None and last - start >= 6:
        spans.append((start * .02, last * .02))
    return spans


def estimate_offset(rows: list[dict], spans: list[tuple[float, float]]) -> dict:
    """Compare section starts separated by long pauses, tolerating missing VAD."""
    from statistics import median
    lyrics = [row for row in rows if not _CREDIT.match(row["text"].strip())]
    anchors = [row["start"] for i, row in enumerate(lyrics)
               if i == 0 or row["start"] - lyrics[i - 1]["end"] >= 3]
    sustained = [(start, end) for start, end in spans if end - start >= .25]
    onsets = [start for i, (start, end) in enumerate(sustained)
              if i == 0 or start - sustained[i - 1][1] >= 3]
    if not anchors or not onsets:
        return {"seconds": 0.0, "confidence": "低：未找到足够声音区间", "anchors": 0}
    def loss(offset):
        return sum(min(15, min(abs(t + offset - a) for a in onsets)) for t in anchors) / len(anchors) + .015 * abs(offset)
    candidates = [0.0, *[a - t for t in anchors for a in onsets if abs(a - t) <= 180]]
    best = min(candidates, key=loss)
    residuals = [min(onsets, key=lambda a: abs(t + best - a)) - t for t in anchors]
    refined = median(residuals)
    if loss(refined) <= loss(best):
        best = refined
    if loss(0) - loss(best) < .6:
        best = 0.0
    errors = [min(abs(t + best - a) for a in onsets) for t in anchors]
    supported = sum(error < 1.5 for error in errors)
    return {"seconds": round(best, 3), "confidence": "较高：多处段落相符" if supported >= 2 else "低：请试听确认",
            "anchors": len(anchors), "supported": supported, "mean_error_s": round(sum(errors) / len(errors), 3)}


def match_song_rows(rows: list[dict], spans, offset: float, duration: float) -> list[dict]:
    shifted = []
    for row in rows:
        shifted.append({**row, "start": row["start"] + offset,
                        "end": duration if row.get("open_end") else row["end"] + offset})
    matched = match_vad(shifted, spans)
    for row in matched:
        if row["end"] <= 0 or row["start"] >= duration:
            row["include"] = False
            row["match"] = "超出干声范围，默认跳过"
        row["start"] = max(0.0, min(duration, row["start"]))
        row["end"] = max(0.0, min(duration, row["end"]))
        if row.get("open_end"):
            active_ends = [b for a, b in spans if b > row["start"] and a < duration]
            if active_ends:
                row["end"] = min(duration, max(active_ends) + .15)
    return matched


def match_vad(rows: list[dict], spans: list[tuple[float, float]]) -> list[dict]:
    """Match by overlap in time, not ASR; keep the LRC window if uncertain."""
    result = []
    for row in rows:
        start, end = row["start"], row["end"]
        overlaps = [(max(start, a), min(end, b)) for a, b in spans if a < end and b > start]
        overlaps = [(a, b) for a, b in overlaps if b > a]
        candidate = dict(row)
        candidate["lyric_start"], candidate["lyric_end"] = start, end
        candidate["include"] = not bool(_CREDIT.match(row["text"].strip()))
        if overlaps:
            first, last = overlaps[0][0], overlaps[-1][1]
            # Do not remove more than 0.3 s from either LRC boundary. VAD can
            # miss sustained singing, so the reviewer sees all proposals.
            candidate["start"] = round(min(first, start + .3), 3)
            candidate["end"] = round(max(last, end - .3), 3)
            candidate["match"] = "声音时间重叠，需检查"
        else:
            candidate["match"] = "未检出声音，保留歌词时间"
        if not candidate["include"]:
            candidate["match"] = "署名信息，默认跳过"
        result.append(candidate)
    return result


def preview_song(store, audio: str, lyrics: str, language: str, emit) -> dict:
    from .audio import import_audio
    from .config import read_config
    from .segment import segment_source
    root = store.directory / "analyses" / "song-preview" / uuid4().hex
    root.mkdir(parents=True)
    emit(stage="导入歌声干声", current=0, total=0)
    asset = import_audio(root, Path(audio), store.meta("speaker_id"), language, source_kind="singing")
    try:
        content = Path(lyrics).read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("歌词文件请保存为 UTF-8 编码的 LRC") from exc
    duration = asset.canonical_samples / 44100
    rows = parse_lrc(content, duration, allow_outside=True)
    emit(stage="VAD 检测并按歌词时间匹配", current=0, total=0, device="cpu")
    utterances = segment_source(root, asset, read_config(store.directory / "config.yaml"))
    spans = [(u.vad_span.start / 44100, u.vad_span.end / 44100) for u in utterances]
    energy = energy_spans(root / asset.canonical_path)
    # Speech VAD can miss almost all of an otherwise clear singing recording.
    # Dry-vocal amplitude is then useful for matching pauses, not phonemes.
    vad_seconds = sum(b - a for a, b in spans)
    energy_seconds = sum(b - a for a, b in energy)
    activity = energy if energy_seconds > vad_seconds * 2 else spans
    method = "能量辅助（语音 VAD 漏检较多）" if activity is energy else "VAD"
    estimate = estimate_offset(rows, activity)
    return {"audio": str(Path(audio).resolve()), "audio_sha256": asset.sha256,
            "preview_audio": str(root / asset.canonical_path), "language": language,
            "duration": duration, "vad_spans": spans, "activity_spans": activity,
            "activity_method": method, "offset_estimate": estimate, "lyric_rows": rows,
            "rows": match_song_rows(rows, activity, estimate["seconds"], duration)}


def save_song_utterances(root: Path, asset, rows: list[dict]) -> None:
    existing = read_jsonl(root / "utterances.jsonl", Utterance)
    related = [u for u in existing if u.source_id == asset.source_id]
    prepared = []
    last_end = 0
    for row in rows:
        start, end = round(float(row["start"]) * 44100), round(float(row["end"]) * 44100)
        # TIFA's punctuation filter does not cover Japanese corner quotes;
        # its MeCab frontend may discard an entire line containing them.
        # Normalize only lyric input; the review still displays original LRC.
        text = str(row["text"]).translate(str.maketrans({c: " " for c in "「」『』〈〉〔〕〝〟"})).strip()
        if not text or start < last_end or start < 0 or end <= start or end > asset.canonical_samples:
            raise ValueError("歌词片段必须有文本、按时间顺序排列、互不重叠且位于音频范围内")
        span = SampleSpan(start=start, end=end)
        prepared.append(Utterance(source_id=asset.source_id, speaker_id=asset.speaker_id, span=span,
                                  vad_span=span, corpus_offset_samples=start, language_id=asset.language_id,
                                  revised_text=text, text_revision=1, quality_flags=["SINGING_USER_REVIEWED"]))
        last_end = end
    if not prepared:
        raise ValueError("没有已确认的歌词片段")
    if related:
        old = [(u.span.start, u.span.end, u.revised_text) for u in related]
        new = [(u.span.start, u.span.end, u.revised_text) for u in prepared]
        if old != new:
            raise ValueError("该歌声素材已导入且歌词片段不同；为保护已有编辑，请使用新工程或不同素材")
        return
    write_jsonl(root / "utterances.jsonl", [*existing, *prepared])
