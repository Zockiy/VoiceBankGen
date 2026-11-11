"""Render test score generation when OpenUtau has no verified headless adapter."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .languages import profile
from .oto import parse_oto
from .storage import atomic_bytes, write_json


class RenderBackend(BaseModel):
    model_config = ConfigDict(extra="forbid")
    engine: str = "OpenUtau"
    engine_version: str = Field(min_length=1)
    renderer: str = "WORLDLINE-R"
    base_midi: int = Field(default=57, ge=0, le=127)


def load_render_backend(path: Path) -> RenderBackend:
    backend = RenderBackend.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    if backend.engine != "OpenUtau":
        raise ValueError("only the OpenUtau manual render backend is currently supported")
    return backend


def _note(header: int, length: int, alias: str, midi: int) -> list[str]:
    return [f"[#{header:04d}]", f"Length={length}", f"Lyric={alias}", f"NoteNum={midi}"]


def prepare_render_suite(export_dir: Path, *, engine_version: str, renderer: str = "WORLDLINE-R", base_midi: int = 57) -> dict:
    bank = export_dir / "voicebank"
    manifest = json.loads((export_dir / "reports" / "build_manifest.json").read_text(encoding="utf-8"))
    encoding = profile(manifest["profile"]).oto_encoding
    lines = (bank / "oto.ini").read_bytes().decode(encoding, errors="strict").splitlines()
    aliases = {parse_oto(line).alias for line in lines}
    render_dir = export_dir / "render_test"
    render_dir.mkdir(exist_ok=True)
    required = ["あ", "い", "う", "え", "お", "か", "き", "く", "け", "こ", "さ", "し", "す", "せ", "そ", "た", "ち", "つ", "な", "に", "ら", "きゃ"]
    available = [alias for alias in required if alias in aliases]
    missing = [alias for alias in required if alias not in aliases]
    score = ["[#VERSION]", "UST Version1.2", "[#SETTING]", "Tempo=120", "Tracks=1", f"VoiceDir={bank.resolve()}", "Mode2=True"]
    index = 0
    for alias in available:
        for midi, length in ((base_midi, 144), (base_midi, 768), (base_midi - 5, 768), (base_midi + 5, 768)):
            score.extend(_note(index, length, alias, midi))
            index += 1
            score.extend(_note(index, 240, "R", midi))
            index += 1
    score.append("[#TRACKEND]")
    atomic_bytes(render_dir / "representative.ust", ("\r\n".join(score) + "\r\n").encode(encoding, errors="strict"))
    full_score = ["[#VERSION]", "UST Version1.2", "[#SETTING]", "Tempo=120", "Tracks=1", f"VoiceDir={bank.resolve()}", "Mode2=True"]
    full_index = 0
    for alias in sorted(aliases):
        full_score.extend(_note(full_index, 480, alias, base_midi))
        full_index += 1
        full_score.extend(_note(full_index, 240, "R", base_midi))
        full_index += 1
    full_score.append("[#TRACKEND]")
    atomic_bytes(render_dir / "all_aliases.ust", ("\r\n".join(full_score) + "\r\n").encode(encoding, errors="strict"))
    sequence_score = ["[#VERSION]", "UST Version1.2", "[#SETTING]", "Tempo=120", "Tracks=1", f"VoiceDir={bank.resolve()}", "Mode2=True"]
    sequences = [["あ", middle, "あ"] for middle in ("か", "さ", "な", "ら", "きゃ")]
    included_sequences = [sequence for sequence in sequences if all(alias in aliases for alias in sequence)]
    skipped_sequences = [sequence for sequence in sequences if sequence not in included_sequences]
    sequence_index = 0
    for sequence in included_sequences:
        for alias in sequence:
            sequence_score.extend(_note(sequence_index, 480, alias, base_midi))
            sequence_index += 1
        sequence_score.extend(_note(sequence_index, 240, "R", base_midi))
        sequence_index += 1
    sequence_score.append("[#TRACKEND]")
    if included_sequences:
        atomic_bytes(render_dir / "sequences.ust", ("\r\n".join(sequence_score) + "\r\n").encode(encoding, errors="strict"))
    instructions = f"""# OpenUtau 实际渲染验收

预期测试版本：{engine_version}；预期渲染器：{renderer}。这些参数尚未证明已实际渲染。

1. 在 OpenUtau 中导入 `../voicebank` 音源，确认别名可见。
2. 打开 `representative.ust` 和 `all_aliases.ust`，若存在也打开 `sequences.ust`；确认轨道选中了该音源和 {renderer}。
3. 导出 WAV，逐个听短音、长音、±5 半音、全部别名及连续过渡；记录漏音、卡顿、音高和拼接伪影。
4. 按 `render_report.template.json` 填写实际结果并保存为 `../reports/render_report.json`；`rendered_wav` 使用相对构建根目录的路径。
5. 运行 `uforge finalize ../` 检查严格发布门槛。

当前缺失的代表性别名：{', '.join(missing) if missing else '无'}。跳过的连续组合：{', '.join('→'.join(item) for item in skipped_sequences) if skipped_sequences else '无'}。缺失项不可算作通过。
"""
    atomic_bytes(render_dir / "README.md", instructions.encode("utf-8"))
    template = {"status": "pending", "import_status": "pending", "human_listening": "pending", "engine": "OpenUtau", "engine_version": "", "renderer": renderer, "bank_sha256": manifest["bank_sha256"], "tested_aliases": [], "rendered_wav": [], "observations": []}
    write_json(render_dir / "render_report.template.json", template)
    report = {"status": "pending", "engine": "OpenUtau", "engine_version_expected": engine_version, "renderer_expected": renderer, "generated_aliases": available, "skipped_aliases": missing, "included_sequences": included_sequences, "skipped_sequences": skipped_sequences, "notes": index, "rendered_wav": [], "human_listening": None}
    write_json(render_dir / "suite_manifest.json", report)
    return {"path": str(render_dir.resolve()), "generated_aliases": len(available), "skipped_aliases": len(missing), "render_status": "pending"}
