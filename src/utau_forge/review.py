"""Append-only human review events and a local-only Gradio frontend."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import spectrogram

from .audio import read_region
from .evaluation import GoldAnnotation, create_gold_template, get_gold_annotation, update_gold_annotation
from .models import Candidate, Landmarks, ReviewEvent, SourceAsset, Utterance
from .storage import read_jsonl, write_json, write_jsonl


def _append_event(project_dir: Path, event: ReviewEvent) -> None:
    path = project_dir / "edits.jsonl"
    write_jsonl(path, [*read_jsonl(path, ReviewEvent), event])


def edit_candidate(project_dir: Path, candidate_id: str, *, landmarks: Landmarks | None = None, review_status: str | None = None, locked_fields: list[str] | None = None) -> Candidate:
    candidates = read_jsonl(project_dir / "candidates.jsonl", Candidate)
    target = next((candidate for candidate in candidates if candidate.candidate_id == candidate_id), None)
    if target is None:
        raise ValueError("candidate not found")
    before = target.model_dump(mode="json")
    if landmarks is not None:
        sources = {asset.source_id: asset for asset in read_jsonl(project_dir / "sources" / "manifest.jsonl", SourceAsset)}
        derived = target.phonetic_context.get("derived_audio_path")
        total = sf.info(project_dir / derived).frames if derived else sources[target.source_id].canonical_samples
        landmarks.validate_oto(total)
        desired_locks = locked_fields if locked_fields is not None else target.locked_fields
        for field in target.locked_fields:
            if field in desired_locks and target.landmarks and getattr(landmarks, field) != getattr(target.landmarks, field):
                raise ValueError(f"field {field} is locked; unlock it before changing the value")
        target.landmarks = landmarks
        target.quality_flags = [flag for flag in target.quality_flags if flag not in {"NO_SUSTAIN_REGION", "TIMING_PENDING"}]
    if review_status is not None:
        if review_status not in {"unreviewed", "accepted", "rejected"}:
            raise ValueError("invalid review status")
        target.review_status = review_status
        if review_status == "accepted":
            target.quality_flags = [flag for flag in target.quality_flags if flag != "REPAIR_REVIEW_REQUIRED"]
    if locked_fields is not None:
        valid = set(Landmarks.model_fields)
        if not set(locked_fields) <= valid:
            raise ValueError("unknown landmark lock field")
        target.locked_fields = sorted(set(locked_fields))
    write_jsonl(project_dir / "candidates.jsonl", candidates)
    _append_event(project_dir, ReviewEvent(kind="edit", candidate_id=candidate_id, before=before, after=target.model_dump(mode="json")))
    return target


def choose_candidate(project_dir: Path, candidate_id: str) -> None:
    candidates = read_jsonl(project_dir / "candidates.jsonl", Candidate)
    candidate = next((item for item in candidates if item.candidate_id == candidate_id), None)
    if candidate is None:
        raise ValueError("candidate not found")
    path = project_dir / "selection.json"
    selection = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"schema_version": 1, "chosen": {}, "manual_chosen": {}, "top_candidates": {}}
    previous = selection["chosen"].get(candidate.alias)
    manual_previous = selection.setdefault("manual_chosen", {}).get(candidate.alias)
    selection["chosen"][candidate.alias] = candidate_id
    selection["manual_chosen"][candidate.alias] = candidate_id
    write_json(path, selection)
    _append_event(project_dir, ReviewEvent(kind="choose", candidate_id=candidate_id, before={"chosen": previous, "manual_chosen": manual_previous, "alias": candidate.alias}, after={"chosen": candidate_id, "manual_chosen": candidate_id, "alias": candidate.alias}))


def revise_transcript(project_dir: Path, utterance_id: str, text: str) -> Utterance:
    utterances = read_jsonl(project_dir / "utterances.jsonl", Utterance)
    target = next((item for item in utterances if item.utterance_id == utterance_id), None)
    if target is None:
        raise ValueError("utterance not found")
    before = {"text": target.revised_text, "revision": target.text_revision}
    target.revised_text = text
    target.text_revision += 1
    write_jsonl(project_dir / "utterances.jsonl", utterances)
    _append_event(project_dir, ReviewEvent(kind="transcript", utterance_id=utterance_id, before=before, after={"text": text, "revision": target.text_revision}))
    return target


def reset_landmark(project_dir: Path, candidate_id: str, field: str) -> Candidate:
    candidate = next((item for item in read_jsonl(project_dir / "candidates.jsonl", Candidate) if item.candidate_id == candidate_id), None)
    if candidate is None or candidate.automatic_landmarks is None or candidate.landmarks is None:
        raise ValueError("no automatic landmark proposal exists")
    if field not in Landmarks.model_fields:
        raise ValueError("unknown landmark field")
    marks = candidate.landmarks.model_copy(update={field: getattr(candidate.automatic_landmarks, field)})
    return edit_candidate(project_dir, candidate_id, landmarks=marks, locked_fields=[name for name in candidate.locked_fields if name != field])


def undo_last_edit(project_dir: Path) -> str:
    events = read_jsonl(project_dir / "edits.jsonl", ReviewEvent)
    if not events:
        raise ValueError("no edits to undo")
    last = events[-1]
    if last.kind == "edit" and last.candidate_id:
        candidates = read_jsonl(project_dir / "candidates.jsonl", Candidate)
        position = next((index for index, item in enumerate(candidates) if item.candidate_id == last.candidate_id), None)
        if position is None:
            raise ValueError("edited candidate is stale; manual recovery required")
        candidates[position] = Candidate.model_validate(last.before)
        write_jsonl(project_dir / "candidates.jsonl", candidates)
    elif last.kind == "choose" and last.candidate_id:
        path = project_dir / "selection.json"
        selection = json.loads(path.read_text(encoding="utf-8"))
        alias = last.before["alias"]
        if last.before["chosen"] is None:
            selection["chosen"].pop(alias, None)
        else:
            selection["chosen"][alias] = last.before["chosen"]
        previous_manual = last.before.get("manual_chosen")
        if previous_manual is None:
            selection.setdefault("manual_chosen", {}).pop(alias, None)
        else:
            selection.setdefault("manual_chosen", {})[alias] = previous_manual
        write_json(path, selection)
    elif last.kind == "transcript" and last.utterance_id:
        utterances = read_jsonl(project_dir / "utterances.jsonl", Utterance)
        target = next(item for item in utterances if item.utterance_id == last.utterance_id)
        target.revised_text = last.before["text"]
        target.text_revision = last.before["revision"]
        write_jsonl(project_dir / "utterances.jsonl", utterances)
    elif last.kind == "gold" and last.candidate_id:
        gold_path = project_dir / "evaluation" / "gold_annotations.jsonl"
        records = read_jsonl(gold_path, GoldAnnotation)
        position = next((index for index, row in enumerate(records) if row.candidate_id == last.candidate_id), None)
        if position is None:
            raise ValueError("gold annotation is missing; manual recovery required")
        if last.before:
            records[position] = GoldAnnotation.model_validate(last.before)
        else:
            records.pop(position)
        write_jsonl(gold_path, records)
    else:
        raise ValueError("latest event cannot be undone automatically")
    _append_event(project_dir, ReviewEvent(kind="undo", candidate_id=last.candidate_id, utterance_id=last.utterance_id, before=last.after, after=last.before))
    return last.kind


def build_draft_from_review(project_dir: Path, pitch_mode: str, target_note: str,
                            include_unreviewed_splices: bool = False, singer_name: str | None = None) -> dict:
    from .build import build, select
    from .project import load

    if pitch_mode not in {"preserve", "flatten"}:
        raise ValueError("音高模式只能选择保留原音高或统一音高")
    note = target_note.strip() if pitch_mode == "flatten" else "auto"
    if pitch_mode == "flatten" and not note:
        raise ValueError("统一音高需要填写目标音符（如 A3）或 auto")
    config = load(project_dir)[1]
    select(project_dir, config)
    result = build(project_dir, config, pitch_mode, note, "draft",
                   include_unreviewed_splices=include_unreviewed_splices, singer_name=singer_name)
    return result


def generate_splices_from_review(project_dir: Path, enabled: bool) -> dict:
    from .project import load
    from .repair import repair_project

    if not enabled:
        return {"enabled": False, "message": "先勾选实验性自动拼接，再生成候选。"}
    return {"enabled": True, **repair_project(project_dir, load(project_dir)[1], mode="cv-splice", experimental=True)}


def build_review_ui(project_dir: Path):
    try:
        import gradio as gr
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise RuntimeError("Review UI missing; install with uv sync --extra review") from exc
    assets = {asset.source_id: asset for asset in read_jsonl(project_dir / "sources" / "manifest.jsonl", SourceAsset)}
    gold_file = project_dir / "evaluation" / "gold_annotations.jsonl"
    if not gold_file.exists() and (project_dir / "candidates.jsonl").exists():
        create_gold_template(project_dir)

    def aliases_now() -> list[str]:
        return sorted({item.alias for item in read_jsonl(project_dir / "candidates.jsonl", Candidate)})

    def list_candidates(alias: str, show_all: bool = False) -> list[str]:
        items = read_jsonl(project_dir / "candidates.jsonl", Candidate)
        ranked = sorted((c for c in items if c.alias == alias), key=lambda c: (c.review_status != "accepted", -(c.quality_score or 0)))
        return [item.candidate_id for item in (ranked if show_all else ranked[:3])]

    def display(candidate_id: str):
        item = next(c for c in read_jsonl(project_dir / "candidates.jsonl", Candidate) if c.candidate_id == candidate_id)
        source = assets[item.source_id]
        derived = item.phonetic_context.get("derived_audio_path")
        if derived:
            audio, rate = sf.read(project_dir / derived, dtype="float32")
            if rate != 44100 or audio.ndim != 1:
                raise ValueError("invalid derived candidate WAV")
            start = 0
        else:
            start = max(0, item.span.start - 4410)
            end = min(source.canonical_samples, item.span.end + 4410)
            audio = read_region(project_dir, source, start, end)
        x = (np.arange(len(audio)) + start) / 44100
        figure = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.38, 0.62], vertical_spacing=0.06)
        figure.add_trace(go.Scatter(x=x[::20], y=audio[::20], mode="lines", name="waveform"), row=1, col=1)
        frequencies, times, power = spectrogram(audio, 44100, nperseg=1024, noverlap=768)
        mask = frequencies <= 8000
        figure.add_trace(go.Heatmap(x=times + start / 44100, y=frequencies[mask], z=10 * np.log10(np.maximum(power[mask], 1e-12)), colorscale="Viridis", showscale=False, name="spectrogram"), row=2, col=1)
        if item.landmarks:
            for name in Landmarks.model_fields:
                figure.add_vline(x=getattr(item.landmarks, name) / 44100, annotation_text=name, row=1, col=1)
                figure.add_vline(x=getattr(item.landmarks, name) / 44100, row=2, col=1)
        figure.update_layout(title=f"{item.alias} | {', '.join(item.quality_flags) or 'no flags'}", height=540)
        figure.update_xaxes(title_text="母版时间（秒）" if not derived else "派生音频时间（秒）", row=2, col=1)
        figure.update_yaxes(title_text="Hz", row=2, col=1)
        clip_dir = project_dir / "audio" / "analysis"
        clip_dir.mkdir(parents=True, exist_ok=True)
        clip = clip_dir / f"review_{candidate_id}.wav"
        if not clip.exists():
            sf.write(clip, audio, 44100, subtype="PCM_16")
        marks = item.landmarks or Landmarks(start=item.span.start, overlap=item.span.start, preutter=item.span.start, fixed_end=item.span.start, end=item.span.end)
        utterances = {utt.utterance_id: utt for utt in read_jsonl(project_dir / "utterances.jsonl", Utterance)}
        details = {"candidate_id": item.candidate_id, "score": item.quality_score, "metrics": item.metrics.model_dump(), "flags": item.quality_flags, "source_id": item.source_id, "utterance_id": item.utterance_id, "raw_phones": item.raw_phones, "provenance": item.provenance_kind.value, "parent_candidates": item.parent_candidate_ids}
        return (figure, str(clip), *[getattr(marks, field) for field in Landmarks.model_fields], item.review_status, item.locked_fields, utterances[item.utterance_id].revised_text or utterances[item.utterance_id].asr_text or "", details)

    def save(candidate_id: str, *values):
        numbers = values[:5]
        status, locks = values[5:]
        marks = Landmarks(**dict(zip(Landmarks.model_fields, (int(value) for value in numbers), strict=True)))
        edit_candidate(project_dir, candidate_id, landmarks=marks, review_status=status, locked_fields=locks)
        return "编辑已追加到 edits.jsonl"

    def display_gold(candidate_id: str):
        row = get_gold_annotation(project_dir, candidate_id)
        sustainable = "yes" if row.sustainable is True else "no" if row.sustainable is False else "unreviewed"
        p = row.preutter_range or (None, None)
        f = row.fixed_end_range or (None, None)
        return row.gold_alias or "", sustainable, p[0], p[1], f[0], f[1], row.split, row.rejection_reason or ""

    def save_gold(candidate_id: str, gold_alias: str, sustainable: str, p_min, p_max, f_min, f_max, split: str, reason: str):
        def interval(left, right):
            if left is None and right is None:
                return None
            if left is None or right is None:
                raise ValueError("金标准区间需要同时填写最小值与最大值")
            return (int(left), int(right))

        row = GoldAnnotation(candidate_id=candidate_id, gold_alias=gold_alias.strip() or None,
                             sustainable={"yes": True, "no": False, "unreviewed": None}[sustainable],
                             preutter_range=interval(p_min, p_max), fixed_end_range=interval(f_min, f_max),
                             split=split, rejection_reason=reason.strip() or None)
        update_gold_annotation(project_dir, row)
        return "金标准标注已保存；运行 uforge evaluate 查看指标"

    def step_field(candidate_id: str, field: str, direction: int) -> str:
        item = next(c for c in read_jsonl(project_dir / "candidates.jsonl", Candidate) if c.candidate_id == candidate_id)
        if item.landmarks is None:
            raise ValueError("此候选还没有五标线，请先输入数值")
        marks = item.landmarks.model_copy(update={field: getattr(item.landmarks, field) + direction * 220})
        edit_candidate(project_dir, candidate_id, landmarks=marks)
        return f"{field} 已移动约 {direction * 5} ms"

    def revise_and_realign(candidate_id: str, text: str):
        from .align import align
        from .analyze import analyze_project
        from .build import select
        from .project import load

        item = next(c for c in read_jsonl(project_dir / "candidates.jsonl", Candidate) if c.candidate_id == candidate_id)
        revise_transcript(project_dir, item.utterance_id, text)
        config = load(project_dir)[1]
        result = align(project_dir, config)
        analyze_project(project_dir, config)
        select(project_dir, config)
        all_aliases = aliases_now()
        current_alias = item.alias if item.alias in all_aliases else (all_aliases[0] if all_aliases else None)
        ids = list_candidates(current_alias) if current_alias else []
        return (f"已重对齐 {result['utterances']} 条语句；请复核新候选", gr.update(choices=all_aliases, value=current_alias), gr.update(choices=ids, value=ids[0] if ids else None))

    aliases = aliases_now()
    display_outputs = []

    with gr.Blocks(title="UTAU Forge Review") as ui:
        gr.Markdown("# 日语 CV 审阅：标线单位为 44.1 kHz 母版的绝对样本")
        gr.Markdown("操作步骤见工程根目录 `CONTROL_PANEL_MANUAL.md`：可选择尝试拼接缺音、审核候选、选择音高模式构建，再导入 OpenUtau 试听。")
        with gr.Accordion("可选：自动拼接缺失音节（实验性）", open=False):
            gr.Markdown("根据缺失音节的辅音与元音，搜索同说话人、同语言、相同辅音类别及音高相近的素材。拼接音会保留双亲来源并标记 COARTICULATION_UNCERTAIN；生成后必须试听并将候选设为 accepted，才会进入新草稿。失败的音节仍列为缺音。")
            splice_enabled = gr.Checkbox(value=False, label="启用本次实验性拼接")
            splice_button = gr.Button("搜索并生成缺音候选")
            splice_result = gr.JSON(label="拼接结果与失败原因")
        with gr.Accordion("构建声库：罗马音文件名与音高选择", open=True):
            gr.Markdown("每次点击都会生成独立的草稿目录和 ZIP。保留原音高不做 WORLD 音高重合成；统一音高可填 A3、C4 等音符，或填 auto 由合格候选自动确定。两种模式都会做电平处理。")
            with gr.Row():
                build_pitch_mode = gr.Radio(choices=[("保留原音高", "preserve"), ("统一到目标音高", "flatten")], value="preserve", label="音高处理")
                build_target_note = gr.Textbox(value="auto", label="目标音符（仅统一音高时使用）")
                build_button = gr.Button("生成新草稿声库", variant="primary")
            with gr.Row():
                build_include_splices = gr.Checkbox(value=False, label="包含未审核拼接音（实验草稿；报告会保留待审核标记）")
                build_singer_name = gr.Textbox(value=project_dir.name, label="OpenUtau 显示名称")
            build_result = gr.JSON(label="构建结果、ZIP 路径与缺音数")
        alias = gr.Dropdown(choices=aliases, label="Alias", value=aliases[0] if aliases else None)
        show_all = gr.Checkbox(value=False, label="显示该别名的全部候选（金标准负例可在此查找）")
        candidate = gr.Dropdown(choices=list_candidates(aliases[0]) if aliases else [], label="候选", value=list_candidates(aliases[0])[0] if aliases and list_candidates(aliases[0]) else None)
        with gr.Row():
            plot = gr.Plot(label="波形与五标线")
            playback = gr.Audio(label="原音或派生候选", type="filepath")
        fields = [gr.Number(label=name, precision=0) for name in Landmarks.model_fields]
        status = gr.Dropdown(choices=["unreviewed", "accepted", "rejected"], label="审核状态")
        locks = gr.CheckboxGroup(choices=list(Landmarks.model_fields), label="锁定字段")
        transcript = gr.Textbox(label="来源语句转写")
        details = gr.JSON(label="候选评分、质量原因与来源")
        display_outputs = [plot, playback, *fields, status, locks, transcript, details]
        with gr.Accordion("金标准标注（用于人工验证集）", open=False):
            gr.Markdown("请独立判断正确别名及是否有可持续元音；P/F 填可接受的绝对样本范围。至少需要 60 条，其中正负例各至少 20 条。")
            gold_alias = gr.Textbox(label="人工确认的 alias")
            gold_sustainable = gr.Dropdown(choices=["unreviewed", "yes", "no"], value="unreviewed", label="可持续元音")
            with gr.Row():
                gold_p_min = gr.Number(label="P 最小样本", precision=0)
                gold_p_max = gr.Number(label="P 最大样本", precision=0)
                gold_f_min = gr.Number(label="F 最小样本", precision=0)
                gold_f_max = gr.Number(label="F 最大样本", precision=0)
            gold_split = gr.Dropdown(choices=["development", "validation"], value="validation", label="数据分组")
            gold_reason = gr.Textbox(label="拒绝原因/备注")
            gold_save_button = gr.Button("保存金标准标注")
        gold_outputs = [gold_alias, gold_sustainable, gold_p_min, gold_p_max, gold_f_min, gold_f_max, gold_split, gold_reason]
        field_choice = gr.Dropdown(choices=list(Landmarks.model_fields), value="preutter", label="步进/重置字段")
        with gr.Row():
            save_button = gr.Button("保存标线/审核")
            choose_button = gr.Button("选中候选")
            transcript_button = gr.Button("保存转写并重对齐该句")
            undo_button = gr.Button("撤销上次编辑")
            reset_button = gr.Button("恢复自动标线")
            minus_button = gr.Button("-5 ms")
            plus_button = gr.Button("+5 ms")
            refresh_button = gr.Button("刷新候选")
        message = gr.Textbox(label="结果")
        alias.change(lambda name, all_items: gr.update(choices=list_candidates(name, all_items), value=(list_candidates(name, all_items) or [None])[0]), [alias, show_all], candidate)
        show_all.change(lambda name, all_items: gr.update(choices=list_candidates(name, all_items), value=(list_candidates(name, all_items) or [None])[0]), [alias, show_all], candidate)
        candidate.change(display, candidate, display_outputs)
        candidate.change(display_gold, candidate, gold_outputs)
        save_button.click(save, [candidate, *fields, status, locks], message).then(display, candidate, display_outputs)
        choose_button.click(lambda identifier: (choose_candidate(project_dir, identifier), "已选中候选")[1], candidate, message)
        transcript_button.click(revise_and_realign, [candidate, transcript], [message, alias, candidate])
        undo_button.click(lambda: f"已撤销 {undo_last_edit(project_dir)}", outputs=message).then(display_gold, candidate, gold_outputs)
        reset_button.click(lambda identifier, field: (reset_landmark(project_dir, identifier, field), f"已重置 {field}")[1], [candidate, field_choice], message).then(display, candidate, display_outputs)
        minus_button.click(lambda identifier, field: step_field(identifier, field, -1), [candidate, field_choice], message).then(display, candidate, display_outputs)
        plus_button.click(lambda identifier, field: step_field(identifier, field, 1), [candidate, field_choice], message).then(display, candidate, display_outputs)
        refresh_button.click(lambda: (gr.update(choices=aliases_now(), value=(aliases_now() or [None])[0]), "候选列表已刷新"), outputs=[alias, message])
        gold_save_button.click(save_gold, [candidate, *gold_outputs], message)
        build_pitch_mode.change(lambda mode: gr.update(interactive=mode == "flatten"), build_pitch_mode, build_target_note)
        build_button.click(lambda mode, note, splices, name: build_draft_from_review(project_dir, mode, note, splices, name),
                           [build_pitch_mode, build_target_note, build_include_splices, build_singer_name], build_result, concurrency_limit=1)
        splice_button.click(lambda enabled: generate_splices_from_review(project_dir, enabled), splice_enabled, splice_result, concurrency_limit=1).then(
            lambda: gr.update(choices=aliases_now(), value=(aliases_now() or [None])[0]), outputs=alias)
        if aliases and list_candidates(aliases[0]):
            ui.load(display, candidate, display_outputs)
            ui.load(display_gold, candidate, gold_outputs)
    return ui


def launch_review(project_dir: Path, port: int = 7860) -> None:
    build_review_ui(project_dir).launch(server_name="127.0.0.1", server_port=port, share=False)
