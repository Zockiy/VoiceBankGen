"""uforge command-line interface."""

from __future__ import annotations

import json
import os
from importlib import metadata
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import typer
from pydantic import ValidationError

from . import __version__
from .align import align
from .analyze import analyze_project
from .audio import import_audio
from .build import build, finalize_export, select, validate_export
from .config import SelectionConfig
from .models import Phone, SampleSpan, SourceAsset, Utterance
from .model_lock import lock_local_model, verify_model
from .languages import language as get_language
from .project import initialize, load
from .render import load_render_backend, prepare_render_suite
from .review import launch_review
from .repair import repair_project
from .segment import segment_source
from .slices import export_slices
from .storage import read_jsonl, write_json
from .pipeline import run_pipeline
from .evaluation import create_gold_template, evaluate
from .transcribe import import_transcripts, transcribe

app = typer.Typer(help="Traceable local UTAU voicebank builder", no_args_is_help=True)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _emit(value: dict[str, Any], json_output: bool) -> None:
    if json_output:
        typer.echo(json.dumps(value, ensure_ascii=False, sort_keys=True))
    else:
        for key, item in value.items():
            typer.echo(f"{key}: {item}")


def _execute(operation: Callable[[], dict[str, Any]], json_output: bool) -> None:
    try:
        _emit(operation(), json_output)
    except (ValueError, ValidationError, FileExistsError, FileNotFoundError) as exc:
        _emit({"error": str(exc), "exit_code": 2}, json_output)
        raise typer.Exit(2) from exc
    except RuntimeError as exc:
        _emit({"error": str(exc), "exit_code": 3}, json_output)
        raise typer.Exit(3) from exc


def _version(executable: str, args: list[str]) -> str | None:
    path = shutil.which(executable)
    if not path:
        return None
    try:
        result = subprocess.run([path, *args], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15)
        return (result.stdout or result.stderr).splitlines()[0] if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired, IndexError):
        return None


@app.command()
def doctor(project: Path | None = typer.Argument(None), json_output: bool = typer.Option(False, "--json")) -> None:
    try:
        vad_version = metadata.version("silero-vad")
    except metadata.PackageNotFoundError:
        vad_version = None
    checks = {
        "uforge": __version__,
        "python": sys.version.split()[0],
        "ffmpeg": _version("ffmpeg", ["-version"]),
        "ffprobe": _version("ffprobe", ["-version"]),
        "mfa": _version("mfa", ["version"]),
        "silero_vad": vad_version,
    }
    checks["ready_for_import"] = sys.version_info[:2] == (3, 11) and bool(checks["ffmpeg"] and checks["ffprobe"])
    if project is not None:
        _, config = load(project)
        executable = Path(config.alignment.executable)
        if executable.is_file():
            environment = os.environ.copy()
            prefix = executable.resolve().parent.parent
            environment["PATH"] = os.pathsep.join((str(prefix), str(prefix / "Library" / "bin"), str(prefix / "Scripts"), environment.get("PATH", "")))
            check = subprocess.run([str(executable), "version"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30, env=environment)
            checks["mfa"] = check.stdout.strip() if check.returncode == 0 else None
        lock = json.loads((project / "models.lock.json").read_text(encoding="utf-8"))
        checks["models"] = {entry["model_id"]: verify_model(project, entry) for entry in lock.get("models", [])}
        checks["model_status"] = "locked_files_verified" if checks["models"] and all(checks["models"].values()) else "incomplete_or_tampered"
        checks["lock_status"] = lock.get("status")
        checks["phone_map_audit_status"] = getattr(get_language(config.target_languages[0]), "phone_map", {}).get("audit_status", "unverified")
        observed = sorted({phone.raw_model_label for phone in read_jsonl(project / "phones.jsonl", Phone)})
        checks["observed_phone_labels"] = observed
        checks["unknown_phone_labels"] = sorted({phone.raw_model_label for phone in read_jsonl(project / "phones.jsonl", Phone) if phone.unknown})
    else:
        checks["model_status"] = "project_not_specified"
    checks["ready_for_alignment"] = bool(checks["mfa"]) and (project is None or all(checks.get("models", {}).get(model_id) for model_id in ("japanese_mfa_acoustic", "japanese_mfa_dictionary")))
    if project is not None:
        write_json(project / "runs" / "doctor.json", checks)
    _emit(checks, json_output)
    if not checks["ready_for_import"]:
        raise typer.Exit(3)


@app.command("init")
def init_cmd(project: Path, language: str = typer.Option("ja", "--language"), profile: str = typer.Option("ja-cv", "--profile"), speaker: str = typer.Option("speaker_001", "--speaker"), json_output: bool = typer.Option(False, "--json")) -> None:
    _execute(lambda: {"project_id": initialize(project, language, profile, speaker).project_id, "path": str(project.resolve())}, json_output)


@app.command("import")
def import_cmd(project: Path, input_path: Path, speaker: str = typer.Option("speaker_001", "--speaker"), language: str = typer.Option("ja", "--language"), session: str | None = typer.Option(None, "--session"), json_output: bool = typer.Option(False, "--json")) -> None:
    def run() -> dict[str, Any]:
        metadata, _ = load(project)
        if language not in metadata.source_languages:
            raise ValueError(f"source language {language} is not registered in this project")
        if speaker != metadata.speaker_id:
            raise ValueError("speaker does not match project speaker_id")
        asset = import_audio(project, input_path, speaker, language, session)
        return {"source_id": asset.source_id, "canonical_samples": asset.canonical_samples, "sha256": asset.sha256}
    _execute(run, json_output)


@app.command("segment")
def segment_cmd(project: Path, start_s: float | None = typer.Option(None, "--start-s"), end_s: float | None = typer.Option(None, "--end-s"), json_output: bool = typer.Option(False, "--json")) -> None:
    def run() -> dict[str, Any]:
        _, config = load(project)
        assets = read_jsonl(project / "sources" / "manifest.jsonl", SourceAsset)
        if not assets:
            raise ValueError("import an audio source first")
        if (start_s is None) != (end_s is None):
            raise ValueError("--start-s and --end-s must be supplied together")
        count = 0
        for asset in assets:
            manual = SampleSpan(start=round(start_s * 44100), end=round(end_s * 44100)) if start_s is not None and end_s is not None else None
            count += len(segment_source(project, asset, config, manual))
        return {"utterances": count}
    _execute(run, json_output)


@app.command("transcribe")
def transcribe_cmd(project: Path, transcripts: Path | None = typer.Option(None, "--transcripts"), json_output: bool = typer.Option(False, "--json")) -> None:
    def run() -> dict[str, Any]:
        _, config = load(project)
        if transcripts is not None:
            return {"revised_utterances": import_transcripts(project, transcripts)}
        return {"transcribed_utterances": transcribe(project, config)}
    _execute(run, json_output)


@app.command("align")
def align_cmd(project: Path, backend: str = typer.Option("mfa","--backend"), device: str = typer.Option("auto","--device"), json_output: bool = typer.Option(False, "--json")) -> None:
    def run():
        if backend=="tifa":
            from .tifa_adapter import align_tifa
            return align_tifa(project,device)
        if backend!="mfa": raise ValueError("backend must be mfa or tifa")
        if any(u.language_id!="ja" for u in read_jsonl(project/"utterances.jsonl",Utterance)):
            raise ValueError("普通话请使用 --backend tifa")
        return align(project, load(project)[1])
    _execute(run, json_output)


@app.command("analyze")
def analyze_cmd(project: Path, json_output: bool = typer.Option(False, "--json")) -> None:
    _execute(lambda: analyze_project(project, load(project)[1]), json_output)


@app.command("select")
def select_cmd(project: Path, top_k: int | None = typer.Option(None, "--top-k"), json_output: bool = typer.Option(False, "--json")) -> None:
    def run() -> dict[str, Any]:
        config = load(project)[1]
        if top_k is not None:
            config = config.model_copy(update={"selection": SelectionConfig(top_k=top_k)})
        return select(project, config)
    _execute(run, json_output)


@app.command("build")
def build_cmd(project: Path, pitch_mode: str = typer.Option("preserve", "--pitch-mode"), target_note: str = typer.Option("auto", "--target-note"), release_mode: str = typer.Option("draft", "--release-mode"), include_unreviewed_splices: bool = typer.Option(False, "--include-unreviewed-splices"), singer_name: str | None = typer.Option(None, "--singer-name"), json_output: bool = typer.Option(False, "--json")) -> None:
    def run() -> dict[str, Any]:
        result = build(project, load(project)[1], pitch_mode, target_note, release_mode,
                       include_unreviewed_splices=include_unreviewed_splices, singer_name=singer_name)
        _emit(result, json_output)
        if result["validation_errors"] or (release_mode == "strict" and not result["strict_ready"]):
            raise typer.Exit(4)
        return {}
    try:
        run()
    except typer.Exit:
        raise
    except (ValueError, ValidationError, FileNotFoundError) as exc:
        _emit({"error": str(exc), "exit_code": 2}, json_output)
        raise typer.Exit(2) from exc
    except RuntimeError as exc:
        _emit({"error": str(exc), "exit_code": 3}, json_output)
        raise typer.Exit(3) from exc


@app.command("validate")
def validate_cmd(export_dir: Path, json_output: bool = typer.Option(False, "--json")) -> None:
    def run() -> dict[str, Any]:
        result = validate_export(export_dir)
        _emit(result, json_output)
        if result["errors"]:
            raise typer.Exit(4)
        return {}
    try:
        run()
    except typer.Exit:
        raise
    except (ValueError, FileNotFoundError, UnicodeError) as exc:
        _emit({"error": str(exc), "exit_code": 2}, json_output)
        raise typer.Exit(2) from exc


@app.command("export-slices")
def export_slices_cmd(project: Path, all_candidates: bool = typer.Option(False, "--all-candidates"), top_k: int = typer.Option(3, "--top-k"), padding_ms: int = typer.Option(0, "--padding-ms"), json_output: bool = typer.Option(False, "--json")) -> None:
    _execute(lambda: export_slices(project, all_candidates, top_k, padding_ms), json_output)


@app.command("lock-model")
def lock_model_cmd(project: Path, source_dir: Path, provider: str = typer.Option(..., "--provider"), model_id: str = typer.Option(..., "--model-id"), revision: str = typer.Option(..., "--revision"), license_id: str = typer.Option(..., "--license"), source_url: str = typer.Option(..., "--source-url"), sample_rate: int = typer.Option(..., "--sample-rate"), device: str = typer.Option("cpu", "--device"), precision: str = typer.Option("int8", "--precision"), json_output: bool = typer.Option(False, "--json")) -> None:
    _execute(lambda: lock_local_model(project, source_dir, provider=provider, model_id=model_id, revision=revision, license_id=license_id, source_url=source_url, sample_rate=sample_rate, device=device, precision=precision), json_output)


@app.command("render-test")
def render_test_cmd(export_dir: Path, backend: Path | None = typer.Option(None, "--backend"), engine_version: str | None = typer.Option(None, "--engine-version"), renderer: str = typer.Option("WORLDLINE-R", "--renderer"), json_output: bool = typer.Option(False, "--json")) -> None:
    def run() -> dict[str, Any]:
        if backend is not None:
            settings = load_render_backend(backend)
            return prepare_render_suite(export_dir, engine_version=settings.engine_version, renderer=settings.renderer, base_midi=settings.base_midi)
        if engine_version is None:
            raise ValueError("provide --backend FILE or --engine-version VERSION")
        return prepare_render_suite(export_dir, engine_version=engine_version, renderer=renderer)
    _execute(run, json_output)


@app.command("finalize")
def finalize_cmd(export_dir: Path, json_output: bool = typer.Option(False, "--json")) -> None:
    result = finalize_export(export_dir)
    _emit(result, json_output)
    if not result["strict_ready"]:
        raise typer.Exit(4)


@app.command("run")
def run_cmd(project: Path, until: str = typer.Option("build", "--until"), json_output: bool = typer.Option(False, "--json")) -> None:
    _execute(lambda: run_pipeline(project, until), json_output)


@app.command("gold-template")
def gold_template_cmd(project: Path, output: Path | None = typer.Option(None, "--output"), json_output: bool = typer.Option(False, "--json")) -> None:
    _execute(lambda: create_gold_template(project, output), json_output)


@app.command("evaluate")
def evaluate_cmd(project: Path, annotations: Path, json_output: bool = typer.Option(False, "--json")) -> None:
    try:
        result = evaluate(project, annotations)
        _emit(result, json_output)
        if result["status"] != "candidate_metrics_passed":
            raise typer.Exit(4)
    except typer.Exit:
        raise
    except (ValueError, ValidationError, FileNotFoundError) as exc:
        _emit({"error": str(exc), "exit_code": 2}, json_output)
        raise typer.Exit(2) from exc


@app.command("review")
def review_cmd(project: Path, port: int = typer.Option(7860, "--port")) -> None:
    launch_review(project, port)


@app.command("repair")
def repair_cmd(project: Path, mode: str = typer.Option(..., "--mode"), experimental: bool = typer.Option(False, "--experimental"), alias: str | None = typer.Option(None, "--alias"), json_output: bool = typer.Option(False, "--json")) -> None:
    _execute(lambda: repair_project(project, load(project)[1], mode=mode, experimental=experimental, alias_filter=alias), json_output)


if __name__ == "__main__":
    app()
