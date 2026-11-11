import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from utau_forge.config import Config, read_config
from utau_forge.desktop_engine import prepare_engine
from utau_forge.desktop_store import ProjectStore
from utau_forge.model_lock import lock_local_model, verify_model
from utau_forge.model_settings import (REQUIRED_FILES, apply_engine_model_paths, detect_model_paths,
                                       load_model_paths, save_model_paths, tifa_resources)
from utau_forge.models import SampleSpan, SourceAsset, Utterance
from utau_forge.project import initialize
from utau_forge.storage import write_json, write_jsonl
from utau_forge.transcribe import transcribe


def test_frozen_tifa_uses_private_python_and_preserves_bundled_worker(tmp_path, monkeypatch):
    from utau_forge import tifa_adapter
    monkeypatch.setattr(tifa_adapter, "application_root", lambda: tmp_path)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    python = tmp_path / "runtime/tifa-cpu/python.exe"
    bridge = tmp_path / "_internal/utau_forge/data/runtime/tifa_entry.py"
    python.parent.mkdir(parents=True)
    bridge.parent.mkdir(parents=True)
    python.write_bytes(b"private interpreter")
    bridge.write_text("# application entry point", encoding="utf-8")
    assert tifa_adapter.runtime_command("cpu") == [str(python), str(bridge)]
    worker = python.with_name("TIFAWorker.exe")
    worker.write_bytes(b"bundled worker")
    assert tifa_adapter.runtime_command("cpu") == [str(worker)]
    with pytest.raises(RuntimeError, match="缺少 TIFA cuda"):
        tifa_adapter.runtime_command("cuda")


def model_directory(base, key):
    base.mkdir(parents=True)
    for name in REQUIRED_FILES[key]:
        path = base / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test model")
    return base


def test_settings_remember_validate_and_reset(tmp_path, monkeypatch):
    monkeypatch.setenv("UFORGE_DATA_DIR", str(tmp_path / "userdata"))
    assert not any(load_model_paths().values())
    model = model_directory(tmp_path / "Whisper 模型", "asr")
    saved = save_model_paths({"asr": str(model)})
    assert load_model_paths() == saved
    with pytest.raises(ValueError, match="缺少"):
        save_model_paths({"asr": str(tmp_path / "missing")})
    assert load_model_paths() == saved
    save_model_paths({})
    assert not any(load_model_paths().values())


def test_direct_external_models_and_tamper_detection(tmp_path):
    model = model_directory(tmp_path / "Whisper", "asr")
    acoustic, dictionary = tmp_path / "ja.zip", tmp_path / "ja.dict"
    acoustic.write_bytes(b"acoustic")
    dictionary.write_bytes(b"dictionary")
    config, lock = Config(), {"models": []}
    apply_engine_model_paths(config, lock, {"asr": str(model), "mfa_acoustic": str(acoustic), "mfa_dictionary": str(dictionary)})
    record = next(r for r in lock["models"] if r["model_id"] == config.asr.model)
    assert verify_model(tmp_path, record)
    assert record["local_absolute_path"] == str(model)
    assert config.alignment.acoustic == str(acoustic)
    assert config.alignment.dictionary == str(dictionary)
    (model / "model.bin").write_bytes(b"changed")
    assert not verify_model(tmp_path, record)


def test_tifa_manual_paths_have_content_signature(tmp_path):
    model = model_directory(tmp_path / "TIFA", "tifa")
    dictionary = model_directory(tmp_path / "UniDic", "unidic")
    paths = {"tifa": str(model), "unidic": str(dictionary)}
    checkpoint, selected_dictionary, first = tifa_resources(tmp_path, paths)
    assert checkpoint == model / "model.pt" and selected_dictionary == dictionary
    (dictionary / "sys.dic").write_bytes(b"changed dictionary")
    assert tifa_resources(tmp_path, paths)[2]["files"] != first["files"]


def test_cancelled_first_model_load_retries_locally(tmp_path, monkeypatch):
    monkeypatch.setenv("UFORGE_DATA_DIR", str(tmp_path / "userdata"))
    app = tmp_path / "app"
    bundle = app / "resources/engine"
    initialize(bundle)
    model = model_directory(tmp_path / "Whisper", "asr")
    lock_local_model(bundle, model, provider="HuggingFace", model_id="Systran/faster-whisper-small", revision="test", license_id="mit", source_url="local", sample_rate=16000, device="cpu", precision="int8")
    monkeypatch.setattr("utau_forge.desktop_engine.application_root", lambda: app)
    store = ProjectStore.create(tmp_path / "project", "test")
    assert read_config(store.directory / "config.yaml").asr.model == "large-v3"
    first = prepare_engine(store)
    # A cancelled task may leave the lock but not the temporary ASR selection.
    restarted = prepare_engine(ProjectStore(store.directory))
    config = read_config(restarted / "config.yaml")
    assert config.asr.model == "Systran/faster-whisper-small"
    assert config.runtime.offline
    span = SampleSpan(start=0, end=4410)
    source = SourceAsset(source_id="s", original_path="sources/a.wav", sha256="test", original_sample_rate=44100, original_channels=1, original_duration_s=.1, canonical_path="audio/canonical/a.wav", canonical_samples=4410, speaker_id="speaker", language_id="ja")
    utterance = Utterance(source_id="s", speaker_id="speaker", span=span, vad_span=span, corpus_offset_samples=0, language_id="ja")
    (restarted / source.canonical_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(restarted / source.canonical_path, np.zeros(4410), 44100)
    write_jsonl(restarted / "sources/manifest.jsonl", [source])
    write_jsonl(restarted / "utterances.jsonl", [utterance])
    attempts = []
    def load(path, **kwargs):
        assert kwargs["local_files_only"] and (restarted / path).is_dir()
        attempts.append(path)
        if len(attempts) == 1:
            raise RuntimeError("simulated cancellation")
        return SimpleNamespace(transcribe=lambda *args, **kw: ([SimpleNamespace(text="テスト", no_speech_prob=0)], None))
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=load))
    with pytest.raises(RuntimeError, match="simulated cancellation"):
        transcribe(first, config)
    assert transcribe(restarted, config) == 1
    assert len(attempts) == 2 and attempts[0] == attempts[1]
    detected = detect_model_paths(app, store.directory)
    assert detected["asr"].startswith(str(bundle))
    assert not (store.directory / "models").exists()
    assert not (restarted / "models").exists()
    locked = json.loads((restarted / "models.lock.json").read_text(encoding="utf-8"))
    assert verify_model(restarted, locked["models"][0])


def test_manual_whisper_without_bundled_resource_pack(tmp_path, monkeypatch):
    monkeypatch.setenv("UFORGE_DATA_DIR", str(tmp_path / "userdata"))
    monkeypatch.setattr("utau_forge.desktop_engine.application_root", lambda: tmp_path / "no-bundle")
    model = model_directory(tmp_path / "Whisper", "asr")
    store = ProjectStore.create(tmp_path / "project", "test")
    root = prepare_engine(store, {"asr": str(model)})
    config = read_config(root / "config.yaml")
    lock = json.loads((root / "models.lock.json").read_text(encoding="utf-8"))
    assert config.asr.model == "VoiceBankGen/local-asr"
    assert verify_model(root, lock["models"][0])


def test_legacy_project_models_are_not_used_or_replicated(tmp_path, monkeypatch):
    monkeypatch.setenv("UFORGE_DATA_DIR", str(tmp_path / "userdata"))
    app = tmp_path / "app"
    bundle = app / "resources/engine"
    initialize(bundle)
    model = model_directory(tmp_path / "Whisper", "asr")
    lock_local_model(bundle, model, provider="HuggingFace", model_id="Systran/faster-whisper-small", revision="test", license_id="mit", source_url="local", sample_rate=16000, device="cpu", precision="int8")
    store = ProjectStore.create(tmp_path / "project", "legacy")
    lock_local_model(store.directory, model, provider="HuggingFace", model_id="Systran/faster-whisper-small", revision="old", license_id="mit", source_url="local", sample_rate=16000, device="cpu", precision="int8")
    old_lock = json.loads((store.directory / "models.lock.json").read_text(encoding="utf-8"))
    stale = store.directory / old_lock["models"][0]["local_relative_path"]
    (stale / "model.bin").write_bytes(b"obsolete local copy")
    monkeypatch.setattr("utau_forge.desktop_engine.application_root", lambda: app)
    engine = prepare_engine(store, {})
    assert detect_model_paths(app, store.directory)["asr"].startswith(str(bundle))
    assert not (engine / "models").exists()
    lock = json.loads((engine / "models.lock.json").read_text(encoding="utf-8"))
    assert lock["models"][0]["local_absolute_path"].startswith(str(bundle))
    assert verify_model(engine, lock["models"][0])
    # Preparing an engine does not destructively remove old user files.
    assert (stale / "model.bin").read_bytes() == b"obsolete local copy"


def test_source_uses_portable_store_and_relocation_rebinds_references(tmp_path, monkeypatch):
    import shutil
    monkeypatch.setenv("UFORGE_DATA_DIR", str(tmp_path / "userdata"))
    app = tmp_path / "app"
    bundle = app / "dist/VoiceBankGen/resources/engine"
    initialize(bundle)
    model = model_directory(tmp_path / "Whisper", "asr")
    lock_local_model(bundle, model, provider="HuggingFace", model_id="Systran/faster-whisper-small", revision="test", license_id="mit", source_url="local", sample_rate=16000, device="cpu", precision="int8")
    store = ProjectStore.create(tmp_path / "project", "relocated")
    monkeypatch.setattr("utau_forge.desktop_engine.application_root", lambda: app)
    prepare_engine(store, {})
    assert detect_model_paths(app)["asr"].startswith(str(bundle))
    moved = tmp_path / "moved-package"
    shutil.copytree(app / "dist/VoiceBankGen", moved)
    monkeypatch.setattr("utau_forge.desktop_engine.application_root", lambda: moved)
    engine = prepare_engine(store, {})
    lock = json.loads((engine / "models.lock.json").read_text(encoding="utf-8"))
    assert lock["models"][0]["local_absolute_path"].startswith(str(moved))
    assert verify_model(engine, lock["models"][0])
    assert not (store.directory / "models").exists() and not (engine / "models").exists()
