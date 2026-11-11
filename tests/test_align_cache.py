from importlib import import_module
from pathlib import Path
from subprocess import CompletedProcess

import numpy as np
import soundfile as sf

from utau_forge.models import Phone, SampleSpan, SourceAsset, Utterance
from utau_forge.project import initialize, load
from utau_forge.storage import read_jsonl, write_jsonl


def test_transcript_edit_realigns_only_its_utterance(tmp_path, monkeypatch):
    module = import_module("utau_forge.align")
    project = tmp_path / "project"
    initialize(project)
    source = SourceAsset(source_id="source", original_path="unused.wav", sha256="source-hash", original_sample_rate=44100, original_channels=1, original_duration_s=2, canonical_path="audio/canonical/source.wav", canonical_samples=88200, speaker_id="speaker_001", language_id="ja")
    sf.write(project / source.canonical_path, np.zeros(88200, dtype="float32"), 44100)
    write_jsonl(project / "sources" / "manifest.jsonl", [source])
    utterances = [Utterance(utterance_id=f"u{i}", source_id="source", speaker_id="speaker_001", span=SampleSpan(start=i * 44100, end=(i + 1) * 44100), vad_span=SampleSpan(start=i * 44100, end=(i + 1) * 44100), corpus_offset_samples=i * 44100, language_id="ja", revised_text="あ") for i in range(2)]
    write_jsonl(project / "utterances.jsonl", utterances)
    executable = tmp_path / "mfa.exe"
    executable.write_bytes(b"fake")
    config = load(project)[1]
    config.alignment.executable = str(executable)
    config.alignment.root_directory = str(tmp_path / "mfa-root")
    corpus_sizes = []

    def fake_run(argv, **kwargs):
        if argv[1] == "version":
            return CompletedProcess(argv, 0, "3.4.2\n", "")
        corpus, output = Path(argv[2]), Path(argv[5])
        files = list(corpus.rglob("*.wav"))
        corpus_sizes.append(len(files))
        for wav in files:
            target = output / wav.parent.name / (wav.stem + ".TextGrid")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text('item [1]:\n name = "phones"\n intervals [1]:\n xmin = 0\n xmax = 0.1\n text = "a"\n', encoding="utf-8")
        return CompletedProcess(argv, 0, "aligned", "")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    assert module.align(project, config)["utterances"] == 2
    old_phone = next(phone for phone in read_jsonl(project / "phones.jsonl", Phone) if phone.utterance_id == "u0")
    utterances[1].revised_text = "い"
    utterances[1].text_revision = 1
    write_jsonl(project / "utterances.jsonl", utterances)
    assert module.align(project, config)["utterances"] == 1
    new_phone = next(phone for phone in read_jsonl(project / "phones.jsonl", Phone) if phone.utterance_id == "u0")
    assert new_phone.phone_id == old_phone.phone_id
    assert module.align(project, config)["run_id"] == "cached"
    assert corpus_sizes == [2, 1]


def test_missing_textgrid_is_reported_without_losing_other_utterances(tmp_path, monkeypatch):
    module = import_module("utau_forge.align")
    project = tmp_path / "project"
    initialize(project)
    source = SourceAsset(source_id="source", original_path="unused.wav", sha256="source-hash", original_sample_rate=44100, original_channels=1, original_duration_s=2, canonical_path="audio/canonical/source.wav", canonical_samples=88200, speaker_id="speaker_001", language_id="ja")
    sf.write(project / source.canonical_path, np.zeros(88200, dtype="float32"), 44100)
    write_jsonl(project / "sources" / "manifest.jsonl", [source])
    utterances = [Utterance(utterance_id=f"u{i}", source_id="source", speaker_id="speaker_001", span=SampleSpan(start=i * 44100, end=(i + 1) * 44100), vad_span=SampleSpan(start=i * 44100, end=(i + 1) * 44100), corpus_offset_samples=i * 44100, language_id="ja", revised_text="あ") for i in range(2)]
    write_jsonl(project / "utterances.jsonl", utterances)
    executable = tmp_path / "mfa.exe"
    executable.write_bytes(b"fake")
    config = load(project)[1]
    config.alignment.executable = str(executable)
    config.alignment.root_directory = str(tmp_path / "mfa-root")
    corpus_sizes = []

    def fake_run(argv, **kwargs):
        if argv[1] == "version":
            return CompletedProcess(argv, 0, "3.4.2\n", "")
        corpus, output = Path(argv[2]), Path(argv[5])
        files = list(corpus.rglob("*.wav"))
        corpus_sizes.append(len(files))
        for wav in files:
            if wav.stem == "u1" and len(corpus_sizes) == 1:
                continue
            target = output / wav.parent.name / (wav.stem + ".TextGrid")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text('item [1]:\n name = "phones"\n intervals [1]:\n xmin = 0\n xmax = 0.1\n text = "a"\n', encoding="utf-8")
        return CompletedProcess(argv, 0, "aligned", "")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    first = module.align(project, config)
    assert first["utterances"] == 1
    assert first["failed"] == 1
    assert {phone.utterance_id for phone in read_jsonl(project / "phones.jsonl", Phone)} == {"u0"}
    assert "ALIGNMENT_FAILED" in read_jsonl(project / "utterances.jsonl", Utterance)[1].quality_flags
    assert module.align(project, config)["run_id"] == "cached"
    revised = read_jsonl(project / "utterances.jsonl", Utterance)
    revised[1].revised_text = "い"
    revised[1].text_revision += 1
    write_jsonl(project / "utterances.jsonl", revised)
    second = module.align(project, config)
    assert second["utterances"] == 1
    assert second["failed"] == 0
    assert {phone.utterance_id for phone in read_jsonl(project / "phones.jsonl", Phone)} == {"u0", "u1"}
    assert "ALIGNMENT_FAILED" not in read_jsonl(project / "utterances.jsonl", Utterance)[1].quality_flags
    assert corpus_sizes == [2, 1]
