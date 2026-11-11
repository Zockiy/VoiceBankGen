import json

from utau_forge.model_lock import lock_local_model, verify_model
from utau_forge.project import initialize


def test_lock_single_file_and_detect_tamper(tmp_path):
    project = tmp_path / "project"
    initialize(project)
    source = tmp_path / "model.zip"
    source.write_bytes(b"model-v1")
    result = lock_local_model(project, source, provider="MFA", model_id="ja-acoustic", revision="3.0.0", license_id="CC-BY-4.0", source_url="https://example.org/model", sample_rate=16000, device="cpu", precision="float32")
    lock = json.loads((project / "models.lock.json").read_text(encoding="utf-8"))
    record = lock["models"][0]
    assert result["files"] == 1
    assert verify_model(project, record)
    (project / record["local_relative_path"] / source.name).write_bytes(b"tampered")
    assert not verify_model(project, record)
