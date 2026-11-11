import json

import numpy as np
import soundfile as sf

from utau_forge.build import _bank_digest, validate_render_report


def test_render_gate_requires_real_audio_matching_the_bank(tmp_path):
    root = tmp_path / "export"
    bank = root / "voicebank"
    reports = root / "reports"
    bank.mkdir(parents=True)
    reports.mkdir()
    (bank / "oto.ini").write_text("", encoding="utf-8")
    bank_hash = _bank_digest(bank)
    (reports / "build_manifest.json").write_text(json.dumps({"bank_sha256": bank_hash}), encoding="utf-8")
    fake = {"status": "passed", "import_status": "passed", "human_listening": "passed", "engine": "OpenUtau", "engine_version": "0.1.570.1", "renderer": "WORLDLINE-R", "bank_sha256": bank_hash, "tested_aliases": ["あ"], "rendered_wav": ["render_test/out.wav"]}
    (reports / "render_report.json").write_text(json.dumps(fake), encoding="utf-8")
    assert not validate_render_report(root, {"あ"})["passed"]
    (root / "render_test").mkdir()
    sf.write(root / "render_test" / "out.wav", np.ones(100, dtype="float32") * 0.1, 44100)
    assert validate_render_report(root, {"あ"})["passed"]
    (bank / "oto.ini").write_text("changed", encoding="utf-8")
    assert validate_render_report(root, {"あ"})["reason"] == "BANK_CHANGED_SINCE_RENDER"
