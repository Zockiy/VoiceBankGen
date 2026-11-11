import json

from utau_forge.storage import atomic_bytes


def test_atomic_update_preserves_previous_content(tmp_path):
    path = tmp_path / "data.json"
    atomic_bytes(path, b'{"a":1}')
    atomic_bytes(path, b'{"a":2}')
    assert json.loads(path.read_text()) == {"a": 2}
    history = list((tmp_path / ".history").iterdir())
    assert len(history) == 1
    assert json.loads(history[0].read_text()) == {"a": 1}
