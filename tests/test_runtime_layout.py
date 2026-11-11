"""Native wheel data must follow the private interpreter's install schemes."""
from pathlib import Path


def test_wheel_mapping_preserves_native_data_and_omits_build_headers(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "packaging"))
    from prepare_runtime_catalog import wheel_target
    assert wheel_target("fugashi-1.4.0.data/data/lib/site-packages/fugashi/libmecab.dll") == "runtime/tifa-cpu/Lib/site-packages/fugashi/libmecab.dll"
    assert wheel_target("pkg.data/platlib/pkg/extension.pyd") == "runtime/tifa-cpu/Lib/site-packages/pkg/extension.pyd"
    assert wheel_target("pkg.data/purelib/pkg/api.py") == "runtime/tifa-cpu/Lib/site-packages/pkg/api.py"
    assert wheel_target("pkg.data/data/share/license.txt") == "runtime/tifa-cpu/share/license.txt"
    assert wheel_target("pkg.data/scripts/unused.exe") is None
    assert wheel_target("torch/include/unused.h") is None


def test_shared_worker_routes_devices_and_rejects_cuda_in_cpu_package(tmp_path,monkeypatch):
    import json
    import pytest
    from utau_forge import tifa_adapter
    monkeypatch.setattr(tifa_adapter,'application_root',lambda:tmp_path)
    monkeypatch.setattr(tifa_adapter.sys,'frozen',True,raising=False)
    marker=tmp_path/'shared-runtime.json'
    marker.write_text(json.dumps({'cuda':False}))
    expected=[str(tmp_path/'VoiceBankGenWorker.exe'),'--tifa-worker']
    assert tifa_adapter.runtime_command('cpu')==expected
    with pytest.raises(RuntimeError,match='仅支持 CPU'):
        tifa_adapter.runtime_command('cuda')
    marker.write_text(json.dumps({'cuda':True}))
    assert tifa_adapter.runtime_command('cuda')==expected
    assert tifa_adapter.runtime_command('cpu')==expected


def test_cpu_portable_auto_never_selects_cuda(tmp_path,monkeypatch):
    import json
    import pytest
    from utau_forge import devices
    monkeypatch.setattr(devices.sys,'frozen',True,raising=False)
    monkeypatch.setattr(devices.sys,'executable',str(tmp_path/'VoiceBankGenWorker.exe'))
    (tmp_path/'shared-runtime.json').write_text(json.dumps({'cuda':False}))
    assert devices.whisper_device('auto')['device']=='cpu'
    with pytest.raises(RuntimeError,match='仅支持 CPU'):
        devices.whisper_device('cuda')
