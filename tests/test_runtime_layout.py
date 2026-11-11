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
