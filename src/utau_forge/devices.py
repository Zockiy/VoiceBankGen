"""Independent runtime detection for Whisper and PyTorch."""
from __future__ import annotations
import os
import json
import sys
from pathlib import Path

_DLL_HANDLES = []

def cpu_only_package(root: Path) -> bool:
    marker=root/"shared-runtime.json"
    return marker.is_file() and not json.loads(marker.read_text(encoding="utf-8")).get("cuda",False)

def prepare_dlls(root: Path):
    for path in (root / "runtime" / "cuda", root/"_internal/torch/lib", root/"runtime/tifa-gpu/_internal/torch/lib", root/".tifa-env/Lib/site-packages/torch/lib", root / "_internal" / "nvidia" / "cublas" / "bin", root / "_internal" / "nvidia" / "cudnn" / "bin"):
        if path.is_dir():
            os.environ["PATH"] = str(path) + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                _DLL_HANDLES.append(os.add_dll_directory(str(path)))

def whisper_device(requested: str) -> dict:
    if getattr(sys,"frozen",False) and cpu_only_package(Path(sys.executable).parent):
        if requested=="cuda":
            raise RuntimeError("此便携包仅支持 CPU；CUDA 需要 GPU 版")
        requested="cpu"
    if requested == "cpu":
        return {"device":"cpu", "precision":"int8", "reason":"用户选择 CPU"}
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() < 1:
            raise RuntimeError("CTranslate2 没有可用 CUDA 设备")
        types = ctranslate2.get_supported_compute_types("cuda")
        return {"device":"cuda", "precision":"float16" if "float16" in types else "float32", "reason":"CTranslate2 CUDA"}
    except Exception as error:
        if requested == "cuda":
            raise RuntimeError(f"NVIDIA Whisper 不可用：{error}；可切换 CPU 后重试") from error
        return {"device":"cpu", "precision":"int8", "reason":str(error)}

def torch_device(requested: str) -> dict:
    import torch
    if requested != "cpu" and torch.cuda.is_available():
        return {"device":"cuda", "precision":"32-true", "reason":torch.cuda.get_device_name(0)}
    if requested == "cuda":
        raise RuntimeError("TIFA 的 PyTorch CUDA runtime 不可用；请安装 GPU 组件或切换 CPU")
    return {"device":"cpu", "precision":"32-true", "reason":"CPU"}
