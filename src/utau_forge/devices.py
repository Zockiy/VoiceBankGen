"""DirectML fork: CPU Whisper and a shared CPU/DirectML TIFA runtime."""
from __future__ import annotations
from pathlib import Path

def prepare_dlls(root: Path):
    """Compatibility hook: DirectML DLLs load only inside the TIFA process.

    Do not add the older TIFA torch DLLs to the main process's search path;
    Whisper/VAD may use a different CPU torch version there.
    """

def whisper_device(requested: str) -> dict:
    normalize_device(requested)
    return {"device": "cpu", "precision": "int8",
            "reason": "faster-whisper 不支持 DirectML；转写使用 CPU，TIFA 单独选择设备"}


def normalize_device(requested: str) -> str:
    # Migrate device values carried by old projects/jobs, without loading CUDA.
    value = "directml" if requested == "cuda" else requested
    if value not in {"auto", "cpu", "directml"}:
        raise ValueError("计算设备须为 auto、cpu 或 directml")
    return value

def torch_device(requested: str) -> dict:
    requested = normalize_device(requested)
    if requested != "cpu":
        try:
            import torch_directml
            if not torch_directml.is_available():
                raise RuntimeError("没有可用 DirectML 设备")
            device = torch_directml.device()
            return {"device": "directml", "precision": "32-true",
                    "reason": torch_directml.device_name(device.index)}
        except Exception as error:
            if requested == "directml":
                raise RuntimeError(f"DirectML 不可用：{error}；请准备 DirectML 环境或选择 CPU") from error
            return {"device": "cpu", "precision": "32-true", "reason": str(error)}
    return {"device": "cpu", "precision": "32-true", "reason": "用户选择 CPU"}
