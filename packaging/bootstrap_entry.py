"""Single EXE installer/launcher. Large payload stays compressed inside the EXE."""
from __future__ import annotations
import argparse
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

from utau_forge.resource_download import (CHUNK, ComponentInstaller, DownloadCancelled, checkpoint,
                                         component_ready, destination, digest, load_preferences,
                                         preserve, save_json, save_preferences, validate_catalog)


def arguments():
    parser = argparse.ArgumentParser(description="VoiceBankGen 单文件安装器")
    parser.add_argument("--install-dir", type=Path)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--payload-dir", type=Path)
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--no-models", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--launch-smoke", action="store_true")
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def install_payload(root, catalog, payload_directory, cancel, emit):
    """Copy embedded runtime archives into persistent repair/download cache."""
    source_bundle = None if payload_directory else zipfile.ZipFile(sys.executable)
    try:
        for component in catalog["components"]:
            for artifact in component["artifacts"]:
                if artifact.get("url"):
                    continue
                path = destination(root / ".component-cache", artifact["cache"])
                if path.is_file() and path.stat().st_size == artifact["size"] and digest(path, cancel) == artifact["sha256"]:
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                preserve(path)
                pending = path.with_name(path.name + ".extracting")
                preserve(pending)
                source = ((payload_directory / artifact["cache"]).open("rb") if payload_directory
                          else source_bundle.open("payload/" + artifact["cache"]))
                current = 0
                with source, pending.open("wb") as output:
                    while chunk := source.read(CHUNK):
                        checkpoint(cancel)
                        output.write(chunk)
                        current += len(chunk)
                        emit(name=component["name"], stage="准备内置环境", current=current, total=artifact["size"])
                if current != artifact["size"] or digest(pending, cancel) != artifact["sha256"]:
                    raise ValueError("内置环境校验失败，请重新取得完整安装器")
                pending.replace(path)
    finally:
        if source_bundle:
            source_bundle.close()


def create_shortcut(root):
    helper = Path(getattr(sys, "_MEIPASS", Path(__file__).parent)) / "create_shortcut.ps1"
    # Use the actual Windows known Desktop, including OneDrive redirection.
    command = ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(helper),
               "-Program", str(root / "VoiceBankGen.exe"), "-Directory", str(root)]
    output = subprocess.run(command, capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
    if output.returncode:
        raise OSError("桌面快捷方式未创建，可从安装目录双击 VoiceBankGen.exe")


def main():
    import tkinter as tk
    from tkinter import filedialog, ttk
    args = arguments()
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))
    catalog = validate_catalog(json.loads((args.catalog or base / "install-catalog.json").read_text(encoding="utf-8")))
    preferences = load_preferences()
    root = tk.Tk()
    root.title("VoiceBankGen 安装程序")
    root.geometry("900x820")
    root.option_add("*Font", "{Microsoft YaHei} 10")
    frame = ttk.Frame(root, padding=20)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text="VoiceBankGen", font=("Microsoft YaHei", 20, "bold")).pack(anchor="w")
    offline = catalog.get("edition") == "full"
    ttk.Label(frame, text=("完整版：已内置主程序、CPU 运行环境和全部必要模型，可离线安装并使用。" if offline else
              "联网版：仅内置主程序，运行环境和模型首次使用时从网络下载，无需手动安装 Python / Conda。"),
              wraplength=840).pack(anchor="w", pady=12)
    location = tk.StringVar(value=str(args.install_dir or Path(os.environ.get("LOCALAPPDATA", Path.home())) / "VoiceBankGen/app"))
    source = tk.StringVar(value="镜像站" if preferences["source"] == "mirror" else "官方上游")
    github = tk.StringVar(value=preferences["github_mirror"])
    huggingface = tk.StringVar(value=preferences["hf_mirror"])
    other = tk.StringVar(value=preferences["other_mirror"])
    proxy = tk.StringVar(value=preferences["proxy"])
    models = tk.BooleanVar(value=not args.no_models)
    shortcut = tk.BooleanVar(value=not args.auto)
    fields = ttk.Frame(frame)
    fields.pack(fill="x")
    for row, (text, variable) in enumerate((("安装位置", location), ("下载源", source), ("GitHub 镜像", github), ("Hugging Face 镜像", huggingface),
                                           ("其他资源镜像", other), ("HTTP/HTTPS 代理", proxy))):
        ttk.Label(fields, text=text).grid(row=row, column=0, sticky="w", pady=5)
        widget = (ttk.Combobox(fields, textvariable=variable, values=["官方上游", "镜像站"], state="readonly")
                  if variable is source else ttk.Entry(fields, textvariable=variable))
        widget.grid(row=row, column=1, sticky="ew", padx=12)
    fields.columnconfigure(1, weight=1)
    def browse():
        selected = filedialog.askdirectory(initialdir=location.get())
        if selected:
            location.set(selected)
    ttk.Button(fields, text="浏览…", command=browse).grid(row=0, column=2)
    network_bytes = sum(a["size"] for c in catalog["components"] for a in c["artifacts"] if a.get("url"))
    ttk.Label(frame, text="GitHub / 其他镜像：URL 前缀或 {url} 模板；Hugging Face：根地址。其他镜像留空则使用官方。\n"
              "代理示例：http://127.0.0.1:7890（填 HTTP 端口）；留空使用系统代理。\n"
              f"完整首次下载约 {network_bytes / 1024**3:.2f} GB；建议预留至少 8 GB，实际按所选组件计算。", wraplength=840).pack(anchor="w", pady=10)
    option_widgets = []
    model_option = ttk.Checkbutton(frame, text="安装必要模型（推荐；完整版从 EXE 提取，联网版从网络下载）", variable=models)
    model_option.pack(anchor="w")
    option_widgets.append(model_option)
    ttk.Label(frame, text="TIFA 使用 CPU / DirectML 共享环境（以所选发行包为准）。DirectML 只加速 TIFA，语音识别使用 CPU。\n"
              "不安装 CUDA、cuDNN 或显卡驱动。自动模式在 DirectML 不可用时回退 CPU。",
              wraplength=840).pack(anchor="w", pady=(2, 6))
    shortcut_option = ttk.Checkbutton(frame, text="创建桌面快捷方式", variable=shortcut)
    shortcut_option.pack(anchor="w")
    option_widgets.append(shortcut_option)
    progress = ttk.Progressbar(frame, maximum=1000)
    progress.pack(fill="x", pady=15)
    status = tk.StringVar(value="准备后进入主界面。完整版安装无需联网；下载源和代理设置会记忆。")
    ttk.Label(frame, textvariable=status, wraplength=760).pack(anchor="w")
    buttons = ttk.Frame(frame)
    buttons.pack(fill="x", pady=15)
    events = queue.Queue()
    cancel = threading.Event()
    state = {"thread": None, "root": None, "name": "", "last_time": 0, "stage": ""}
    started = time.monotonic()

    def emit(**event):
        now = time.monotonic()
        stage = event.get("stage", "")
        if now - state["last_time"] > .08 or stage != state["stage"] or event.get("current") == event.get("total"):
            events.put(("progress", event))
            state["last_time"], state["stage"] = now, stage

    def task(install_root, selected_preferences, include_models):
        try:
            selected = [c for c in catalog["components"] if
                        (c.get("required") and (include_models or c["id"] in {"app-core", "tifa-cpu", "ffmpeg", "vad-model"}))]
            selected_catalog = {**catalog, "components": selected}
            unpacked = sum(f["size"] for c in selected for a in c["artifacts"] for f in a["files"])
            archives = sum(a["size"] for c in selected for a in c["artifacts"])
            minimum = max(4 * 1024**3, unpacked + archives + 512 * 1024**2)
            if shutil.disk_usage(install_root if install_root.exists() else install_root.anchor).free < minimum:
                raise OSError(f"所选组件需要约 {minimum / 1024**3:.1f} GiB 空闲空间（包含安装缓存）。")
            installer = ComponentInstaller(install_root, catalog, selected_preferences, cancel, emit)
            identifiers = [c["id"] for c in selected]
            installer.run(identifiers, prepare=lambda: install_payload(install_root, selected_catalog, args.payload_dir, cancel, emit))
            save_json(install_root / "components.json", catalog)
            events.put(("done", {"install_dir": str(install_root), "components": identifiers,
                                 "elapsed_s": round(time.monotonic() - started, 3), "passed": True}))
        except Exception as error:
            events.put(("error", {"error": str(error), "cancelled": isinstance(error, DownloadCancelled), "passed": False}))

    def start():
        if state["thread"]:
            return
        try:
            install_root = Path(location.get().strip()).expanduser()
            if not install_root.is_absolute():
                raise ValueError("请选择绝对安装路径")
            install_root = install_root.resolve()
            selected_preferences = save_preferences({"source": "mirror" if source.get() == "镜像站" else "official",
                                                    "github_mirror": github.get(), "hf_mirror": huggingface.get(),
                                                    "other_mirror": other.get(), "proxy": proxy.get()})
        except (OSError, ValueError) as error:
            status.set(str(error))
            return
        state["root"] = install_root
        cancel.clear()
        start_button.configure(state="disabled")
        cancel_button.configure(state="normal")
        for child in fields.winfo_children():
            child.configure(state="disabled")
        for widget in option_widgets:
            widget.configure(state="disabled")
        state["thread"] = threading.Thread(target=task, args=(install_root, selected_preferences, models.get()), daemon=True)
        state["thread"].start()

    def stop():
        if state["thread"]:
            cancel.set()
            status.set("正在取消，安装写入阶段会先安全完成；再次点击可继续下载。")
        else:
            root.destroy()

    start_button = ttk.Button(buttons, text="准备并启动", command=start)
    start_button.pack(side="left")
    cancel_button = ttk.Button(buttons, text="取消", command=stop)
    cancel_button.pack(side="right")
    root.protocol("WM_DELETE_WINDOW", stop)

    def poll():
        while not events.empty():
            kind, event = events.get_nowait()
            if kind == "progress":
                state["name"] = event.get("name", state["name"])
                status.set(f"{state['name']} · {event.get('stage', '')}")
                progress["value"] = 1000 * event.get("current", 0) / max(1, event.get("total", 1))
                if event.get("stage") in {"下载", "准备内置环境"}:
                    status.set(f"{state['name']} · {event['stage']} · "
                               f"{event.get('current', 0) / 1024**2:.1f} / {event.get('total', 1) / 1024**2:.1f} MB")
                continue
            state["thread"] = None
            if args.report:
                save_json(args.report, event)
            if kind == "error":
                status.set("未完成：" + event["error"])
                start_button.configure(state="normal")
                for child in fields.winfo_children():
                    child.configure(state="normal")
                for widget in option_widgets:
                    widget.configure(state="normal")
                if args.auto:
                    root.destroy()
                    return
                continue
            if shortcut.get():
                try:
                    create_shortcut(state["root"])
                except OSError as error:
                    status.set(str(error))
            if not args.prepare_only:
                command = [str(state["root"] / "VoiceBankGen.exe")]
                if args.launch_smoke:
                    command += ["--smoke"]
                    output = subprocess.run(command, timeout=90, capture_output=True)
                    if output.returncode:
                        status.set("资源准备完成，但主程序启动失败；请查看安装目录。")
                        if args.report:
                            save_json(args.report, {**event, "passed": False, "launch_code": output.returncode})
                        if args.auto:
                            root.destroy()
                        return
                else:
                    subprocess.Popen(command, cwd=state["root"])
            root.destroy()
            return
        if root.winfo_exists():
            root.after(80, poll)
    root.after(80, poll)
    if args.auto:
        root.after(100, start)
    root.mainloop()


if __name__ == "__main__":
    main()
