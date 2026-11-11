"""Threaded resource setup; no model inference is run in the UI thread."""
from __future__ import annotations
import threading
import time
from pathlib import Path

from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QFormLayout, QHBoxLayout, QLabel,
                              QLineEdit, QProgressBar, QPushButton, QTableWidget, QTableWidgetItem,
                              QVBoxLayout)

from .resource_download import (ComponentInstaller, DownloadCancelled, component_ready,
                                load_catalog, load_preferences, save_preferences)


class DownloadThread(QThread):
    progress = Signal(dict)
    result = Signal(str)

    def __init__(self, root, catalog, preferences, identifiers, repair, parent=None):
        super().__init__(parent)
        self.root, self.catalog, self.preferences = root, catalog, preferences
        self.identifiers, self.repair = identifiers, repair
        self.cancel = threading.Event()
        self.last_time, self.last_stage = 0.0, ""

    def emit_progress(self, **event):
        now = time.monotonic()
        stage = event.get("stage", "")
        if now - self.last_time > .08 or stage != self.last_stage or event.get("current") == event.get("total"):
            self.progress.emit(event)
            self.last_time, self.last_stage = now, stage

    def run(self):
        try:
            installer = ComponentInstaller(self.root, self.catalog, self.preferences, self.cancel, self.emit_progress)
            installer.run(self.identifiers, repair=self.repair)
            self.result.emit("完成；模型和环境已就绪，下一次任务直接使用本地文件。")
        except DownloadCancelled as error:
            self.result.emit(str(error))
        except Exception as error:
            self.result.emit("未完成：" + str(error))


class ResourceDownloadDialog(QDialog):
    def __init__(self, root: Path, parent=None, *, first_run=False):
        super().__init__(parent)
        self.root = root
        # Source sessions share the installed portable resources, as model settings do.
        if (root / "dist/VoiceBankGen/resources").is_dir():
            self.root = root / "dist/VoiceBankGen"
        self.catalog = load_catalog(self.root)
        self.thread = None
        self.setWindowTitle("首次使用：准备模型与环境" if first_run else "模型 / 环境下载")
        self.resize(940, 720)
        layout = QVBoxLayout(self)
        note = QLabel("选择下载源后勾选组件，点击下载。已安装的完整组件会跳过；支持续传，校验失败不安装。\n"
                      "完整版内置全部资源；联网版从 GitHub、Hugging Face、Python、PyTorch 和 PyPI 下载。")
        note.setWordWrap(True)
        layout.addWidget(note)
        preferences = load_preferences()
        form = QFormLayout()
        self.source = QComboBox()
        self.source.addItem("官方上游（GitHub / Hugging Face）", "official")
        self.source.addItem("镜像站（自定义地址）", "mirror")
        self.source.setCurrentIndex(max(0, self.source.findData(preferences["source"])))
        self.github = QLineEdit(preferences["github_mirror"])
        self.github.setPlaceholderText("如 https://镜像域名/{url}；未含 {url} 时作为官方完整 URL 的前缀")
        self.huggingface = QLineEdit(preferences["hf_mirror"])
        self.huggingface.setPlaceholderText("Hugging Face 镜像根地址，如 https://你的镜像域名")
        self.github.setToolTip("将 GitHub 下载 URL 中转到该站；下载内容仍按程序内固定哈希校验。")
        self.huggingface.setToolTip("仅替换 huggingface.co 域名，保留固定版本和文件路径。")
        self.other_mirror = QLineEdit(preferences["other_mirror"])
        self.other_mirror.setPlaceholderText("其他资源中转：URL 前缀或 {url} 模板；留空使用官方上游")
        self.other_mirror.setToolTip("Python / PyPI / PyTorch 下载的中转地址，需要支持完整官方 URL；不是 pip 索引地址。")
        self.proxy = QLineEdit(preferences["proxy"])
        self.proxy.setPlaceholderText("例如 http://127.0.0.1:7890；留空使用系统代理")
        self.proxy.setToolTip("全部资源下载均使用该 HTTP/HTTPS 代理，包括镜像下载；填写代理软件的 HTTP 端口。")
        form.addRow("下载源", self.source)
        form.addRow("GitHub 镜像", self.github)
        form.addRow("Hugging Face 镜像", self.huggingface)
        form.addRow("其他资源镜像", self.other_mirror)
        form.addRow("网络代理", self.proxy)
        layout.addLayout(form)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["选择", "组件", "下载大小", "状态"])
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)
        self.checks = {}
        self.refresh()
        layout.addWidget(QLabel("安装位置：" + str(self.root)))
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        layout.addWidget(self.progress)
        self.message = QLabel("已有工程和手动模型路径保留。默认 CPU；此窗口不会下载 MFA。")
        self.message.setWordWrap(True)
        self.message.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.message)
        controls = QHBoxLayout()
        self.save_button = QPushButton("保存下载源")
        self.save_button.setToolTip("保存官方/镜像选择和地址，下次启动保留；不开始下载。")
        self.save_button.clicked.connect(self.remember)
        self.detect_button = QPushButton("重新检测")
        self.detect_button.setToolTip("检查全部模型文件及关键环境文件；下载或修复时完整校验全部文件的 SHA-256。")
        self.detect_button.clicked.connect(self.refresh)
        self.download_button = QPushButton("下载 / 继续下载")
        self.download_button.setToolTip("下载勾选组件，复用缓存和续传片段；校验完成才安装。")
        self.download_button.clicked.connect(lambda: self.start(False))
        self.repair_button = QPushButton("校验 / 修复所选")
        self.repair_button.setToolTip("重新校验并安装所选组件，替换的旧文件保存在组件备份目录。环境修复需单文件安装器缓存。")
        self.repair_button.clicked.connect(lambda: self.start(True))
        self.cancel_button = QPushButton("取消下载")
        self.cancel_button.setEnabled(False)
        self.cancel_button.setToolTip("取消传输/校验，保留续传缓存；安装写入阶段等待安全完成。")
        self.cancel_button.clicked.connect(self.cancel_download)
        self.close_button = QPushButton("稍后配置" if first_run else "关闭")
        self.close_button.setToolTip("关闭此窗口；缺少模型时自动切片仍需先准备资源。")
        self.close_button.clicked.connect(self.reject)
        for button in (self.save_button, self.detect_button, self.download_button, self.repair_button, self.cancel_button, self.close_button):
            controls.addWidget(button)
        layout.addLayout(controls)

    def refresh(self):
        previous = {key: check.isChecked() for key, check in self.checks.items()}
        self.table.setRowCount(len(self.catalog["components"]))
        self.checks = {}
        for row, component in enumerate(self.catalog["components"]):
            ready = component_ready(self.root, component)
            check = QCheckBox()
            check.setChecked(previous.get(component["id"], component.get("required", False) and not ready))
            check.setEnabled(component["id"] != "app-core")
            self.checks[component["id"]] = check
            self.table.setCellWidget(row, 0, check)
            label = QTableWidgetItem(component["name"])
            label.setToolTip(component.get("description", ""))
            self.table.setItem(row, 1, label)
            amount = sum(artifact["size"] for artifact in component["artifacts"])
            bundled = all(not artifact.get("url") for artifact in component["artifacts"])
            self.table.setItem(row, 2, QTableWidgetItem(("随 EXE 提供 · " if bundled else "") + f"{amount / 1024**2:.1f} MB"))
            self.table.setItem(row, 3, QTableWidgetItem(("已安装（关键文件）" if component.get("probe_files") else "已就绪") if ready else "缺失 / 不完整"))
        self.table.resizeColumnsToContents()

    def remember(self):
        try:
            preferences = save_preferences({"source": self.source.currentData(), "github_mirror": self.github.text(), "hf_mirror": self.huggingface.text(),
                                            "other_mirror": self.other_mirror.text(), "proxy": self.proxy.text()})
            self.message.setText("下载源已保存。")
            return preferences
        except (OSError, ValueError) as error:
            self.message.setText(str(error))
            return None

    def start(self, repair):
        if self.thread:
            return
        identifiers = [key for key, check in self.checks.items() if check.isChecked() and check.isEnabled()]
        if not identifiers:
            self.message.setText("请勾选要下载或修复的组件。")
            return
        preferences = self.remember()
        if preferences is None:
            return
        self.thread = DownloadThread(self.root, self.catalog, preferences, identifiers, repair, self)
        self.thread.progress.connect(self.on_progress)
        self.thread.result.connect(self.message.setText)
        self.thread.finished.connect(self.finished_download)
        self.set_busy(True)
        self.thread.start()

    def set_busy(self, busy):
        for widget in (self.source, self.github, self.huggingface, self.other_mirror, self.proxy, self.table, self.save_button, self.detect_button,
                       self.download_button, self.repair_button, self.close_button):
            widget.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)

    def on_progress(self, event):
        if "name" in event:
            self.current_name = event["name"]
        self.message.setText(f"{getattr(self, 'current_name', '')} · {event.get('stage', '')}")
        total = max(1, event.get("total", 1))
        self.progress.setValue(int(1000 * event.get("current", 0) / total))
        if event.get("stage") in {"下载", "缓存"}:
            self.message.setText(f"{getattr(self, 'current_name', '')} · {event['stage']} · "
                                 f"{event.get('current', 0) / 1024**2:.1f} / {total / 1024**2:.1f} MB")

    def cancel_download(self):
        if self.thread:
            self.thread.cancel.set()
            self.cancel_button.setEnabled(False)
            self.message.setText("正在取消；若已进入安装阶段，等待安全完成。")

    def finished_download(self):
        thread, self.thread = self.thread, None
        thread.deleteLater()
        self.set_busy(False)
        self.refresh()

    def reject(self):
        if self.thread:
            self.cancel_download()
            return
        super().reject()

    def closeEvent(self, event):
        if self.thread:
            self.cancel_download()
            event.ignore()
        else:
            super().closeEvent(event)
