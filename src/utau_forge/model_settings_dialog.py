"""Global resource paths, with visible automatic detection and remembered edits."""
from pathlib import Path

from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QFileDialog, QFormLayout,
                              QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton,
                              QVBoxLayout)

from .model_settings import MODEL_FIELDS, detect_model_paths, load_model_paths, save_model_paths


class ModelSettingsDialog(QDialog):
    def __init__(self, root: Path, project: Path | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("模型设置")
        self.resize(1000, 640)
        self.root, self.project = root, project
        self.fields, self.locations = {}, {}
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("留空 = 自动识别工程和程序附带的本地资源。保存后下次启动继续使用；下一次任务生效。"))
        current = load_model_paths()
        for key, label, kind, help_text in MODEL_FIELDS:
            group = QGroupBox(label)
            form = QFormLayout(group)
            row = QHBoxLayout()
            field = QLineEdit(current[key])
            field.setPlaceholderText("自动识别（可输入绝对路径）")
            field.setToolTip(help_text)
            self.fields[key] = field
            row.addWidget(field)
            browse = QPushButton("浏览…")
            browse.setToolTip("选择此模型的本地目录或文件；保存时检查必需文件是否齐全。")
            browse.clicked.connect(lambda checked=False, k=key, t=kind: self.browse(k, t))
            row.addWidget(browse)
            reset = QPushButton("自动识别")
            reset.setToolTip("清空此项手动路径，恢复自动选择；点击保存后记忆。")
            reset.clicked.connect(field.clear)
            row.addWidget(reset)
            form.addRow(row)
            location = QLabel()
            location.setWordWrap(True)
            self.locations[key] = location
            form.addRow(location)
            field.textChanged.connect(self.refresh)
            layout.addWidget(group)
        self.error_label = QLabel()
        self.error_label.setWordWrap(True)
        layout.addWidget(self.error_label)
        controls = QHBoxLayout()
        detect = QPushButton("重新检测")
        detect.setToolTip("重新读取工程和程序附带的模型位置，更新下方显示；不改变手动填写的路径。")
        detect.clicked.connect(self.refresh)
        controls.addWidget(detect)
        downloads = QPushButton("模型 / 环境下载")
        downloads.setToolTip("选择官方上游或镜像站，下载缺失组件；校验后安装到软件目录。")
        downloads.clicked.connect(self.downloads)
        controls.addWidget(downloads)
        controls.addStretch()
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        save = buttons.button(QDialogButtonBox.StandardButton.Save)
        save.setText("保存")
        save.setToolTip("检查并保存全部模型路径，下次启动保留；取消不会改变已保存设置。")
        cancel = buttons.button(QDialogButtonBox.StandardButton.Cancel)
        cancel.setText("取消")
        cancel.setToolTip("关闭窗口，不保存本次路径修改。")
        buttons.accepted.connect(self.save)
        buttons.rejected.connect(self.reject)
        controls.addWidget(buttons)
        layout.addLayout(controls)
        self.refresh()

    def refresh(self, *_):
        detected = detect_model_paths(self.root, self.project)
        for key, field in self.fields.items():
            text = field.text().strip()
            self.locations[key].setText("手动指定：" + text if text else "自动识别：" + (detected[key] or "未找到，请指定路径"))

    def browse(self, key, kind):
        current = self.fields[key].text()
        if kind == "directory":
            chosen = QFileDialog.getExistingDirectory(self, "选择模型目录", current)
        else:
            chosen, _ = QFileDialog.getOpenFileName(self, "选择模型文件", current, "ZIP (*.zip)" if key == "mfa_acoustic" else "词典 (*.dict)")
        if chosen:
            self.fields[key].setText(chosen)

    def downloads(self):
        from .resource_download_dialog import ResourceDownloadDialog
        ResourceDownloadDialog(self.root, self).exec()
        self.refresh()

    def save(self):
        try:
            save_model_paths({key: field.text() for key, field in self.fields.items()})
        except (OSError, ValueError) as error:
            self.error_label.setText(str(error))
            return
        self.accept()
