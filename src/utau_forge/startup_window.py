"""Small startup-only Qt surface; no numerical/audio dependencies."""
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout, QProgressBar, QPushButton


class StartupWindow(QWidget):
    def __init__(self, icon):
        super().__init__(None, Qt.WindowType.SplashScreen | Qt.WindowType.WindowStaysOnTopHint)
        self.setWindowTitle("VoiceBankGen · 正在加载")
        self.setFixedSize(460, 210)
        self.setObjectName("startupWindow")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground,True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 22)
        row = QHBoxLayout()
        mark = QLabel()
        mark.setPixmap(QPixmap(str(icon)).scaled(60, 60, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        row.addWidget(mark)
        title = QLabel("VoiceBankGen")
        title.setStyleSheet("font-size:24px;font-weight:600;")
        row.addWidget(title)
        row.addStretch()
        layout.addLayout(row)
        layout.addSpacing(12)
        self.label = QLabel("正在启动…")
        self.label.setWordWrap(True)
        layout.addWidget(self.label)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(10)
        layout.addWidget(self.progress)
        self.percentage = QLabel("启动阶段 0%")
        self.percentage.setProperty("role","muted")
        self.percentage.setStyleSheet("font-size:11px;")
        layout.addWidget(self.percentage)

    def set_stage(self, value, text):
        self.progress.setValue(value)
        self.label.setText(text)
        self.percentage.setText(f"启动阶段 {value}%")

    def show_error(self, message):
        self.setFixedSize(460, 310)
        self.label.setText("启动失败："+message)
        self.percentage.setText("请检查程序文件是否完整")
        button = QPushButton("关闭")
        button.clicked.connect(self.close)
        self.layout().addWidget(button)
