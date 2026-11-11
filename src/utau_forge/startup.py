"""Lightweight launcher: show startup stages before importing the audio UI."""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

_application = None
_window = None
_started = time.perf_counter()
_steps = []


def icon_path():
    return Path(__file__).resolve().parent / "data/ui/voicebankgen.png"


def begin_startup():
    global _application, _window
    if _window is not None:
        return
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QFont, QIcon, QFontDatabase
    from .startup_window import StartupWindow
    _application = QApplication.instance() or QApplication(sys.argv)
    font_path=Path(os.environ.get("WINDIR","C:/Windows"))/"Fonts/msyh.ttc"
    if font_path.is_file():
        QFontDatabase.addApplicationFont(str(font_path))
    _application.setFont(QFont("Microsoft YaHei", 10))
    _application.setStyle("Fusion")
    from PySide6.QtCore import QSettings
    from .ui_theme import apply_theme,THEME_KEY,DEFAULT_THEME
    settings=QSettings("UTAUForge","DesktopWorkflowFork")
    settings.setFallbacksEnabled(False)
    apply_theme(_application,settings.value(THEME_KEY,DEFAULT_THEME))
    _application.setWindowIcon(QIcon(str(icon_path())))
    _window = StartupWindow(icon_path())
    _window.show()
    startup_progress(5, "正在启动 VoiceBankGen…")


def startup_progress(value, text):
    if _window is None:
        return
    _window.set_stage(value, text)
    _application.processEvents()
    _steps.append({"value": value, "stage": text, "elapsed_s": round(time.perf_counter()-_started, 3),
                   "visible": _window.isVisible(), "numpy_loaded": "numpy" in sys.modules})
    report = os.environ.get("UFORGE_STARTUP_REPORT")
    if report:
        Path(report).write_text(json.dumps(_steps, ensure_ascii=False, indent=2), encoding="utf-8")
    screenshot = os.environ.get("UFORGE_STARTUP_IMAGE")
    if screenshot and value == 55:
        _window.grab().save(screenshot)


def finish_startup():
    global _window
    if _window is not None:
        startup_progress(100, "加载完成")
        _window.close()
        _window = None


def launch():
    # Workers need no GUI, pyqtgraph or multimedia imports.
    if "--worker" in sys.argv:
        from .desktop_worker import main
        main(sys.argv[sys.argv.index("--worker")+1])
        return
    begin_startup()
    try:
        from .app import main
        main()
    except Exception as error:
        if _window is None:
            raise
        _window.show_error(str(error))
        _application.exec()
        raise SystemExit(1)
