"""Keep process creation and cancellable process-tree shutdown off the GUI."""
from __future__ import annotations

import os
import subprocess
import threading

from PySide6.QtCore import QObject, QProcess, QRunnable, QThread, Signal, Slot


class _ProcessTask(QThread):
    output = Signal(bytes)
    processStarted = Signal(int)
    completed = Signal(int, object)
    failed = Signal(object, str)

    def __init__(self, program, arguments, mode, parent=None):
        super().__init__(parent)
        self.program, self.arguments, self.mode = program, arguments, mode
        self.process = None
        self.cancel = threading.Event()
        self.stop_lock = threading.Lock()
        self.stop_sent = False

    def run(self):
        code = -1
        try:
            if self.cancel.is_set():
                return
            self.process = self._start_process()
            self.processStarted.emit(self.process.pid)
            if self.cancel.is_set():
                self.stop()
            while data := self.process.stdout.read1(65536):
                self.output.emit(data)
            code = self.process.wait()
            if code > 2**31 - 1:
                code -= 2**32
            if code < 0 and not self.cancel.is_set():
                self.failed.emit(QProcess.ProcessError.Crashed, f"后台进程异常退出：{code}")
        except Exception as error:
            self.failed.emit(QProcess.ProcessError.FailedToStart, str(error))
        finally:
            if self.process is not None and self.process.stdout is not None:
                self.process.stdout.close()
            status = QProcess.ExitStatus.CrashExit if code < 0 or self.cancel.is_set() else QProcess.ExitStatus.NormalExit
            self.completed.emit(code, status)

    def _start_process(self):
        """CPython releases the GIL around Windows CreateProcess and pipe reads.

        QProcess.start's binding can retain it during a cold native EXE launch,
        starving Python GUI callbacks even when QProcess lives on a QThread.
        """
        return subprocess.Popen([self.program, *self.arguments], stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)

    def stop(self):
        self.cancel.set()
        with self.stop_lock:
            if self.process is None or self.stop_sent:
                return
            self.stop_sent = True
            threading.Thread(target=_stop_process, args=(self.process,), daemon=True).start()


def _stop_process(process):
    if process.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True,
                           creationflags=subprocess.CREATE_NO_WINDOW, timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            pass
    if process.poll() is None:
        try:
            process.kill()
        except OSError:
            pass


class BackgroundProcess(QObject):
    """QProcess-shaped GUI facade. All native process calls run on its thread."""
    readyReadStandardOutput = Signal()
    started = Signal()
    finished = Signal(int, object)
    errorOccurred = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.program, self.arguments = "", []
        self.mode = QProcess.ProcessChannelMode.MergedChannels
        self.buffer = bytearray()
        self.pid, self.error = 0, ""
        self.result = (-1, QProcess.ExitStatus.CrashExit)

    def setProgram(self, program): self.program = program
    def setArguments(self, arguments): self.arguments = arguments
    def setProcessChannelMode(self, mode): self.mode = mode
    def processId(self): return self.pid
    def errorString(self): return self.error
    def readAllStandardOutput(self):
        data = bytes(self.buffer)
        self.buffer.clear()
        return data

    def start(self):
        self.task = _ProcessTask(self.program, self.arguments, self.mode, self)
        self.thread = self.task
        self.task.output.connect(self.on_output)
        self.task.processStarted.connect(self.on_started)
        self.task.failed.connect(self.on_error)
        self.task.completed.connect(self.on_completed)
        self.thread.finished.connect(self.on_thread_finished)
        self.thread.start()

    @Slot(bytes)
    def on_output(self, data):
        self.buffer.extend(data)
        self.readyReadStandardOutput.emit()

    @Slot(int)
    def on_started(self, pid):
        self.pid = pid
        self.started.emit()

    @Slot(object, str)
    def on_error(self, error, message):
        self.error = message
        self.errorOccurred.emit(error)

    @Slot(int, object)
    def on_completed(self, code, status):
        self.result = (code, status)

    @Slot()
    def on_thread_finished(self):
        self.pid = 0
        self.finished.emit(*self.result)

    def kill(self):
        self.task.stop()


class TaskSignals(QObject):
    finished = Signal(object, object)


class FunctionTask(QRunnable):
    def __init__(self, function):
        super().__init__()
        self.function = function
        self.signals = TaskSignals()

    def run(self):
        try:
            self.signals.finished.emit(self.function(), None)
        except Exception as error:
            self.signals.finished.emit(None, str(error))
