import json
import sys
import time

import pytest
pytest.importorskip("PySide6.QtCore")
from PySide6.QtCore import QCoreApplication, QEventLoop, QProcess, QTimer, QThread

from utau_forge.desktop_jobs import BackgroundProcess, _ProcessTask


@pytest.fixture
def application():
    return QCoreApplication.instance() or QCoreApplication([])


def execute(process, *, cancel_after=None, limit=8):
    loop=QEventLoop()
    output=[]
    finished=[]
    errors=[]
    ticks=[time.monotonic()]
    timer=QTimer()
    timer.setInterval(10)
    timer.timeout.connect(lambda:ticks.append(time.monotonic()))
    process.readyReadStandardOutput.connect(lambda:output.append(process.readAllStandardOutput()))
    process.errorOccurred.connect(lambda error:errors.append(error))
    process.finished.connect(lambda code,status:(finished.append(code),loop.quit()))
    deadline=QTimer()
    deadline.setSingleShot(True)
    deadline.timeout.connect(loop.quit)
    timer.start()
    deadline.start(limit*1000)
    process.start()
    if cancel_after is not None:
        QTimer.singleShot(cancel_after,process.kill)
    loop.exec()
    timer.stop()
    deadline.stop()
    assert finished,"Background process did not terminate"
    assert not process.thread.isRunning()
    return finished,errors,b"".join(output),ticks


def test_slow_native_start_does_not_block_gui_event_loop(application,monkeypatch):
    original=_ProcessTask._start_process
    def slow_start(self):
        assert QThread.currentThread()!=QCoreApplication.instance().thread()
        time.sleep(.7)
        return original(self)
    monkeypatch.setattr(_ProcessTask,"_start_process",slow_start)
    process=BackgroundProcess()
    process.setProgram(sys.executable)
    process.setArguments(["-c","import time;time.sleep(.15);print('ready',flush=True)"])
    finished,errors,output,ticks=execute(process)
    assert finished==[0] and not errors and b"ready" in output
    assert len(ticks)>25
    assert max(b-a for a,b in zip(ticks,ticks[1:]))<.3


def test_cancellation_during_process_start_is_queued_and_gui_stays_alive(application,monkeypatch):
    original=_ProcessTask._start_process
    def slow_start(self):
        time.sleep(.5)
        return original(self)
    monkeypatch.setattr(_ProcessTask,"_start_process",slow_start)
    process=BackgroundProcess()
    process.setProgram(sys.executable)
    process.setArguments(["-c","import time;time.sleep(20)"])
    finished,_,_,ticks=execute(process,cancel_after=50)
    assert finished[0]!=0 and len(ticks)>10


def test_failed_start_emits_error_and_exactly_one_completion(application):
    process=BackgroundProcess()
    process.setProgram("Z:/VoiceBankGen-missing-worker.exe")
    finished,errors,_,_=execute(process)
    assert finished==[-1] and errors==[QProcess.ProcessError.FailedToStart]
    assert process.errorString()


def test_worker_bytes_and_final_output_survive_thread_shutdown(application):
    process=BackgroundProcess()
    process.setProgram(sys.executable)
    process.setArguments(["-c","import sys;sys.stdout.buffer.write('{\"stage\":\"准备完成\"}\\n'.encode('utf-8'));sys.stdout.flush()"])
    finished,errors,output,_=execute(process)
    assert finished==[0] and not errors
    assert json.loads(output)=={"stage":"准备完成"}
