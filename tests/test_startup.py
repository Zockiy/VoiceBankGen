import json
import os
import subprocess
import sys

from utau_forge.issue_labels import issue_text,issue_details


def test_unknown_issue_has_chinese_description_and_keeps_code_in_details():
    assert issue_text(["NEW_FLAG"])=="其他问题，查看悬停详情"
    assert "NEW_FLAG" in issue_details(["NEW_FLAG"])
    assert issue_text([])=="边界可用"


def test_worker_launcher_skips_gui_imports():
    code = """
import sys,json
from types import SimpleNamespace
from utau_forge.startup import launch
sys.argv=['VoiceBankGenWorker.exe','--worker','request.json']
result={}
def worker(path):
    result.update(path=path,gui_loaded='PySide6.QtWidgets' in sys.modules,app_loaded='utau_forge.app' in sys.modules)
sys.modules['utau_forge.desktop_worker']=SimpleNamespace(main=worker)
launch()
print(json.dumps(result))
"""
    result=subprocess.run([sys.executable,"-c",code],capture_output=True,text=True,check=True,env=os.environ.copy())
    assert json.loads(result.stdout)=={"path":"request.json","gui_loaded":False,"app_loaded":False}
