"""Windows Recycle Bin only; never fall back to permanent deletion."""
from pathlib import Path
import ctypes
import sys
import threading
from uuid import UUID


def recycle_paths(paths: list[Path]) -> None:
    if sys.platform != "win32":
        raise RuntimeError("当前平台暂不支持安全移入回收站，已停止；未执行永久删除")
    failures = []

    def run():
        from ctypes import wintypes
        ole = ctypes.OleDLL("ole32")
        shell = ctypes.OleDLL("shell32")
        guid_type = ctypes.c_ubyte * 16
        def guid(value):
            return guid_type.from_buffer_copy(UUID(value).bytes_le)
        def checked(value):
            if value & 0x80000000:
                raise OSError(value, "Windows 无法移入回收站，已停止；未执行永久删除")
        def call(pointer, index, argtypes, *args):
            table = ctypes.cast(pointer, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
            return ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, *argtypes)(table[index])(pointer, *args)
        op, items = ctypes.c_void_p(), []
        initialized = False
        try:
            checked(ole.CoInitializeEx(None, 2))
            initialized = True
            clsid = guid("3ad05575-8857-4850-9277-11b85bdb8e09")
            iid = guid("947aab5f-0a5c-4c13-b4d6-4bf7836fc9f8")
            checked(ole.CoCreateInstance(ctypes.byref(clsid), None, 1, ctypes.byref(iid), ctypes.byref(op)))
            # RECYCLEONDELETE, ADDUNDORECORD, EARLYFAILURE, ALLOWUNDO,
            # SILENT, NOCONFIRMATION, NOERRORUI, WANTNUKEWARNING.
            checked(call(op, 5, [wintypes.DWORD], 0x00080000 | 0x20000000 | 0x00100000 | 0x40 | 0x4 | 0x10 | 0x400 | 0x4000))
            shell_iid = guid("43826d1e-e718-42ee-bc55-a1e261c37bfe")
            shell.SHCreateItemFromParsingName.argtypes = [ctypes.c_wchar_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            for path in paths:
                item = ctypes.c_void_p()
                checked(shell.SHCreateItemFromParsingName(str(path), None, ctypes.byref(shell_iid), ctypes.byref(item)))
                items.append(item)
                checked(call(op, 18, [ctypes.c_void_p, ctypes.c_void_p], item, None))
            checked(call(op, 21, []))
            aborted = wintypes.BOOL()
            checked(call(op, 22, [ctypes.POINTER(wintypes.BOOL)], ctypes.byref(aborted)))
            if aborted.value or any(path.exists() for path in paths):
                raise RuntimeError("回收站操作未完成，已停止；未执行永久删除")
        except Exception as error:
            failures.append(error)
        finally:
            for item in items:
                call(item, 2, [])
            if op.value:
                call(op, 2, [])
            if initialized:
                ole.CoUninitialize()

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()
    if failures:
        raise failures[0]
