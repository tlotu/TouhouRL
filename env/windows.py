"""Lazy Windows bindings for read-only process access and foreground input."""
from contextlib import contextmanager
import ctypes
from ctypes import wintypes as w
import os
from pathlib import Path
import time

from .action_emulator import InputError
from .memory_reader import MemoryReader, load_profile


INPUT_TIMING_CONTRACT = "windows-sequential-input-v1"
MULTI_KEY_INPUT_TIMING_CONTRACT = "windows-multi-key-input-v1"


def libraries():
    if os.name != "nt":
        raise OSError("Live game access requires Windows")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    user = ctypes.WinDLL("user32", use_last_error=True)
    return kernel, user


class ProcessEntry(ctypes.Structure):
    _fields_ = [("size", w.DWORD), ("usage", w.DWORD), ("pid", w.DWORD),
                ("heap", ctypes.c_size_t), ("module", w.DWORD), ("threads", w.DWORD),
                ("parent", w.DWORD), ("priority", w.LONG), ("flags", w.DWORD),
                ("exe", w.WCHAR * 260)]


class ModuleEntry(ctypes.Structure):
    _fields_ = [("size", w.DWORD), ("id", w.DWORD), ("pid", w.DWORD),
                ("global_usage", w.DWORD), ("usage", w.DWORD),
                ("base", ctypes.c_void_p), ("bytes", w.DWORD), ("handle", w.HMODULE),
                ("name", w.WCHAR * 256), ("path", w.WCHAR * 260)]


def discover(pid=None):
    if ctypes.sizeof(ctypes.c_void_p) != 8:
        raise OSError("Use 64-bit Python for th06nc")
    kernel, _ = libraries()
    kernel.CreateToolhelp32Snapshot.argtypes = [w.DWORD, w.DWORD]
    kernel.CreateToolhelp32Snapshot.restype = w.HANDLE
    kernel.CloseHandle.argtypes = [w.HANDLE]
    kernel.CloseHandle.restype = w.BOOL
    kernel.Process32FirstW.argtypes = [w.HANDLE, ctypes.POINTER(ProcessEntry)]
    kernel.Process32NextW.argtypes = [w.HANDLE, ctypes.POINTER(ProcessEntry)]
    kernel.Module32FirstW.argtypes = [w.HANDLE, ctypes.POINTER(ModuleEntry)]
    kernel.Module32NextW.argtypes = [w.HANDLE, ctypes.POINTER(ModuleEntry)]

    def snapshot(flags, target):
        handle = kernel.CreateToolhelp32Snapshot(flags, target)
        if handle == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        return handle

    handle = snapshot(2, 0)
    candidates = []
    try:
        entry = ProcessEntry()
        entry.size = ctypes.sizeof(entry)
        ok = kernel.Process32FirstW(handle, ctypes.byref(entry))
        while ok:
            if entry.exe.lower() == "th06nc.exe" and (pid is None or entry.pid == pid):
                candidates.append(entry.pid)
            ok = kernel.Process32NextW(handle, ctypes.byref(entry))
    finally:
        kernel.CloseHandle(handle)
    if len(candidates) != 1:
        raise RuntimeError("Expected one th06nc.exe process; start the game or specify --pid")
    pid = candidates[0]
    handle = snapshot(0x18, pid)
    try:
        entry = ModuleEntry()
        entry.size = ctypes.sizeof(entry)
        ok = kernel.Module32FirstW(handle, ctypes.byref(entry))
        while ok:
            if entry.name.lower() == "th06nc.exe":
                return pid, entry.base, Path(entry.path)
            ok = kernel.Module32NextW(handle, ctypes.byref(entry))
    finally:
        kernel.CloseHandle(handle)
    raise RuntimeError("Could not find the game module")


class ReadOnlyProcess:
    def __init__(self, pid):
        self.kernel, _ = libraries()
        self.kernel.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
        self.kernel.OpenProcess.restype = w.HANDLE
        self.kernel.CloseHandle.argtypes = [w.HANDLE]
        self.kernel.ReadProcessMemory.argtypes = [w.HANDLE, ctypes.c_void_p,
                                                ctypes.c_void_p, ctypes.c_size_t,
                                                ctypes.POINTER(ctypes.c_size_t)]
        self.kernel.ReadProcessMemory.restype = w.BOOL
        self.handle = self.kernel.OpenProcess(0x1010, False, pid)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())

    def read(self, address, size):
        buffer = ctypes.create_string_buffer(size)
        read = ctypes.c_size_t()
        ok = self.kernel.ReadProcessMemory(self.handle, address, buffer, size, ctypes.byref(read))
        return buffer.raw if ok and read.value == size else None

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


class KeyboardInput(ctypes.Structure):
    _fields_ = [("vk", w.WORD), ("scan", w.WORD), ("flags", w.DWORD),
                ("time", w.DWORD), ("extra", ctypes.c_size_t)]


class MouseInput(ctypes.Structure):
    _fields_ = [("x", w.LONG), ("y", w.LONG), ("data", w.DWORD),
                ("flags", w.DWORD), ("time", w.DWORD), ("extra", ctypes.c_size_t)]


class InputUnion(ctypes.Union):
    _fields_ = [("keyboard", KeyboardInput), ("mouse", MouseInput)]


class Input(ctypes.Structure):
    _fields_ = [("type", w.DWORD), ("value", InputUnion)]


class WindowsKeyboard:
    _input_profile = None
    SCANS = {"up": 0x48, "down": 0x50, "left": 0x4B, "right": 0x4D,
             "z": 0x2C, "shift": 0x2A}

    def __init__(self, pid):
        _, self.user = libraries()
        self.pid = pid
        self.user.GetForegroundWindow.restype = w.HWND
        self.user.GetWindowThreadProcessId.argtypes = [w.HWND, ctypes.POINTER(w.DWORD)]
        self.user.SendInput.argtypes = [w.UINT, ctypes.POINTER(Input), ctypes.c_int]
        self.user.SendInput.restype = w.UINT
        self.user.GetAsyncKeyState.argtypes = [ctypes.c_int]
        self.user.GetAsyncKeyState.restype = ctypes.c_short

    @contextmanager
    def profile_input(self):
        """Collect native call timings for one action dispatch."""
        if self._input_profile is not None:
            raise RuntimeError("Input dispatch profiling is already active")
        profile = {"foreground_seconds": [], "events": []}
        self._input_profile = profile
        try:
            yield profile
        finally:
            self._input_profile = None

    def is_foreground(self):
        profile = self._input_profile
        if profile is not None:
            started = time.perf_counter()
        pid = w.DWORD()
        hwnd = self.user.GetForegroundWindow()
        self.user.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if profile is not None:
            profile["foreground_seconds"].append(time.perf_counter() - started)
        return pid.value == self.pid

    def _input_event(self, key, up):
        event = Input()
        event.type = 1
        event.value.keyboard.scan = self.SCANS[key]
        event.value.keyboard.flags = 8 | (2 if up else 0) | (1 if key in ("up", "down", "left", "right") else 0)
        return event

    def _send(self, key, up):
        if not up and not self.is_foreground():
            raise InputError("Game window is not in the foreground")
        profile = self._input_profile
        if profile is not None:
            started = time.perf_counter()
        event = self._input_event(key, up)
        pointer, size = ctypes.byref(event), ctypes.sizeof(event)
        if profile is not None:
            prepared = time.perf_counter()
        sent = self.user.SendInput(1, pointer, size)
        if profile is not None:
            returned = time.perf_counter()
            profile["events"].append({"key": key, "down": not up, "prepare_seconds": prepared - started,
                                      "send_input_seconds": returned - prepared})
        if sent != 1:
            raise InputError("SendInput failed; check process privilege levels")

    def key_down(self, key):
        self._send(key, False)

    def send_batch(self, changes):
        if not changes:
            return 0
        if not self.is_foreground():
            raise InputError("Game window is not in the foreground")
        profile = self._input_profile
        if profile is not None:
            started = time.perf_counter()
        events = (Input * len(changes))(*(self._input_event(key, not down) for key, down in changes))
        if profile is not None:
            prepared = time.perf_counter()
        sent = self.user.SendInput(len(changes), events, ctypes.sizeof(Input))
        if profile is not None:
            returned = time.perf_counter()
            profile["events"].extend({"key": key, "down": down} for key, down in changes)
            profile["batch"] = {"prepare_seconds": prepared - started, "send_input_seconds": returned - prepared}
        return sent

    def shift_state(self):
        # OS high bits describe key state, not the game's consumed input.
        return {name: bool(self.user.GetAsyncKeyState(vk) & 0x8000)
                for name, vk in (("shift", 0x10), ("left_shift", 0xA0), ("right_shift", 0xA1))}

    def key_up(self, key):
        self._send(key, True)


def connect(pid=None, profiles_path=None):
    pid, base, exe = discover(pid)
    digest, addresses = load_profile(exe, **({"profiles_path": profiles_path} if profiles_path else {}))
    reader = MemoryReader(ReadOnlyProcess(pid), base, addresses)
    return reader, pid, digest
