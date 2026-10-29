"""ABI checks only; no process attachment or input is performed."""
import ctypes
import os
import unittest

from env.windows import Input, KeyboardInput, ModuleEntry, ProcessEntry


@unittest.skipUnless(os.name == "nt" and ctypes.sizeof(ctypes.c_void_p) == 8,
                     "Windows x64 ABI only")
class WindowsLayoutTests(unittest.TestCase):
    def test_win32_structure_sizes(self):
        self.assertEqual(ctypes.sizeof(KeyboardInput), 24)
        self.assertEqual(ctypes.sizeof(Input), 40)
        self.assertEqual(ctypes.sizeof(ProcessEntry), 568)
        self.assertEqual(ctypes.sizeof(ModuleEntry), 1080)
