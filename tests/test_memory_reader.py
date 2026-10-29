import hashlib
import json
from pathlib import Path
import struct
import unittest
from unittest.mock import patch

from env.memory_reader import (MemoryReader, MemoryReadError, REQUIRED,
                               UnsupportedBuildError, load_profile)


class ByteProcess:
    def __init__(self):
        self.data = bytearray(0x120000)
        self.short = False
        self.closed = False

    def read(self, address, size):
        return bytes(self.data[address:address + size - int(self.short)])

    def close(self):
        self.closed = True


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.process = ByteProcess()
        self.addresses = {name: 0x100 + i * 16 for i, name in enumerate(REQUIRED)}
        self.addresses.update(PLAYER_BASE=0x1000, PLAYER_X=0x200, BM_RVA=0x20000)
        self.reader = MemoryReader(self.process, 0, self.addresses)
        for name, value, fmt in (("CALC_COUNT", 10, "<B"), ("CUR_STATE", 2, "<i"),
                                  ("WANT_STATE", 2, "<i"), ("LIVES", 0, "<B"),
                                  ("BOMBS", 3, "<B"), ("DEATHS", 0, "<i")):
            struct.pack_into(fmt, self.process.data, self.addresses[name], value)
        struct.pack_into("<ff", self.process.data, self.addresses["PLAYER_X"], 192, 400)

    def bullet(self, slot, state):
        offset = self.addresses["BM_RVA"] + 8 + slot * 0x620
        struct.pack_into("<ff", self.process.data, offset + 8, 1.0, 2.0)
        struct.pack_into("<ff", self.process.data, offset + 0x30, 100.0, 200.0)
        struct.pack_into("<H", self.process.data, offset + 0x44, state)
        struct.pack_into("<ff", self.process.data, offset + 0x5F4, 4.0, 6.0)
        return offset

    def test_zero_reserves_is_playable(self):
        result = self.reader.snapshot()
        self.assertTrue(result.ready)
        self.assertEqual(result.lives, 0)

    def test_bullet_filter_and_radius(self):
        self.bullet(2, 1)
        self.bullet(3, 3)
        self.bullet(4, 5)
        result = self.reader.snapshot()
        self.assertEqual([b.slot for b in result.bullets], [2, 3])
        self.assertEqual(result.bullets[0].radius, 2.0)
        self.assertEqual(result.bullets[0].vy, 2.0)

    def test_short_reads_raise_instead_of_returning_zero_state(self):
        self.process.short = True
        with self.assertRaises(MemoryReadError):
            self.reader.snapshot()

    def test_nonfinite_player_is_rejected(self):
        struct.pack_into("<f", self.process.data, self.addresses["PLAYER_X"], float("nan"))
        with self.assertRaises(MemoryReadError):
            self.reader.snapshot()

    def test_missing_hitbox_dimensions_use_marked_reference_fallback(self):
        offset = self.bullet(115, 1)
        for width, height in ((0, 4), (-1, 4), (100, 100),
                              (float("nan"), 4), (4, float("nan")),
                              (float("inf"), 4)):
            with self.subTest(width=width, height=height):
                struct.pack_into("<ff", self.process.data, offset + 0x5F4, width, height)
                bullet = self.reader.snapshot().bullets[0]
                self.assertEqual((bullet.slot, bullet.radius), (115, 3.0))
                self.assertTrue(bullet.radius_estimated)

    def test_invalid_bullet_position_still_fails(self):
        offset = self.bullet(115, 1)
        struct.pack_into("<f", self.process.data, offset + 0x30, float("nan"))
        with self.assertRaises(MemoryReadError):
            self.reader.snapshot()

    def test_retry_menu_is_not_playable(self):
        self.process.data[self.addresses["RETRY"]] = 255
        self.assertFalse(self.reader.snapshot().running)

    def test_menu_does_not_decode_stale_player_or_bullet_storage(self):
        self.process.data[self.addresses["RETRY"]] = 255
        self.process.data[self.addresses["PLAYER_BASE"] + 0x7898] = 255
        struct.pack_into("<f", self.process.data, self.addresses["PLAYER_X"], float("nan"))
        self.bullet(0, 1)
        result = self.reader.snapshot()
        self.assertFalse(result.ready)
        self.assertEqual(result.bullets, ())

    def test_samples_crossing_frames_are_retried(self):
        frames = iter([10, 11, 11, 11])
        self.reader.frame = lambda: next(frames)
        self.assertEqual(self.reader.snapshot().frame, 11)

    def test_consistently_torn_sample_fails(self):
        frames = iter(range(6))
        self.reader.frame = lambda: next(frames)
        with self.assertRaises(MemoryReadError):
            self.reader.snapshot()

    def test_close(self):
        self.reader.close()
        self.assertTrue(self.process.closed)

    def test_unknown_and_incomplete_builds_fail_closed(self):
        digest = hashlib.md5(b"test build").hexdigest()
        with patch.object(Path, "read_bytes", return_value=b"test build"):
            for data in ({"builds": {}}, {"builds": {digest: {"rvas": {}}}}):
                with patch.object(Path, "read_text", return_value=json.dumps(data)):
                    with self.assertRaises(UnsupportedBuildError):
                        load_profile("game.exe")
            data = {"builds": {digest: {"rvas": {
                name: hex(address) for name, address in self.addresses.items()}}}}
            with patch.object(Path, "read_text", return_value=json.dumps(data)):
                self.assertEqual(load_profile("game.exe"), (digest, self.addresses))
