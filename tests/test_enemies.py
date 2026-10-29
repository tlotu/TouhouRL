"""Binary enemy-pool fixtures; no game process is required."""
import struct
import unittest

from env.memory_reader import MemoryReader, MemoryReadError, REQUIRED
from env.action_emulator import ActionEmulator
from env.session import GameSession, observation
from tests.fakes import FakeClock, FakeKeyboard, FakeReader, sample
from tests.test_memory_reader import ByteProcess


class EnemyReadTests(unittest.TestCase):
    def test_observed_defeat_tail_does_not_fabricate_a_spell_result(self):
        # enemy-check-001: HP remains 1, collision turns off, then slot disappears.
        self.enemy(hp=1, flags1=134, flags2=13, flags3=1)
        enemies = self.reader.snapshot().enemies
        self.assertEqual(enemies[0].hp, 1)
        self.assertFalse(enemies[0].record()["collidable"])
        self.assertTrue(enemies[0].record()["is_boss"])
        clock = FakeClock()
        reader = FakeReader(sample(), sample(frame=11, enemies=enemies),
                            sample(frame=13, enemies=enemies), sample(frame=15))
        with GameSession(reader, ActionEmulator(FakeKeyboard()),
                         clock=clock, sleep=clock.sleep) as session:
            session.reset()
            self.assertFalse(session.step(0)[2])
            _, _, terminated, truncated, info = session.step(0)
            self.assertFalse(terminated or truncated)
            self.assertEqual(info["enemy_count"], 0)
            self.assertIsNone(info["reason"])

    def setUp(self):
        self.process = ByteProcess()
        addresses = {name: 0x100 + i * 16 for i, name in enumerate(REQUIRED)}
        addresses.update(PLAYER_BASE=0x1000, PLAYER_X=0x200,
                         BM_RVA=0x20000, EM_RVA=0x130000)
        self.reader = MemoryReader(self.process, 0, addresses)
        for name in ("CUR_STATE", "WANT_STATE"):
            struct.pack_into("<i", self.process.data, addresses[name], 2)
        struct.pack_into("<ff", self.process.data, addresses["PLAYER_X"], 192, 400)

    def enemy(self, slot=0, x=160, y=80, hp=1000, flags1=0x80, flags2=0x0A, flags3=0):
        offset = self.reader.addresses["EM_RVA"] + 8 + slot * 0x10B0
        struct.pack_into("<ff", self.process.data, offset + 0xB0, x, y)
        struct.pack_into("<BBB", self.process.data, offset + 0xBC, flags1, flags2, flags3)
        struct.pack_into("<i", self.process.data, offset + 0x234, hp)

    def test_position_hp_flags_and_last_slot(self):
        self.enemy(slot=255, x=123.5, y=-12, hp=7654)
        records = [e.record() for e in self.reader.snapshot().enemies]
        self.assertEqual(len(records), 1)
        self.assertEqual((records[0]["slot"], records[0]["x"], records[0]["y"], records[0]["hp"]),
                         (255, 123.5, -12, 7654))
        self.assertTrue(records[0]["is_boss"])
        self.assertTrue(records[0]["collidable"])
        self.assertTrue(records[0]["visible"])

    def test_only_occupied_bit_selects_slots(self):
        self.enemy(slot=0, flags1=0x7F, x=float("nan"))
        self.enemy(slot=3, flags1=0x81, flags2=0, flags3=8, hp=0)
        result = self.reader.snapshot().enemies
        self.assertEqual([e.slot for e in result], [3])
        self.assertFalse(result[0].record()["visible"])
        self.assertFalse(result[0].record()["collidable"])
        self.assertFalse(result[0].record()["is_boss"])

    def test_signed_hp_is_preserved_without_declaring_success(self):
        for hp in (0, -1, 2_000_000):
            with self.subTest(hp=hp):
                self.enemy(hp=hp)
                self.assertEqual(self.reader.snapshot().enemies[0].hp, hp)

    def test_hp_changes_and_slot_disappearance_are_fresh_reads(self):
        self.enemy(hp=1000)
        before = self.reader.snapshot()
        self.enemy(hp=500)
        after = self.reader.snapshot()
        self.enemy(flags1=0)
        self.assertEqual(before.enemies[0].hp, 1000)
        self.assertEqual(after.enemies[0].hp, 500)
        self.assertEqual(self.reader.snapshot().enemies, ())

    def test_nonfinite_active_coordinates_fail(self):
        for x, y in ((float("nan"), 0), (0, float("inf"))):
            with self.subTest(x=x, y=y):
                self.enemy(x=x, y=y)
                with self.assertRaisesRegex(MemoryReadError, "enemy position"):
                    self.reader.snapshot()

    def test_pool_is_read_once_and_short_reads_fail(self):
        calls = []
        original = self.process.read
        address = self.reader.addresses["EM_RVA"] + 8
        def read(addr, size):
            calls.append((addr, size))
            return original(addr, size)
        self.process.read = read
        self.reader.snapshot()
        self.assertEqual(calls.count((address, 256 * 0x10B0)), 1)
        self.process.read = lambda addr, size: original(addr, size)[:-1] if addr == address else original(addr, size)
        with self.assertRaises(MemoryReadError):
            self.reader.snapshot()

    def test_failure_menu_skips_stale_enemy_pool(self):
        self.enemy(x=float("nan"))
        self.process.data[self.reader.addresses["RETRY"]] = 255
        self.assertEqual(self.reader.snapshot().enemies, ())

    def test_torn_enemy_sample_is_retried(self):
        self.enemy()
        frames = iter([1, 2, 2, 2])
        self.reader.frame = lambda: next(frames)
        result = self.reader.snapshot()
        self.assertEqual(result.frame, 2)
        self.assertEqual(result.enemies[0].hp, 1000)

    def test_enemies_reach_info_and_policy_without_changing_bullet_observation(self):
        self.enemy()
        enemies = self.reader.snapshot().enemies
        start = sample(frame=11, enemies=enemies)
        end = sample(frame=13, enemies=enemies)
        clock = FakeClock()
        with GameSession(FakeReader(sample(), start, end), ActionEmulator(FakeKeyboard()),
                         clock=clock, sleep=clock.sleep) as session:
            session.reset()
            obs, _, terminated, _, info = session.step(0)
        self.assertEqual(info["enemy_count"], 1)
        self.assertEqual(info["enemies"][0]["hp"], 1000)
        self.assertEqual(obs["bullets"], observation(sample(frame=13))["bullets"])
        self.assertEqual(obs["enemies"][0][2:], [1.0, 1.0, 1.0, 1.0])
        self.assertNotIn("hp", obs)
        self.assertFalse(terminated)
