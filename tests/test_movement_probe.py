"""Candidate telemetry and mode schedules without native reads or key injection."""
import json
import struct
from types import SimpleNamespace
import unittest

from env.movement_probe import BASELINE_BUILD, MovementProbe, annotate_movement
from env.memory_reader import MemoryReadError
from env.windows import WindowsKeyboard
from env.action_emulator import ActionEmulator
from tests.fakes import FakeKeyboard
from scripts.probe_input import MOVEMENT_PHASES, MovementModePolicy


class CandidateReader:
    addresses = {"PLAYER_BASE": 0x4FF3A0}

    def __init__(self, frames=(10, 10)):
        self.frames = iter(frames)
        self.raw = bytearray(20)
        self.raw[0] = 1
        struct.pack_into("<ffff", self.raw, 4, 4, 2, 2.828427, 1.414214)

    def frame(self):
        return next(self.frames)

    def _read(self, address, size):
        if address == 0xA6EC60 and size == 2:
            return struct.pack("<H", 4)
        if address == 0x4FF3A0 + 0x785C and size == 20:
            return bytes(self.raw)
        raise AssertionError("Unexpected read")


class MovementProbeTests(unittest.TestCase):
    def test_candidate_values_remain_unverified_and_frame_tagged(self):
        r = MovementProbe(CandidateReader(), BASELINE_BUILD).capture(10)
        self.assertTrue(r["same_transition_frame"])
        self.assertTrue(r["slow_input_bit"])
        self.assertEqual(r["focus_raw"], 1)
        self.assertEqual(r["speed_parameters"]["slow"], 2)
        self.assertFalse(r["verified"])

    def test_cross_frame_capture_is_not_tagged_as_aligned(self):
        r = MovementProbe(CandidateReader((10, 11)), BASELINE_BUILD).capture(10)
        self.assertFalse(r["same_transition_frame"])

    def test_nonfinite_candidates_are_json_safe_not_fabricated(self):
        reader = CandidateReader()
        struct.pack_into("<f", reader.raw, 4, float("nan"))
        r = MovementProbe(reader, BASELINE_BUILD).capture(10)
        self.assertFalse(r["plausible"])
        self.assertIsNone(r["speed_parameters"]["fast"])
        json.dumps(r, allow_nan=False)

    def test_read_errors_are_diagnostic_and_other_builds_rejected(self):
        reader = CandidateReader()
        reader._read = lambda *args: (_ for _ in ()).throw(MemoryReadError("short"))
        self.assertFalse(MovementProbe(reader, BASELINE_BUILD).capture(10)["available"])
        with self.assertRaises(ValueError):
            MovementProbe(reader, "other")

    def test_os_key_state_uses_high_bit_not_pressed_since_last_call(self):
        keyboard = WindowsKeyboard.__new__(WindowsKeyboard)
        keyboard.user = SimpleNamespace(GetAsyncKeyState=lambda vk: {0x10: -32768, 0xA0: 1, 0xA1: 0}[vk])
        self.assertEqual(keyboard.shift_state(), {"shift": True, "left_shift": False, "right_shift": False})

    def test_mode_schedule_uses_frames_and_changes_intent_only(self):
        session = SimpleNamespace(elapsed_frames=0, controls=SimpleNamespace(slow=True))
        policy = MovementModePolicy(session)
        end = 0
        for action, frames, slow in MOVEMENT_PHASES:
            for elapsed in (end, end+frames-1):
                session.elapsed_frames = elapsed
                self.assertEqual(policy({}), action)
                self.assertEqual(session.controls.slow, slow)
            end += frames
        self.assertEqual(end, 108)
        session.elapsed_frames = end+1
        self.assertEqual(policy({}), 0)
        self.assertTrue(session.controls.slow)

    def test_annotation_records_sources_and_skips_menu(self):
        controls = SimpleNamespace(slow=True, held={"shift", "right", "z"},
                                   keyboard=SimpleNamespace(shift_state=lambda: {"shift": True}))
        probe = MovementProbe(CandidateReader(), BASELINE_BUILD)
        row = {"event": "transition", "info": {"player_x": 192, "frame": 10}}
        annotated = annotate_movement(row, probe, controls)
        self.assertNotIn("movement_diagnostic", row)
        self.assertTrue(annotated["movement_diagnostic"]["requested_slow"])
        menu = {"event": "transition", "info": {"player_x": None}}
        self.assertIs(annotate_movement(menu, probe, controls), menu)

    def test_requested_mode_change_sends_shift_without_changing_direction(self):
        keyboard = FakeKeyboard()
        controls = ActionEmulator(keyboard)
        controls.apply(3)
        keyboard.events.clear()
        controls.slow = False
        controls.apply(3)
        self.assertEqual(keyboard.events, [("up", "shift")])
        self.assertIn("right", controls.held)
        keyboard.events.clear()
        controls.slow = True
        controls.apply(3)
        self.assertEqual(keyboard.events, [("down", "shift")])
        controls.release_all()
        self.assertFalse(controls.held)

    def test_truncation_telemetry_is_tagged_after_key_cleanup(self):
        controls = SimpleNamespace(slow=True, held=set(),
                                   keyboard=SimpleNamespace(shift_state=lambda: {"shift": False}))
        probe = MovementProbe(CandidateReader(), BASELINE_BUILD)
        row = {"event": "transition", "truncated": True, "terminated": False,
               "info": {"player_x": 192, "frame": 10}}
        result = annotate_movement(row, probe, controls)["movement_diagnostic"]
        self.assertTrue(result["post_step_cleanup"])
        self.assertFalse(result["os_shift"]["shift"])
        self.assertTrue(result["game_candidates"]["slow_input_bit"])
