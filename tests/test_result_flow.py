import struct
import unittest

from env.action_emulator import ActionEmulator, InputError
from env.memory_reader import MemoryReader, MemoryReadError, ResultMenu
from env.reset_controller import ResetController, ResetError
from env.session import GameSession
from scripts.diagnostics import run_episodes
from tests.fakes import FakeClock, FakeKeyboard, FakeReader, sample
from tests.test_memory_reader import ByteProcess


class ResultReadTests(unittest.TestCase):
    def test_inactive_does_not_decode_stale_menu_fields(self):
        process = ByteProcess()
        reader = MemoryReader(process, 0, {"SPELL_RESULT_ACTIVE": 0x5000, "SPELL_RESULT_MENU": 0x6000})
        self.assertEqual(reader._result_menu(), ResultMenu(False))
        process.data[0x5000] = 1
        raw = 0x6000 + 0xB40
        struct.pack_into("<i", process.data, raw + 4, 2)
        struct.pack_into("<i", process.data, raw + 0x254, 1)
        process.data[raw + 0x258] = 1
        self.assertEqual(reader._result_menu().outcome, "success")
        process.data[raw + 0x258] = 0
        self.assertEqual(reader._result_menu().outcome, "failure")
        struct.pack_into("<i", process.data, raw + 4, 3)
        with self.assertRaises(MemoryReadError):
            reader._result_menu()

    def test_unsettled_menu_has_no_outcome(self):
        self.assertIsNone(ResultMenu(True, 0, 0, 1).outcome)
        self.assertFalse(sample(result_menu=ResultMenu(True, 0, 0, 1)).running)


class ResultResetTests(unittest.TestCase):
    def simulate(self, selection, outcome=1, focus=True, responding=True):
        clock = FakeClock()
        keyboard = FakeKeyboard()
        keyboard.focus = focus
        controls = ActionEmulator(keyboard)
        state = {"selection": selection, "confirmed": False, "loading": True}
        original = keyboard.key_down
        def down(key):
            original(key)
            if responding and key == "down":
                state["selection"] = (state["selection"] + 1) % 3
            if key == "z":
                self.assertEqual(state["selection"], 2)
                state["confirmed"] = True
        keyboard.key_down = down
        reader = FakeReader(sample())
        def snapshot():
            if not state["confirmed"]:
                return sample(result_menu=ResultMenu(True, state["selection"], 1, outcome))
            if state["loading"]:
                state["loading"] = False
                return sample(frame=0, scene=12)
            return sample(frame=12, result_menu=ResultMenu(False))
        reader.snapshot = snapshot
        reset = ResetController(reader, controls, True, timeout=5, clock=clock, sleep=clock.sleep)
        return reset, keyboard, controls

    def test_success_default_moves_twice_then_confirms(self):
        reset, keyboard, controls = self.simulate(0)
        self.assertTrue(reset.reset().ready)
        self.assertEqual([key for event, key in keyboard.events if event == "down"], ["down", "down", "z"])
        self.assertFalse(controls.held)

    def test_existing_retry_selection_is_not_moved(self):
        for outcome in (0, 1):
            with self.subTest(outcome=outcome):
                reset, keyboard, _ = self.simulate(2, outcome)
                reset.reset()
                self.assertEqual([key for event, key in keyboard.events if event == "down"], ["z"])

    def test_save_replay_selection_moves_once(self):
        reset, keyboard, _ = self.simulate(1)
        reset.reset()
        self.assertEqual([key for event, key in keyboard.events if event == "down"], ["down", "z"])

    def test_unresponsive_cursor_never_gets_confirmed(self):
        reset, keyboard, controls = self.simulate(0, responding=False)
        with self.assertRaises(ResetError):
            reset.reset()
        self.assertNotIn(("down", "z"), keyboard.events)
        self.assertFalse(controls.held)

    def test_lost_focus_sends_no_navigation(self):
        reset, keyboard, _ = self.simulate(0, focus=False)
        with self.assertRaises(InputError):
            reset.reset()
        self.assertFalse(keyboard.events)

    def test_phase_transition_waits_without_keys(self):
        clock = FakeClock()
        keyboard = FakeKeyboard()
        reader = FakeReader(sample(result_menu=ResultMenu(True, 0, 2, 1)))
        reset = ResetController(reader, ActionEmulator(keyboard), True, timeout=0.2,
                                clock=clock, sleep=clock.sleep)
        with self.assertRaises(ResetError):
            reset.reset()
        self.assertFalse(keyboard.events)


class ResultEpisodeTests(unittest.TestCase):
    def test_success_retry_then_failure_runs_two_episodes(self):
        clock = FakeClock()
        keyboard = FakeKeyboard()
        controls = ActionEmulator(keyboard)
        state = {"mode": "initial", "reads": 0, "episode": 0, "selection": 0}
        reader = FakeReader(sample())
        original = keyboard.key_down
        def down(key):
            original(key)
            if state["mode"] == "menu":
                if key == "down":
                    state["selection"] = (state["selection"] + 1) % 3
                elif key == "z":
                    self.assertEqual(state["selection"], 2)
                    state["mode"] = "loading"
        keyboard.key_down = down
        def snapshot():
            if state["mode"] == "initial":
                state["reads"] += 1
                if state["reads"] == 2:
                    state["mode"] = "combat"
                return sample(frame=9 + state["reads"], result_menu=ResultMenu(False))
            if state["mode"] == "loading":
                state["mode"] = "ready"
                return sample(frame=0, scene=12)
            if state["mode"] == "ready":
                state["episode"] = 1
                state["mode"] = "combat"
                return sample(frame=11, result_menu=ResultMenu(False))
            if state["mode"] == "combat":
                state["mode"] = "menu"
                state["selection"] = 0 if state["episode"] == 0 else 2
            return sample(frame=13, result_menu=ResultMenu(True, state["selection"], 1,
                                                         int(state["episode"] == 0)))
        reader.snapshot = snapshot
        rows = []
        with GameSession(reader, controls, spell_retry=True, clock=clock, sleep=clock.sleep) as session:
            run_episodes(session, 2, rows.append, policy="stay")
        self.assertEqual([r["reason"] for r in rows if r["event"] == "episode_end"],
                         ["spell_success", "spell_failure"])
        self.assertEqual(rows[-1]["success_rate"], 0.5)
        self.assertTrue(rows[-1]["all_requested_completed"])
        self.assertFalse(controls.held)

    def test_death_before_result_activation_waits_for_verified_failure(self):
        clock = FakeClock()
        controls = ActionEmulator(FakeKeyboard())
        reader = FakeReader(sample(), sample(frame=11),
                            sample(frame=13, deaths=1, retry=True, result_menu=ResultMenu(False)),
                            sample(frame=14, deaths=1, result_menu=ResultMenu(True, 2, 1, 0)))
        with GameSession(reader, controls, clock=clock, sleep=clock.sleep) as session:
            session.reset()
            _, _, terminated, truncated, info = session.step(0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["reason"], "spell_failure")

    def test_success_and_timeout_failure_are_terminal(self):
        for flag, reason in ((1, "spell_success"), (0, "spell_failure")):
            with self.subTest(flag=flag):
                clock = FakeClock()
                controls = ActionEmulator(FakeKeyboard())
                reader = FakeReader(sample(), sample(frame=11),
                                    sample(frame=13, result_menu=ResultMenu(True, 0, 0, flag)),
                                    sample(frame=14, result_menu=ResultMenu(True, 0, 1, flag)))
                rows = []
                with GameSession(reader, controls, clock=clock, sleep=clock.sleep) as session:
                    run_episodes(session, 1, rows.append, policy="stay")
                end = next(r for r in rows if r["event"] == "episode_end")
                self.assertEqual(end["reason"], reason)
                self.assertEqual(rows[-1]["success_rate"], float(flag))
                self.assertEqual(rows[-1]["unresolved_episodes"], 0)
                self.assertFalse(controls.held)
