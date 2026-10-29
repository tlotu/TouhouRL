import unittest

from env.action_emulator import ActionEmulator, DIRECTIONS, InputError
from env.frame_stepper import FrameStepper, FrameTimeout
from env.reset_controller import ResetController, ResetError
from tests.fakes import FakeClock, FakeKeyboard, FakeReader, sample


class ControlsTests(unittest.TestCase):
    def setUp(self):
        self.keyboard = FakeKeyboard()
        self.controls = ActionEmulator(self.keyboard)
        self.clock = FakeClock()

    def test_all_nine_directions_and_key_transitions(self):
        for action, keys in enumerate(DIRECTIONS):
            self.controls.apply(action)
            self.assertEqual(self.controls.held, set(keys) | {"z", "shift"})
        before = list(self.keyboard.events)
        self.controls.apply(8)
        self.assertEqual(before, self.keyboard.events)
        self.controls.release_all()
        self.assertEqual(self.controls.held, set())

    def test_focus_loss_releases_owned_keys(self):
        self.controls.apply(2)
        self.keyboard.focus = False
        start = len(self.keyboard.events)
        with self.assertRaises(InputError):
            self.controls.apply(3)
        self.assertFalse(self.controls.held)
        self.assertTrue(all(event == "up" for event, _ in self.keyboard.events[start:]))

    def test_invalid_actions(self):
        for action in (-1, 9, 1.5):
            with self.assertRaises((ValueError, TypeError)):
                self.controls.apply(action)
        self.assertFalse(self.keyboard.events)

    def test_failed_key_press_is_still_released(self):
        def fail(key):
            raise InputError("incomplete input")
        self.keyboard.key_down = fail
        with self.assertRaises(InputError):
            self.controls.apply(2)
        self.assertFalse(self.controls.held)
        self.assertTrue(self.keyboard.events)
        self.assertTrue(all(event == "up" for event, _ in self.keyboard.events))

    def test_failed_release_is_retained_for_cleanup_retry(self):
        self.controls.apply(0)
        key_up = self.keyboard.key_up
        self.keyboard.key_up = lambda key: (_ for _ in ()).throw(InputError("release failed"))
        with self.assertRaises(InputError):
            self.controls.release_all()
        self.assertEqual(self.controls.held, {"z", "shift"})
        self.keyboard.key_up = key_up
        self.controls.release_all()
        self.assertFalse(self.controls.held)

    def stepper(self, reader):
        return FrameStepper(reader, self.controls, clock=self.clock, sleep=self.clock.sleep)

    def test_byte_counter_wrap_and_actual_elapsed_frames(self):
        reader = FakeReader(sample(frame=255), sample(frame=1))
        current, frames, _ = self.stepper(reader).advance(sample(frame=254))
        self.assertEqual(current.frame, 1)
        self.assertEqual(frames, 3)

    def test_stalled_game_times_out(self):
        with self.assertRaises(FrameTimeout):
            self.stepper(FakeReader(sample())).advance(sample())

    def test_counter_reset_is_not_a_large_valid_step(self):
        with self.assertRaises(FrameTimeout):
            self.stepper(FakeReader(sample(frame=1))).advance(sample(frame=100))

    def test_slow_acquisition_is_not_mistaken_for_a_short_counter_wrap(self):
        reader = FakeReader(sample(frame=11))
        def slow_sample():
            self.clock.sleep(5)
            return sample(frame=11)
        reader.snapshot = slow_sample
        with self.assertRaises(FrameTimeout):
            self.stepper(reader).advance(sample())

    def resetter(self, reader, spell_retry=False):
        return ResetController(reader, self.controls, spell_retry, timeout=0.2,
                               clock=self.clock, sleep=self.clock.sleep, menu_settle=0)

    def test_first_reset_requires_advancing_frames(self):
        reader = FakeReader(sample(), sample(), sample(frame=11))
        self.assertEqual(self.resetter(reader).reset(first=True).frame, 11)
        with self.assertRaises(ResetError):
            self.resetter(FakeReader(sample())).reset(first=True)

    def test_failure_menu_requires_explicit_spell_mode(self):
        with self.assertRaises(ResetError):
            self.resetter(FakeReader(sample(retry=True))).reset()
        self.assertFalse(self.keyboard.events)

    def test_spell_retry_confirms_once_and_waits_for_playable_state(self):
        reader = FakeReader(sample(retry=True), sample(retry=True),
                            sample(scene=12), sample(frame=0, player_state=1), sample(frame=2))
        result = self.resetter(reader, True).reset()
        self.assertTrue(result.ready)
        self.assertEqual(self.keyboard.events, [("down", "z"), ("up", "z")])

    def test_timeout_cannot_reset_a_running_task(self):
        with self.assertRaises(ResetError):
            self.resetter(FakeReader(sample()), True).reset()

    def test_retry_timeout_does_not_repeat_menu_input(self):
        with self.assertRaises(ResetError):
            self.resetter(FakeReader(sample(retry=True)), True).reset()
        self.assertEqual(self.keyboard.events, [("down", "z"), ("up", "z")])

    def test_new_failure_menu_settles_and_recovers_a_missed_confirm(self):
        presses = []
        original_down = self.keyboard.key_down
        def down(key):
            presses.append(self.clock())
            original_down(key)
        self.keyboard.key_down = down
        reader = FakeReader(sample(retry=True))
        reader.snapshot = lambda: (sample(retry=True) if len(presses) < 2
                                   else sample(frame=11))
        controller = ResetController(reader, self.controls, True, timeout=3,
                                     clock=self.clock, sleep=self.clock.sleep)
        self.assertTrue(controller.reset().ready)
        self.assertEqual(len(presses), 2)
        self.assertGreaterEqual(presses[0], 0.35)
        self.assertGreaterEqual(presses[1] - presses[0], 0.75)
        self.assertFalse(self.controls.held)

    def test_confirm_retries_are_bounded_and_timeout_has_state(self):
        controller = ResetController(FakeReader(sample(retry=True)), self.controls,
                                     True, timeout=4, clock=self.clock, sleep=self.clock.sleep)
        with self.assertRaisesRegex(ResetError, "confirms=3.*retry=True"):
            controller.reset()
        self.assertEqual(self.keyboard.events.count(("down", "z")), 3)

    def test_no_more_confirmations_after_leaving_failure_menu(self):
        presses = []
        self.keyboard.key_down = lambda key: presses.append(key)
        reader = FakeReader(sample())
        reader.snapshot = lambda: (sample(retry=True) if not presses
                                   else sample(scene=12))
        controller = ResetController(reader, self.controls, True, timeout=2,
                                     clock=self.clock, sleep=self.clock.sleep)
        with self.assertRaisesRegex(ResetError, "left_menu=True"):
            controller.reset()
        self.assertEqual(presses, ["z"])
