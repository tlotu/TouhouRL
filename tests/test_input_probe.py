"""Input timing and probe scheduling without native process access."""
import unittest

from agent.corrections import CorrectionPolicy
from env.action_emulator import ActionEmulator, InputError
from env.session import GameSession
from scripts.probe_input import PHASES, PROBE_FRAMES, ProbePolicy
from scripts.diagnostics import run_episodes
from tests.fakes import FakeClock, FakeKeyboard, FakeReader, sample


class InputProbeTests(unittest.TestCase):
    def test_timing_separates_observation_age_send_and_snapshot(self):
        clock = FakeClock()

        class Reader(FakeReader):
            def snapshot(self):
                clock.sleep(0.003)
                return super().snapshot()

        class Controls(ActionEmulator):
            def apply(self, action):
                super().apply(action)
                clock.sleep(0.002)

        controls = Controls(FakeKeyboard())
        session = GameSession(Reader(sample(frame=12), sample(frame=14)), controls,
                              clock=clock, sleep=clock.sleep)
        with session:
            session.active = True
            session.sample = sample(frame=10)
            session.sample_received_at = session.last_observation_time = clock()
            clock.sleep(0.010)
            _, _, _, _, info = session.step(3)
            timing = info["timing"]
            self.assertAlmostEqual(timing["observation_age_seconds"], 0.010)
            self.assertAlmostEqual(timing["input_send_seconds"], 0.002)
            self.assertAlmostEqual(timing["snapshot_seconds"], 0.003)
            self.assertAlmostEqual(timing["post_input_wait_seconds"], 0.003)
            self.assertEqual(timing["snapshot_calls"], 1)
            self.assertIsNone(timing["previous_action"])
            self.assertEqual(timing["applied_action"], 3)
            clock.sleep(0.007)
            timing = session.step(7)[-1]["timing"]
            self.assertAlmostEqual(timing["observation_age_seconds"], 0.007)
            self.assertEqual(timing["previous_action"], 3)
        self.assertFalse(controls.held)

    def test_probe_uses_elapsed_frames_and_contains_releases(self):
        class Session:
            elapsed_frames = 0
        session = Session()
        choose = ProbePolicy(session)
        boundary = 0
        for action, frames in PHASES:
            for elapsed in (boundary, boundary + frames - 1):
                session.elapsed_frames = elapsed
                self.assertEqual(choose({}), action)
            boundary += frames
        self.assertEqual(boundary, PROBE_FRAMES)
        session.elapsed_frames = PROBE_FRAMES + 4
        self.assertEqual(choose({}), 0)

    def test_student_and_teacher_inference_are_separate(self):
        clock = FakeClock()
        def student(obs):
            clock.sleep(0.004)
            return 3
        def teacher(obs):
            clock.sleep(0.009)
            return 7
        choose = CorrectionPolicy(student, 0, teacher=teacher, clock=clock)
        action = choose({})
        row = choose.annotate({"event": "transition", "action": action})
        self.assertEqual(action, 3)
        self.assertAlmostEqual(row["student_inference_seconds"], 0.004)
        self.assertAlmostEqual(row["teacher_inference_seconds"], 0.009)

    def test_focus_loss_still_releases_probe_keys(self):
        clock = FakeClock()
        keyboard = FakeKeyboard()
        controls = ActionEmulator(keyboard)
        with GameSession(FakeReader(sample(frame=12)), controls, repeat=1,
                         clock=clock, sleep=clock.sleep) as session:
            session.active = True
            session.sample = sample(frame=10)
            session.sample_received_at = session.last_observation_time = clock()
            session.step(3)
            keyboard.focus = False
            with self.assertRaises(InputError):
                session.step(7)
            self.assertFalse(controls.held)

    def test_probe_stops_at_budget_and_releases_keys(self):
        clock = FakeClock()
        controls = ActionEmulator(FakeKeyboard())
        reader = FakeReader(sample(), sample(frame=11),
                            *(sample(frame=12+i) for i in range(PROBE_FRAMES)))
        rows = []
        with GameSession(reader, controls, repeat=1, max_frames=PROBE_FRAMES,
                         clock=clock, sleep=clock.sleep) as session:
            run_episodes(session, 1, rows.append, policy="input_probe", trace=True,
                         action_fn=ProbePolicy(session))
            self.assertFalse(session.active)
            self.assertFalse(controls.held)
        trace = [r for r in rows if r["event"] == "transition"]
        self.assertEqual(len(trace), PROBE_FRAMES)
        self.assertEqual(trace[-1]["info"]["reason"], "time_limit")
        self.assertTrue(trace[-1]["truncated"])
        self.assertEqual(rows[-1]["successful_episodes"], 0)
        self.assertEqual(sum(r["event"] == "reset" for r in rows), 1)
