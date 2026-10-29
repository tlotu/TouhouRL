import importlib.util
import unittest

from env.action_emulator import ActionEmulator
from env.frame_stepper import FrameTimeout
from env.memory_reader import Bullet, MemoryReadError
from env.reset_controller import ResetError
from env.session import GameSession, observation
from tests.fakes import FakeClock, FakeKeyboard, FakeReader, sample


class SessionTests(unittest.TestCase):
    def make_session(self, *following, **kwargs):
        self.clock = FakeClock()
        self.keyboard = FakeKeyboard()
        self.reader = FakeReader(sample(), sample(frame=11), *following)
        self.controls = ActionEmulator(self.keyboard)
        return GameSession(self.reader, self.controls, clock=self.clock,
                           sleep=self.clock.sleep, **kwargs)

    def test_real_observation_and_time_reward(self):
        session = self.make_session(sample(frame=13, lives=0))
        obs, _ = session.reset()
        self.assertEqual(len(obs["bullets"]), 64)
        _, reward, terminated, truncated, info = session.step(2)
        self.assertAlmostEqual(reward, 2 / 60)
        self.assertFalse(terminated or truncated)
        self.assertEqual(info["delta_frames"], 2)
        session.close()
        self.assertTrue(self.reader.closed)
        self.assertFalse(self.controls.held)

    def test_confirmed_death_terminates_and_releases_keys(self):
        session = self.make_session(sample(frame=12, deaths=1, dying=True))
        session.reset()
        _, reward, terminated, truncated, info = session.step(3)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual((reward, info["reason"]), (-1.0, "death"))
        self.assertFalse(self.controls.held)
        with self.assertRaises(RuntimeError):
            session.step(0)

    def test_death_window_alone_does_not_confirm_death(self):
        session = self.make_session(sample(frame=13, dying=True, player_state=2))
        session.reset()
        self.assertFalse(session.step(0)[2])

    def test_external_limit_truncates(self):
        session = self.make_session(sample(frame=13), max_frames=2)
        session.reset()
        _, _, terminated, truncated, info = session.step(0)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["reason"], "time_limit")
        self.assertFalse(self.controls.held)
        with self.assertRaises(ResetError):
            session.reset()

    def test_read_failure_releases_keys_and_invalidates_episode(self):
        session = self.make_session(MemoryReadError("disconnected"))
        session.reset()
        with self.assertRaises(MemoryReadError):
            session.step(2)
        self.assertFalse(self.controls.held)
        self.assertFalse(session.active)

    def test_long_gap_does_not_become_a_valid_transition(self):
        session = self.make_session(sample(frame=13))
        session.reset()
        self.clock.sleep(1)
        with self.assertRaises(FrameTimeout):
            session.step(0)
        self.assertFalse(self.controls.held)

    def test_complete_failure_and_retry_cycle(self):
        session = self.make_session(sample(frame=13, deaths=1),
                                    sample(frame=14, deaths=1, retry=True),
                                    sample(frame=0, scene=12), sample(frame=1),
                                    sample(frame=3, deaths=1, retry=True),
                                    spell_retry=True)
        session.reset()
        self.assertTrue(session.step(1)[2])
        _, info = session.reset()
        self.assertEqual(info["deaths"], 0)
        self.assertEqual(info["elapsed_frames"], 0)
        self.assertTrue(session.step(1)[2])
        session.close()

    def test_scene_exit_is_not_reported_as_success(self):
        session = self.make_session(sample(frame=12, scene=6, wanted_scene=6))
        session.reset()
        _, reward, terminated, _, info = session.step(0)
        self.assertTrue(terminated)
        self.assertEqual(reward, 0)
        self.assertEqual(info["reason"], "scene_exit")

    def test_steps_require_reset_and_close_is_idempotent(self):
        session = self.make_session(sample())
        with self.assertRaises(RuntimeError):
            session.step(0)
        session.close()
        session.close()
        with self.assertRaises(RuntimeError):
            session.reset()

    def test_observation_sorts_pads_and_clips(self):
        far = Bullet(0, -1000, 2000, 50, -50, 2, 1)
        near = Bullet(1, 193, 400, 1, 0, 2, 1)
        obs = observation(sample(bullets=(far, near)), capacity=3)
        self.assertAlmostEqual(obs["bullets"][0][0], 1 / 384)
        self.assertEqual(obs["bullets"][1][:4], [-1, 1, 1, -1])
        self.assertEqual(obs["bullets"][2], [0.0] * 7)


@unittest.skipUnless(importlib.util.find_spec("gymnasium"), "Install optional Gymnasium dependencies")
class GymAdapterTests(unittest.TestCase):
    def test_observations_match_declared_space(self):
        from env.touhou_env import TouhouEnv
        clock = FakeClock()
        reader = FakeReader(sample(), sample(frame=11), sample(frame=13))
        session = GameSession(reader, ActionEmulator(FakeKeyboard()), clock=clock, sleep=clock.sleep)
        env = TouhouEnv(session)
        obs, _ = env.reset(seed=7)
        self.assertTrue(env.observation_space.contains(obs))
        obs, reward, terminated, truncated, _ = env.step(2)
        self.assertTrue(env.observation_space.contains(obs))
        self.assertIsInstance(reward, float)
        self.assertFalse(terminated or truncated)
        env.close()
