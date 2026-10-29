import unittest

from agent.baselines import make_policy, reactive
from env.action_emulator import ActionEmulator, DIRECTIONS
from env.memory_reader import Bullet
from env.session import GameSession, observation
from scripts.diagnostics import run_episodes
from tests.fakes import FakeClock, FakeKeyboard, FakeReader, sample


class BaselineTests(unittest.TestCase):
    def test_random_policy_seed_is_repeatable(self):
        a, b = make_policy("random", 42), make_policy("random", 42)
        self.assertEqual([a({}) for _ in range(20)], [b({}) for _ in range(20)])

    def test_stay_policy(self):
        self.assertEqual(make_policy("stay")({}), 0)

    def test_reactive_stays_when_no_threat_exists(self):
        self.assertEqual(reactive(observation(sample(y=224))), 0)

    def test_reactive_avoids_incoming_bullet(self):
        obs = observation(sample(y=224, bullets=(Bullet(0, 192, 200, 0, 2, 3, 1),)))
        self.assertNotEqual(reactive(obs), 0)

    def test_reactive_does_not_move_through_wall(self):
        obs = observation(sample(x=8, y=224, bullets=(Bullet(0, 8, 200, 0, 2, 3, 1),)))
        self.assertNotIn("left", DIRECTIONS[reactive(obs)])

    def test_padding_does_not_create_a_fake_threat(self):
        obs = observation(sample(y=224))
        obs["bullets"][0] = [0, 0, 0, 0, 1, 1, 0]
        self.assertEqual(reactive(obs), 0)

    def test_trace_aligns_action_with_before_and_after_observations(self):
        clock = FakeClock()
        bullet = Bullet(115, 190, 390, 0, 1, 3, 1, radius_estimated=True)
        reader = FakeReader(sample(), sample(frame=11),
                            sample(frame=13, x=194, bullets=(bullet,)),
                            sample(frame=15, deaths=1, retry=True))
        rows = []
        with GameSession(reader, ActionEmulator(FakeKeyboard()),
                         clock=clock, sleep=clock.sleep) as session:
            run_episodes(session, 1, rows.append, policy="stay", trace=True)
        trace = [r for r in rows if r["event"] == "transition"]
        self.assertEqual(len(trace), 2)
        self.assertEqual(trace[0]["action"], 0)
        self.assertNotEqual(trace[0]["observation"]["player"], trace[0]["next_observation"]["player"])
        self.assertEqual(trace[0]["next_observation"], trace[1]["observation"])
        self.assertTrue(trace[-1]["terminated"])
        self.assertEqual(rows[-1]["transitions"], 2)
        self.assertEqual(rows[-1]["radius_fallback_steps"], 1)
