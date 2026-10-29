import unittest

from env.action_emulator import ActionEmulator
from env.memory_reader import Bullet
from env.session import GameSession
from scripts.diagnostics import run_episodes
from tests.fakes import FakeClock, FakeKeyboard, FakeReader, sample


class QuietStopTests(unittest.TestCase):
    def test_quiet_result_is_unresolved_even_with_zero_deaths(self):
        bullet = Bullet(0, 192, 100, 0, 1, 3, 1)
        rows = []
        with self.session(sample(frame=13, bullets=(bullet,)),
                          sample(frame=15), quiet_limit=2) as session:
            run_episodes(session, 1, rows.append, policy="stay")
        self.assertEqual(rows[-1]["completed_episodes"], 1)
        self.assertEqual(rows[-1]["death_or_retry_episodes"], 0)
        self.assertEqual(rows[-1]["unresolved_episodes"], 1)
        self.assertIsNone(rows[-1]["success_rate"])

    def session(self, *samples, **options):
        clock = FakeClock()
        return GameSession(FakeReader(sample(), sample(frame=11), *samples),
                           ActionEmulator(FakeKeyboard()), clock=clock,
                           sleep=clock.sleep, **options)

    def test_initial_quiet_does_not_count_as_completion(self):
        with self.session(sample(frame=13), sample(frame=15), quiet_limit=2) as session:
            session.reset()
            self.assertFalse(session.step(0)[3])
            self.assertFalse(session.step(0)[3])

    def test_post_combat_quiet_truncates_without_claiming_success(self):
        bullet = Bullet(0, 192, 100, 0, 1, 3, 1)
        with self.session(sample(frame=13, bullets=(bullet,)), sample(frame=15),
                          sample(frame=17), quiet_limit=4) as session:
            session.reset()
            session.step(0)
            self.assertFalse(session.step(0)[3])
            _, _, terminated, truncated, info = session.step(0)
            self.assertFalse(terminated)
            self.assertTrue(truncated)
            self.assertEqual(info["reason"], "quiet_after_combat")
            self.assertFalse(session.controls.held)

    def test_new_bullets_reset_quiet_counter(self):
        bullet = Bullet(0, 192, 100, 0, 1, 3, 1)
        with self.session(sample(frame=13, bullets=(bullet,)), sample(frame=15),
                          sample(frame=17, bullets=(bullet,)), sample(frame=19),
                          quiet_limit=4) as session:
            session.reset()
            for _ in range(4):
                self.assertFalse(session.step(0)[3])
            self.assertEqual(session.quiet_frames, 2)

    def test_death_takes_priority_over_quiet_stop(self):
        bullet = Bullet(0, 192, 100, 0, 1, 3, 1)
        with self.session(sample(frame=13, bullets=(bullet,)),
                          sample(frame=15, retry=True, deaths=1), quiet_limit=2) as session:
            session.reset()
            session.step(0)
            _, _, terminated, truncated, info = session.step(0)
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertEqual(info["reason"], "death")
