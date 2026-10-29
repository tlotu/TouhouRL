import unittest

from env.action_emulator import ActionEmulator
from env.memory_reader import ResultMenu
from env.session import GameSession
from tests.fakes import FakeClock, FakeKeyboard, FakeReader, sample


class TrainingBoundaryTests(unittest.TestCase):
    def session(self, menu):
        clock = FakeClock()
        return GameSession(FakeReader(sample(result_menu=menu)), ActionEmulator(FakeKeyboard()),
                           clock=clock, sleep=clock.sleep)

    def test_update_boundary_requires_recognized_settled_result(self):
        with self.session(ResultMenu(True, 0, 1, 1)) as session:
            self.assertEqual(session.verify_training_boundary(), "success")
        for menu in (None, ResultMenu(False), ResultMenu(True, 0, 0, 1)):
            with self.subTest(menu=menu), self.session(menu) as session:
                with self.assertRaises(RuntimeError):
                    session.verify_training_boundary()

    def test_active_episode_cannot_be_used_as_update_boundary(self):
        with self.session(ResultMenu(True, 0, 1, 1)) as session:
            session.active = True
            with self.assertRaises(RuntimeError):
                session.verify_training_boundary()
