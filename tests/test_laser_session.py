"""Coherent laser snapshot wrappers and fail-closed session cleanup."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from env.laser_probe import BUILD
from env.laser_session import LaserReader, LaserSession, LaserSnapshot, LaserTouhouEnv
from tests.fakes import FakeKeyboard, sample
from env.action_emulator import ActionEmulator
from tests.test_laser_geometry import laser


class LaserSessionTests(unittest.TestCase):
    def test_reader_wraps_same_capture_and_forwards_close(self):
        reader=SimpleNamespace(addresses={'BM_RVA':0x3EC2A0},close=lambda:None)
        snap=sample()
        with patch('env.laser_session.capture',return_value=(snap,{'lasers':[laser()]})) as capture:
            wrapped=LaserReader(reader,BUILD).snapshot()
        capture.assert_called_once_with(reader)
        self.assertEqual(wrapped.frame,snap.frame)
        self.assertEqual(wrapped.lasers[0]['phase_raw'],1)

    def test_session_adds_exact_snapshot_and_space(self):
        controls=ActionEmulator(FakeKeyboard())
        session=object.__new__(LaserSession);session.controls=controls;session.active=True;session.capacity=4
        session.sample=LaserSnapshot(sample(),[laser()])
        obs={'player':[0.,0.,0.,0.], 'bullets':[[0.]*7]*4,'enemies':[[0.]*6]*16}
        with patch('env.session.GameSession.reset',return_value=(obs,{})):
            result,info=session.reset()
        self.assertEqual(info['laser_count'],1)
        self.assertEqual(result['lasers'].shape,(64,12))
        env=LaserTouhouEnv(session)
        self.assertTrue(env.observation_space.contains(env._arrays(result)))

    def test_invalid_laser_ends_episode_and_releases_keys(self):
        keyboard=FakeKeyboard();controls=ActionEmulator(keyboard);controls.apply(3)
        session=object.__new__(LaserSession);session.controls=controls;session.active=True
        bad=laser();bad['speed']=1
        session.sample=LaserSnapshot(sample(),[bad])
        with patch('env.session.GameSession.step',return_value=({},0.,False,False,{})):
            with self.assertRaises(ValueError):session.step(3)
        self.assertFalse(session.active)
        self.assertFalse(controls.held)
