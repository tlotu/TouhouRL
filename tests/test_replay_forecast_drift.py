"""Checks for replay path reconstruction and censored forecast comparisons."""
from types import SimpleNamespace
import unittest

import numpy as np

from agent.direction_buffer import DirectionBuffer
from agent.forward_model import Circle, Scene
from scripts.audit_replay_forecast_drift import audit, reconstruct


class ForecastDriftTests(unittest.TestCase):
    def test_delayed_switch_preserves_per_frame_path(self):
        episode = SimpleNamespace(samples={0: SimpleNamespace(scene=Scene((20, 20)))})
        history = [{'frame': 0, 'request': 3, 'start': [20, 20], 'end': [26, 20]}]
        edges = reconstruct(episode, history, DirectionBuffer(), 1, 4)
        np.testing.assert_allclose(edges[0][1], [20, 20])
        np.testing.assert_allclose(edges[1][1], [22, 20])
        np.testing.assert_allclose(edges[3][1], [26, 20])
        with self.assertRaises(AssertionError):
            reconstruct(episode, history, DirectionBuffer(), 0, 4)

    def test_newly_observed_contact_and_censoring(self):
        clear = Scene((20, 20))
        hazard = Scene((20, 20), circles=(Circle(20, 20, 0, 0, 2),))
        episode = SimpleNamespace(samples={0: SimpleNamespace(scene=clear),
                                           2: SimpleNamespace(scene=hazard)})
        edges = {frame: (np.array([20., 20.]), np.array([20., 20.])) for frame in range(4)}
        result = audit(episode, edges, 0, 8)
        self.assertTrue(result['censored'])
        self.assertEqual(result['checked_frames'], 4)
        self.assertIsNone(result['first_frozen_contact'])
        self.assertEqual(result['first_refreshed_contact']['frame'], 2)

    def test_consistent_stationary_hazard_has_no_drift(self):
        scene = Scene((20, 20), circles=(Circle(30, 20, 0, 0, 2),))
        episode = SimpleNamespace(samples={f: SimpleNamespace(scene=scene) for f in (0, 2)})
        edges = {frame: (np.array([20., 20.]), np.array([20., 20.])) for frame in range(4)}
        result = audit(episode, edges, 0, 4)
        self.assertFalse(result['censored'])
        self.assertIsNone(result['first_clearance_drift_over_2px'])
        self.assertEqual(result['frozen_min_clearance'], result['refreshed_min_clearance'])


if __name__ == '__main__':
    unittest.main()
