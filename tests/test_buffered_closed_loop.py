"""Check that simulated branches resume policy control from their own state."""
import unittest

import numpy as np

from agent.direction_buffer import DirectionBuffer
from agent.forward_model import ForwardModel, Scene
from scripts.replay_buffered_closed_loop import rebase, rollout


class ClosedLoopReplayTests(unittest.TestCase):
    def test_each_branch_recomputes_relative_geometry_after_the_first_hold(self):
        obs = {'player': np.zeros(4, dtype=np.float32),
               'bullets': np.zeros((4, 7), dtype=np.float32),
               'enemies': np.zeros((4, 6), dtype=np.float32),
               'lasers': np.zeros((64, 12), dtype=np.float32)}
        obs['enemies'][0] = [.25, -.25, 1., 1., 1., 1.]
        moved = rebase(obs, (200., 224.))
        self.assertAlmostEqual(moved['enemies'][0, 0] * 384 + 200., 288., places=5)
        np.testing.assert_array_equal(moved['enemies'][1:], obs['enemies'][1:])
        self.assertIs(moved['lasers'], obs['lasers'])
        self.assertEqual(obs['player'][0], 0.)

        rows = [{'action': 0, 'info': {'elapsed_frames': frame + 2, 'delta_frames': 2, 'dropped_bullets': 0}}
                for frame in range(0, 8, 2)]
        model = ForwardModel(Scene((192., 224.)))

        def logits(state):
            scores = np.zeros(9)
            scores[1 if state['enemies'][0, 0] == .25 else 5] = 1.
            return scores

        traces = [rollout(rows, [obs] * len(rows), lambda _: model, 1, first, DirectionBuffer(),
                          0, 6, logits, 1.) for first in (0, 3)]
        self.assertEqual([[step['action'] for step in result['trajectory']] for result in traces],
                         [[0, 0, 1], [3, 3, 5]])
        self.assertEqual([result['trajectory'][-1]['forced'] for result in traces], [False, False])
