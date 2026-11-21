"""Regression checks for robust, budget-limited replay search."""
import unittest
from dataclasses import replace

from agent.direction_buffer import DirectionBuffer
from agent.forward_model import Circle, Scene
from agent.replay_hrl import PADDING
from agent.swept_collision import SweptCollision
from scripts.evaluate_replay_budget import small_search


class BudgetSearchTests(unittest.TestCase):
    def test_second_proposal_survives_when_only_one_delay_hits_first(self):
        scene = Scene((20., 20.), circles=(Circle(29., 20., 0., 0., .1),),
                      bounds=(0., 80., 0., 80.), player_radius=.1)
        belief = frozenset((DirectionBuffer(),))
        model = SweptCollision(replace(scene, player_radius=scene.player_radius + PADDING))
        self.assertFalse(model.edges([(26., 20.)], [(28., 20.)], 3, 1)[0][0])
        for depth in (2, 3):
            only, checked = small_search(scene, belief, [3], depth)
            self.assertEqual((only, checked), ((), 1))
            choices, checked = small_search(scene, belief, [3, 0], depth)
            self.assertEqual(choices[0].route.actions[0], 0)
            self.assertEqual(len(choices[0].route.actions), depth)
            self.assertEqual(checked, 2 + 9 * (depth - 1))
