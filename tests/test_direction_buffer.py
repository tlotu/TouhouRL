"""Verify buffered motion and search around transition-specific obstacles."""
import unittest

from agent.direction_buffer import DirectionBuffer
from agent.forward_model import Circle, Scene
from agent.input_switch import check_switch_prefix
from agent.short_planner import plan
from agent.timed_planner import TimedGoal
from scripts.probe_short_planning import _verify


class DirectionBufferTests(unittest.TestCase):
    def test_reversal_tail_is_not_replaced_by_the_next_request(self):
        state = DirectionBuffer(8, 8)
        moves, tail = state.advance(4, 4, 1)
        self.assertEqual(moves, (8, 1, 0, 5))
        self.assertEqual(tail, DirectionBuffer(4, 5, (4,)))
        self.assertEqual(tail.advance(3, 4, 1), ((4, 3, 3, 3), DirectionBuffer(3, 3)))

    def test_search_chooses_a_different_safe_sequence_before_dispatch(self):
        scene = Scene((100., 100.), circles=(Circle(101.7, 103., 0, 0, .1),), player_radius=3.25)
        goal = TimedGoal(112, 88, 0, 16, 2.5)
        nominal = plan(scene, goal).selected
        self.assertEqual(nominal.actions[0], 2)
        self.assertEqual(check_switch_prefix(scene, 0, 2, lambda: False)['status'], 'blocked')
        buffered = plan(scene, goal, input_buffer=DirectionBuffer()).selected
        self.assertEqual(buffered.actions[0], 1)
        self.assertEqual(buffered.arrival, 12)
        _verify(scene, buffered)
