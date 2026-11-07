"""Timed reachability and explicit limits of pruning and held actions."""
import unittest

from agent.forward_model import Circle, Scene
from agent.planning_cases import micro_cases
from agent.timed_planner import Path, TimedGoal, path_points, replay_path, search


class TimedPlannerTests(unittest.TestCase):
    def test_reference_micro_cases_and_returned_paths(self):
        for case in micro_cases():
            with self.subTest(case=case['name']):
                result = search(case['scene'], case['horizon'], goal=case['goal'])
                self.assertEqual(result.goal_path is not None, case['expect_arrival'])
                self.assertEqual(result.reached_frames, case['horizon'])
                if result.goal_path:
                    self.assertTrue(replay_path(case['scene'], result.goal_path))
                    self.assertTrue(case['goal'].reached(result.goal_path.point, len(result.goal_path.actions) * 4))

    def test_detour_crosses_only_above_wall(self):
        case = next(c for c in micro_cases() if c['name'] == 'detour_before_approach')
        result = search(case['scene'], case['horizon'], goal=case['goal'])
        points = path_points(case['scene'], result.goal_path)
        crossings = [(a, b) for a, b in zip(points, points[1:]) if a[0] < 20 < b[0]]
        self.assertTrue(crossings)
        self.assertTrue(all(a[1] < 10 for a, _ in crossings))
        self.assertGreaterEqual(len(result.goal_path.actions) * 4, 28)

    def test_beam_states_are_reference_reachable_and_pruning_is_reported(self):
        scene = Scene((24, 24), bounds=(0, 48, 0, 48))
        reference = search(scene, 24)
        beam = search(scene, 24, beam_width=5)
        self.assertGreater(beam.pruned_states, 0)
        self.assertTrue(all(len(layer) <= 5 for layer in beam.layers))
        for full, limited in zip(reference.layers, beam.layers):
            self.assertTrue(limited.keys() <= full.keys())
        for path in beam.frontier:
            self.assertTrue(replay_path(scene, path))

    def test_goal_window_and_no_hard_following_reward(self):
        scene = Scene((8, 20), bounds=(0, 40, 0, 40))
        goal = TimedGoal(8, 20, 8, 8, 1)
        result = search(scene, 16, goal=goal)
        self.assertEqual(len(result.goal_path.actions) * 4, 8)
        # Arrival queries operate at held-action boundaries, not arbitrary times.
        self.assertIsNone(search(scene, 16, goal=TimedGoal(8, 20, 1, 3, 1)).goal_path)

    def test_blocked_seed_is_not_a_zero_survival_guarantee(self):
        scene = Scene((20, 20), circles=(Circle(20, 20, 0, 0, 4),), bounds=(0, 40, 0, 40))
        result = search(scene, 16)
        self.assertFalse(result.layers[0])
        self.assertEqual(result.frontier, [])
        self.assertFalse(replay_path(scene, Path(scene.player)))

    def test_invalid_search_and_goal_settings(self):
        scene = Scene((8, 16))
        for horizon in (0, 3, 260):
            with self.assertRaises(ValueError):
                search(scene, horizon)
        for width in (0, 4, True):
            with self.assertRaises(ValueError):
                search(scene, beam_width=width)
        with self.assertRaises(ValueError):
            TimedGoal(0, 0, 4, 3, 1)
