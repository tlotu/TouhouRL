"""Regressions for swept culling, time-aware scoring and action retention."""
from dataclasses import replace
import math
import unittest
from unittest.mock import patch

import numpy as np

from agent.direction_buffer import DirectionBuffer
from agent.forward_model import Circle, ForwardModel, Scene, TimedLaser
from agent.short_planner import HOLD, LONG_HORIZON, SHORT_HORIZON, SearchBudget, exit_hint, plan, route_points
from agent.swept_collision import SweptCollision
from agent.short_planner import forced_beams
from agent.timed_planner import TimedGoal
from scripts.probe_short_planning import _rollout, benchmark_cases


class ForcedBeamTests(unittest.TestCase):
    def test_blocked_directions_do_not_consume_other_beams(self):
        scene = Scene((8., 30.), lasers=(TimedLaser(20., 0., 20., 60., 2., 6., 30.),))
        state = DirectionBuffer()
        actions = [1, 3, 7]
        for delay in (0, 1):
            grouped, checked = forced_beams(SweptCollision(scene), state, actions, delay, 9)
            expected = {}
            edges = 0
            for action in actions:
                result = plan(scene, width=9, ttc_weight=0., input_buffer=state,
                              switch_delay=delay, first_action=action)
                edges += result.checked_edges
                full = [r for r in result.candidates if len(r.motion) == SHORT_HORIZON]
                if full:
                    expected[action] = full
            self.assertIn(1, expected)
            self.assertNotIn(7, expected)
            self.assertEqual((grouped, checked), (expected, edges))


class SweptCollisionTests(unittest.TestCase):
    def test_far_fast_circle_and_spanning_laser_survive_culling(self):
        circle = Circle(100., 20., -20., 0., 2.)
        laser = TimedLaser(24., -200., 24., 200., 1., 2., 12.)
        scene = Scene((8., 20.), (circle,), (laser,), (0, 48, 0, 48))
        starts, ends = [[8., 20.], [20., 40.]], [[8., 20.], [28., 40.]]
        valid, gaps = SweptCollision(scene).edges(starts, ends, 0, 8)
        self.assertEqual(valid.tolist(), [False, False])
        np.testing.assert_allclose(gaps, [-3.25, -1.75])

    def test_full_scan_agreement_includes_near_misses_and_activation_edges(self):
        rng = np.random.default_rng(17)
        starts = rng.uniform(0, 80, (140, 2))
        ends = np.clip(starts + rng.uniform(-8, 8, (140, 2)), 0, 80)
        circles = tuple(Circle(*position, *velocity, radius)
                        for position, velocity, radius in zip(rng.uniform(-30, 110, (30, 2)),
                                                              rng.uniform(-12, 12, (30, 2)),
                                                              rng.uniform(0, 8, 30)))
        lasers = (TimedLaser(-80, 10, 180, 55, 3, 3, 7), TimedLaser(40, 50, 40, 50, 4, 0, 11))
        scene = Scene((40, 40), circles, lasers, (0, 80, 0, 80))
        reference, swept = ForwardModel(scene), SweptCollision(scene)
        for frame, duration in [(0, 4), (3, 4), (7, 4), (12, 4), (2, 0)]:
            finish = ends if duration else starts
            expected, raw = reference.edges(starts, finish, frame, duration)
            actual, gaps = swept.edges(starts, finish, frame, duration)
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_allclose(gaps, np.minimum(raw, 24.), rtol=0, atol=1e-9)
        self.assertLess(swept.counts.checked_pairs, swept.counts.potential_pairs)

    def test_boundary_rejects_before_hazard_pairs_and_keeps_edge_contact(self):
        scene = Scene((8, 16), (Circle(80, 80, 0, 0, 2),))
        model = SweptCollision(scene)
        valid, _ = model.edges([[8, 16], [8, 16]], [[7, 16], [8, 24]], 0, 4)
        self.assertEqual(valid.tolist(), [False, True])
        self.assertEqual(model.counts.boundary_pruned, 1)
        self.assertEqual(model.counts.potential_pairs, 1)

    def test_stationary_ttc_uses_first_contact_and_leaf_time(self):
        circle = Circle(50, 20, -10, 0, 2)
        scene = Scene((20, 20), (circle,), bounds=(0, 60, 0, 60))
        self.assertAlmostEqual(SweptCollision(scene).stationary_ttc([[20, 20]], 1, 12)[0], 1.675)
        scene = replace(scene, lasers=(TimedLaser(20, 10, 20, 30, 2, 1.5, 8),))
        times = SweptCollision(scene).stationary_ttc([[20, 20], [20, 0], [40, 20]], 1, 12)
        np.testing.assert_allclose(times, [.5, 12., 0.])
        self.assertEqual(SweptCollision(scene).stationary_ttc([[20, 20]], 8, 12)[0], 12.)


class ShortPlannerTests(unittest.TestCase):
    def test_reserved_leaf_time_scores_a_truncated_layer(self):
        scene = Scene((40, 40), bounds=(0, 80, 0, 80))
        now = 0.
        original = SweptCollision.edges

        def edges(model, starts, ends, frame, duration):
            nonlocal now
            result = original(model, starts, ends, frame, duration)
            if frame == 7:
                now = .0092
            return result

        with patch.object(SweptCollision, 'edges', edges):
            result = plan(scene, width=9, input_buffer=DirectionBuffer(), clock=lambda: now,
                          deadline=.010, leaf_reserve_ms=1.)
        self.assertEqual(result.stop_reason, 'time_limit')
        self.assertEqual(len(result.selected.motion), 8)
        self.assertTrue(result.ttc_evaluated)
        self.assertEqual(result.selected.ttc, 60.)

    def test_absolute_deadline_preserves_completed_batch_in_partial_layer(self):
        scene = Scene((40, 40), bounds=(0, 80, 0, 80))
        now = 0.
        original = SweptCollision.edges

        def edges(model, starts, ends, frame, duration):
            nonlocal now
            result = original(model, starts, ends, frame, duration)
            if frame == 7:  # Last motion frame in the first batch of layer two.
                now = 1.
            return result

        with patch.object(SweptCollision, 'edges', edges):
            result = plan(scene, width=9, input_buffer=DirectionBuffer(), clock=lambda: now, deadline=1.)
        self.assertEqual(result.stop_reason, 'time_limit')
        self.assertEqual(result.checked_edges, 36)
        self.assertEqual(len(result.selected.motion), 8)
        self.assertFalse(result.ttc_evaluated)
        points = route_points(scene, result.selected)
        reference = ForwardModel(scene)
        self.assertTrue(all(reference.edges(points[i:i+1], points[i+1:i+2], i, 1)[0][0]
                            for i in range(8)))

    def test_edge_limit_returns_only_a_checked_prefix(self):
        scene = Scene((40, 40), (Circle(46, 40, 0, 0, 1),), bounds=(0, 80, 0, 80))
        result = plan(scene, previous_action=3, budget=SearchBudget(9))
        self.assertEqual(result.stop_reason, 'edge_limit')
        self.assertEqual(result.checked_edges, 9)
        self.assertEqual(len(result.selected.actions), 1)
        self.assertNotIn(3, result.first_actions_at(HOLD))
        points = route_points(scene, result.selected)
        self.assertTrue(ForwardModel(scene).edges(points[:1], points[1:], 0, HOLD)[0][0])

    def test_time_limit_reports_unscored_ttc_and_retains_verified_layer(self):
        scene = Scene((40, 40), bounds=(0, 80, 0, 80))
        ticks = iter((0., 0., .002, .002))
        result = plan(scene, budget=SearchBudget(378, 1.), clock=lambda: next(ticks))
        self.assertEqual(result.stop_reason, 'time_limit')
        self.assertEqual(result.checked_edges, 9)
        self.assertEqual(len(result.selected.actions), 1)
        self.assertFalse(result.ttc_evaluated)
        ticks = iter((0., .002))
        result = plan(scene, budget=SearchBudget(378, 1.), clock=lambda: next(ticks))
        self.assertEqual(result.stop_reason, 'time_limit')
        self.assertEqual(result.checked_edges, 0)
        self.assertIsNone(result.selected)

    def test_unpruned_reference_distinguishes_reachable_actions_from_blocked_ones(self):
        case = next(c for c in benchmark_cases() if c['name'] == 'required_reversal')
        result = plan(case['scene'], width=None, ttc_weight=0.)
        self.assertEqual(result.first_actions_at(SHORT_HORIZON), (3,))
        result = plan(Scene((40, 40), bounds=(0, 80, 0, 80)), cell_size=8.)
        self.assertEqual(result.first_actions_at(SHORT_HORIZON), tuple(range(9)))

    def test_required_reversal_is_kept_with_angular_cost(self):
        case = next(c for c in benchmark_cases() if c['name'] == 'required_reversal')
        route = plan(case['scene']).selected
        self.assertEqual(route.actions[:2], (3, 7))
        self.assertAlmostEqual(route.turn, math.pi)
        self.assertTrue(_rollout(case, {})['pass'])

    def test_trigger_uses_clearance_time_and_ignores_unrelated_beam(self):
        scene = Scene((8, 20), lasers=(TimedLaser(20, 0, 20, 40, 2, 24, 80),), bounds=(0, 40, 0, 40))
        goal = TimedGoal(32, 20, 0, 24, 1)
        hint = exit_hint(scene, goal)
        self.assertEqual((hint.closing_frames, hint.clear_frames, hint.slack_frames), (24, 8, 12))
        self.assertEqual(plan(scene, goal, adaptive=True).horizon, LONG_HORIZON)
        later = replace(scene, lasers=(replace(scene.lasers[0], on=25),))
        self.assertEqual(plan(later, goal, adaptive=True).horizon, SHORT_HORIZON)
        unrelated = replace(scene, lasers=(TimedLaser(0, 4, 40, 4, 2, 1, 40),))
        self.assertIsNone(exit_hint(unrelated, goal))

    def test_ttc_evacuation_before_hazard_enters_short_horizon(self):
        case = next(c for c in benchmark_cases() if c['name'] == 'evacuate_before_short_horizon_sees_laser')
        self.assertFalse(_rollout(case, {'ttc_weight': 0.})['pass'])
        self.assertTrue(_rollout(case, {})['pass'])

    def test_smoothing_selects_valid_alternative_and_drops_blocked_old_action(self):
        scene = Scene((40, 40), bounds=(0, 120, 0, 100))
        goal = TimedGoal(63.8, 20, 0, 32, .1)
        result = plan(scene, goal, previous_action=3)
        self.assertEqual(result.proposed.actions[0], 2)
        self.assertTrue(result.smoothed)
        self.assertEqual(result.selected.actions[0], 3)
        points = route_points(scene, result.selected)
        reference = ForwardModel(scene)
        for i, (start, end) in enumerate(zip(points, points[1:])):
            self.assertTrue(reference.edges([start], [end], i * HOLD, HOLD)[0][0])
        blocked = replace(scene, circles=(Circle(46, 40, 0, 0, 1),))
        result = plan(blocked, goal, previous_action=3)
        self.assertIsNotNone(result.selected)
        self.assertEqual(result.selected.actions[0], 2)
        self.assertFalse(result.smoothed)
