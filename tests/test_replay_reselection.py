"""Checks for screened forecast selection and first-hold validity."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from agent.direction_buffer import DirectionBuffer
from agent.forward_model import Circle, Scene, TimedLaser
from agent.swept_collision import SweptCollision, UniqueSweeps
from scripts.audit_replay_replanning_scores import forecast, forecast_candidates
from scripts.audit_replay_terminal_routes import feasible_prefixes
from scripts.evaluate_replay_reselection import choose


class ReselectionTests(unittest.TestCase):
    def test_unique_sweeps_restore_validity_and_clearance_order(self):
        scene = Scene((20., 20.), circles=(Circle(24., 20., 0., 0., .5),),
                      lasers=(TimedLaser(40., 10., 40., 40., 1., 1., 10.),))
        starts = np.tile([[20., 20.], [38., 28.], [20., 26.], [8., 30.]], (12, 1))
        ends = starts + np.tile([[4., 0.], [4., 0.], [4., 0.], [-4., 0.]], (12, 1))
        baseline = SweptCollision(scene)
        cached = UniqueSweeps(scene)
        valid, gaps = baseline.edges(starts, ends, 0, 2)
        unique_valid, unique_gaps = cached.edges(starts, ends, 0, 2)
        np.testing.assert_array_equal(unique_valid, valid)
        np.testing.assert_array_equal(unique_gaps, gaps)
        self.assertLess(cached.counts.potential_pairs, baseline.counts.potential_pairs)

    def test_forecast_candidates_share_first_plan_and_preserve_branches(self):
        scene = Scene((100., 100.))
        sample = SimpleNamespace(observation={})
        origin = SimpleNamespace(scene=scene, frame=0, belief=frozenset((DirectionBuffer(),)),
                                 episode=SimpleNamespace(samples={0: sample}))
        choices = tuple(SimpleNamespace(route=SimpleNamespace(actions=(a,))) for a in (0, 3))
        options = Mock(return_value=(choices, 0, 0))
        worker = SimpleNamespace(options=options)
        with patch('scripts.audit_replay_replanning_scores.rebase', return_value={}):
            expected = {a: forecast(origin, a, worker, 4) for a in (0, 3)}
            self.assertEqual(options.call_count, 2)
            options.reset_mock()
            self.assertEqual(dict(forecast_candidates(origin, (0, 3), worker, 4)), expected)
            self.assertEqual(options.call_count, 1)

    def test_pruned_sweeps_preserve_routes_with_mixed_delay_outcomes(self):
        scene = Scene((20., 20.), circles=(Circle(28., 20., 0., 0., .5),),
                      lasers=(TimedLaser(0., 12., 50., 12., 1., 6., 16.),),
                      bounds=(10., 50., 10., 50.), player_radius=.1)
        queues = (DirectionBuffer(),)

        def layers(delays, **kwargs):
            return list(feasible_prefixes(scene, queues, delays, 0., **kwargs))

        self.assertNotIn((3,), dict(layers((0,))[0][1]))
        self.assertIn((3,), dict(layers((1,))[0][1]))
        expected = layers((0, 1))
        self.assertEqual(layers((0, 1), prune_failed=True, reuse_edges=True), expected)
        result = layers((0, 1), prune_failed=True, reuse_edges=True, final_counts=True)
        self.assertEqual(result[-1][1], [sum(a[0] == move for a, _ in expected[-1][1])
                                         for move in range(9)])
        self.assertEqual(layers((0, 1), prune_failed=True, reuse_edges=True, final_counts=True,
                                vector_terminal=True), result)

    def test_forecast_ceiling_respects_the_tie_break(self):
        def forecasts(env, actions, worker, horizon, policy, *, can_win):
            for action in actions:
                if action != 5 and not can_win(action, 36.):
                    yield action, None
                else:
                    frames = 12 if action == 5 else 36
                    yield action, {'branches': [{'frames': frames}]}

        pruned = {}
        with patch('scripts.evaluate_replay_reselection.forecast_candidates', side_effect=forecasts):
            selected, scores, calls = choose(None, None, None, 5, 60, [5, 8, 7, 6], pruned=pruned)
        self.assertEqual((selected, scores, calls), (6, {5: 12., 8: 36., 7: 36., 6: 36.}, 4))
        self.assertEqual(pruned, {})
        with patch('scripts.evaluate_replay_reselection.forecast_candidates', side_effect=forecasts):
            selected, scores, calls = choose(None, None, None, 5, 60, [5, 7, 8], pruned=pruned)
        self.assertEqual((selected, scores, calls), (7, {5: 12., 7: 36.}, 3))
        self.assertEqual(pruned, {8: 36.})

    def test_only_shortlisted_actions_are_forecast_and_nominal_wins_ties(self):
        env = SimpleNamespace(choices=[SimpleNamespace(route=SimpleNamespace(actions=(a,))) for a in range(9)])
        def forecasts(env, actions, worker, horizon, policy):
            for action in actions:
                yield action, {'branches': [{'frames': 60}, {'frames': 60}]}

        with patch('scripts.evaluate_replay_reselection.forecast_candidates', side_effect=forecasts) as mock:
            selected, scores, calls = choose(env, None, None, 8, 60, [0, 2, 5, 8])
        self.assertEqual(selected, 8)
        self.assertEqual(set(scores), {0, 2, 5, 8})
        self.assertEqual(calls, 4)
        self.assertEqual(mock.call_args.args[1], [0, 2, 5, 8])

    def test_candidate_failing_one_first_hold_is_excluded(self):
        def forecasts(env, actions, worker, horizon, policy):
            for action in actions:
                yield action, {'branches': [{'frames': 60}, {'frames': 3 if action == 2 else 4}]}

        with patch('scripts.evaluate_replay_reselection.forecast_candidates', side_effect=forecasts):
            selected, scores, _ = choose(None, None, None, 0, 60, [0, 2])
        self.assertEqual(selected, 0)
        self.assertEqual(set(scores), {0})

    def test_upper_bound_skips_forecasts_only_when_every_nominal_branch_reaches_it(self):
        for nominal_frames, expected in [((56, 56), (8, {8: 56.}, 1)),
                                         ((56, 52), (3, {8: 54., 3: 56.}, 2))]:
            def forecasts(env, actions, worker, horizon, policy):
                for action in actions:
                    frames = nominal_frames if action == 8 else (56, 56)
                    yield action, {'branches': [{'frames': f} for f in frames]}

            with patch('scripts.evaluate_replay_reselection.forecast_candidates', side_effect=forecasts):
                self.assertEqual(choose(None, None, None, 8, 56, [3, 8], stop_at_horizon=True), expected)


if __name__ == '__main__':
    unittest.main()
