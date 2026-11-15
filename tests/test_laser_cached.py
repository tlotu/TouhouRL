"""Clearance, timing and hold regressions for cached laser control."""
from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

from agent.forward_model import Circle, ForwardModel, Scene
from agent.input_switch import check_switch_prefix, settled_by
from agent.laser_cached import CachedLaserCommitment, intervention_reason, recheck_routes
from scripts.analyze_laser_shadow import validate_cached_shadow


class CachedLaserTests(unittest.TestCase):
    def test_switch_checks_intermediate_motion_and_finishes_the_delayed_tail(self):
        scene = Scene((100., 100.), circles=(Circle(101.7, 103., 0., 0., .1),), player_radius=3.25)
        nominal_end = (105.656854, 94.343146)
        self.assertTrue(ForwardModel(scene).edges([scene.player], [nominal_end], 0, 4)[0][0])
        blocked = check_switch_prefix(scene, 0, 2, lambda: False)
        self.assertEqual(blocked['status'], 'blocked')
        self.assertEqual(blocked['checked_frames'], 1)
        clear = check_switch_prefix(replace(scene, circles=()), 8, 4, lambda: False)
        self.assertEqual(clear['directions'], [[1, 0, 5, 4, 4], [8, 1, 0, 5, 4]])
        self.assertEqual(clear['checked_frames'], 5)

    def test_switch_guard_defers_while_old_events_may_be_queued(self):
        self.assertEqual(settled_by(8, 4, 12, 0), 17)
        self.assertEqual(settled_by(4, 2, 14, 17), 19)
        self.assertEqual(settled_by(4, 4, 14, 17), 17)
        control = CachedLaserCommitment(lambda _: [0.] * 9, lambda: 2, sample_status=lambda: (9, 1.),
                                        read_frame=lambda: 9, read_direction=lambda _: 1)
        scene = Scene((100., 100.))
        self.assertEqual(control._check_switch(scene, 2, 9, 3, lambda: False)['status'], 'pending_keys')
        self.assertEqual(control._check_switch(scene, 4, 9, 3, lambda: False)['status'], 'unconfirmed_direction')

    def test_recheck_requires_padded_clearance_on_every_interval(self):
        scene = Scene((100., 100.), circles=(Circle(104., 104., 0., 0., 1.),))
        routes = [(3, 3, 3, 3)]
        candidate = recheck_routes(scene, routes, 0, 2, lambda: False)
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate['suggested_action'], 3)
        self.assertEqual(candidate['prefix_frames'], 16)
        padded = replace(scene, player_radius=scene.player_radius + 2.)
        self.assertIsNone(recheck_routes(padded, routes, 0, 2, lambda: False))

    def test_ttc_threshold_requires_both_measured_routes(self):
        candidate = {'baseline_first_hold_safe': True, 'baseline_ttc': 19., 'alternative_ttc': 41.}
        self.assertEqual(intervention_reason(candidate), 'leaf_ttc')
        self.assertIsNone(intervention_reason(dict(candidate, baseline_ttc=20.)))
        self.assertIsNone(intervention_reason(dict(candidate, alternative_ttc=40.)))
        self.assertIsNone(intervention_reason(dict(candidate, baseline_ttc=None)))

    def test_hold_uses_only_recheck_and_interrupts_only_with_fresh_input(self):
        obs = {'player': [0., 0., 0., 0.], 'bullets': [[0.] * 7], 'lasers': [[0.] * 12] * 64}
        candidate = {'baseline_first_hold_safe': True, 'baseline_ttc': 19., 'alternative_ttc': 41.,
                     'suggested_action': 3}
        for live, raw_frame, age, expected in ((False, 9, .003, 0), (True, 10, .003, 0),
                                               (True, 9, .015, 0), (True, 9, .003, 3)):
            with self.subTest(live=live, raw_frame=raw_frame, age=age):
                control = CachedLaserCommitment(lambda _: [0.] * 9, lambda: 2, clock=lambda: 1.,
                                                sample_status=lambda: (9, 1. - age),
                                                read_frame=lambda: raw_frame, intervene=live)
                control.held_action, control.since = 0, 0
                control.cached_routes, control.cached_frame, control.cached_horizon = ((3,) * 4,), 0, 16
                with patch('agent.laser_cached.recheck_routes', return_value=candidate) as recheck, \
                        patch('agent.laser_cached.plan') as search:
                    action = control(obs)
                search.assert_not_called()
                if age < .014:
                    self.assertEqual(recheck.call_args.args[3], 2)
                self.assertEqual(action, expected)
                self.assertEqual(control.since, 2 if expected else 0)
                telemetry = control.annotate({'event': 'transition', 'action': action})['planner_cached']
                self.assertTrue(telemetry['baseline_hold_active'])
                self.assertEqual(telemetry['applied'], bool(expected))
                control.annotate({'event': 'reset'})
                self.assertFalse(control.cached_routes)

    def test_gate_requires_ten_complete_episodes_and_fresh_work_below_five(self):
        episode = {'outcome': {'reason': 'spell_success'}, 'counts': {'fresh': 2, 'cached_eligible': 1}}
        report = {'planner_cache': {'mode': 'shadow', 'build_md5': 'build', 'work_ms': {'p95': 4.5},
                                   'fresh_work_ms': {'p95': 4.9}},
                  'episodes': dict.fromkeys(range(10), episode), 'model_sha256': 'model',
                  'candidate_slack': 1., 'errors': [], 'sha256': 'source'}
        with patch('scripts.analyze_laser_shadow.analyze', return_value=report):
            self.assertEqual(validate_cached_shadow(Path('trace.jsonl'), 'model', 'build', 1.), 'source')
            with self.assertRaises(ValueError):
                validate_cached_shadow(Path('trace.jsonl'), 'model', 'build', 1., switch_guard=True)
            report['planner_cache']['fresh_work_ms']['p95'] = 5.
            with self.assertRaises(ValueError):
                validate_cached_shadow(Path('trace.jsonl'), 'model', 'build', 1.)
            report['planner_cache']['fresh_work_ms']['p95'] = 4.9
            report['episodes'].pop(9)
            with self.assertRaises(ValueError):
                validate_cached_shadow(Path('trace.jsonl'), 'model', 'build', 1.)
