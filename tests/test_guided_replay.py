"""Regressions for timed detours and the replay's interpretation of cutoffs."""
import json
from pathlib import Path
import tempfile
import unittest

from agent.forward_model import Circle, Scene, TimedLaser
from agent.replay_data import decision_record
from agent.short_planner import SearchBudget, plan
from scripts.probe_short_planning import _rollout, benchmark_cases
from scripts.replay_laser_counterfactual import Sample, checked_suffix, roll
from scripts.replay_laser_trace import recorded_decisions
from scripts.replay_planning import write_replay


class GuidedReplayTests(unittest.TestCase):
    def test_counterfactual_refreshes_hazards_during_a_held_move(self):
        clear = Scene((8., 20.), bounds=(0., 40., 0., 40.))
        later = Scene(clear.player, circles=(Circle(15., 20., 0., 0., .5),), bounds=clear.bounds)
        samples = [Sample(0, 2, clear, 3, clear.player, True, 0),
                   Sample(2, 4, later, 3, clear.player, False, 0)]
        frames, _, summary = roll(samples, 'short_ttc', keep_first=True)
        self.assertEqual(frames[0]['branch_path'][1], (12., 20.))
        self.assertEqual(summary['stop'], 'model_contact')
        self.assertEqual(summary['first_contact_interval_by_margin_px'][0], [2, 4])

    def test_prior_route_is_rejected_if_a_new_beam_blocks_its_tail(self):
        clear = Scene((8., 20.), bounds=(0., 40., 0., 40.))
        later = Scene(clear.player, lasers=(TimedLaser(20, 0, 20, 40, 2, 4, 40),), bounds=clear.bounds)
        self.assertEqual(checked_suffix(clear, (3, 3))[-1], [24., 20.])
        self.assertIsNotNone(checked_suffix(later, (3,)))
        self.assertIsNone(checked_suffix(later, (3, 3)))

    def test_recorded_hold_uses_preact_metadata_and_stops_at_death(self):
        def observation(x):
            return {'player': [x, 0, 0, 0], 'bullets': [[0] * 7], 'lasers': [[0] * 12] * 64}

        def transition(step, start, x, next_x, *, held=False, deaths=0):
            return {'step': step, 'observation': observation(x), 'next_observation': observation(next_x),
                    'action': 3 if step else 7, 'commitment_active': held,
                    'info': {'elapsed_frames': start + 2, 'deaths': deaths, 'dropped_bullets': step + 5}}

        prior = transition(0, 0, 0, .125)
        fresh = transition(1, 2, .125, .25)
        terminal = transition(2, 4, .25, 0, held=True, deaths=1)
        [(_, previous, frame)] = recorded_decisions([(2, fresh, prior), (4, terminal, fresh)], 4)
        self.assertEqual(previous, 7)
        self.assertEqual(frame['frame'], 2)
        self.assertEqual(frame['dropped_circles'], 5)
        self.assertEqual(frame['recorded_path'], [(216, 224), (240, 224)])
        self.assertEqual(frame['death_observed'], 6)

    def test_reference_subgoals_fix_static_detour(self):
        case = next(case for case in benchmark_cases() if case['name'] == 'detour_before_approach')
        self.assertFalse(_rollout(case, {})['pass'])
        guided = _rollout(case, {}, guide_static=True)
        self.assertTrue(guided['pass'])
        self.assertLess(guided['waypoints'][0]['y'], case['scene'].lasers[0].ay)
        self.assertLessEqual(guided['goal_arrival_frame'], case['goal'].deadline)

    def test_replay_keeps_budget_stop_separate_from_predicted_collision(self):
        scene = Scene((8., 20.), lasers=(TimedLaser(12, 0, 12, 40, 2, 0, 80),), bounds=(0, 40, 0, 40))
        ticks = iter((0., .002))
        result = plan(scene, budget=SearchBudget(378, 1.), clock=lambda: next(ticks))
        record = decision_record(scene, 0, result=result)
        self.assertEqual(record['stop_reason'], 'time_limit')
        self.assertIsNone(record['action'])
        self.assertEqual(len(record['directions']), 9)
        self.assertEqual(record['directions'][3]['status'], 'collision')
        self.assertIsNone(record['directions'][3]['ttc_after_hold'])

    def test_export_embeds_tracks_without_ending_the_data_script(self):
        report = {'tracks': [{'label': '</script>', 'frames': [{'frame': 0}]}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'replay.html'
            write_replay(report, path)
            page = path.read_text(encoding='utf-8')
            data = page.split('id="data" type="application/json">', 1)[1].split('</script>', 1)[0]
            self.assertEqual(json.loads(data), report['tracks'])
            with self.assertRaises(FileExistsError):
                write_replay(report, path)
