"""Offline reports preserve trace provenance and pre-action metadata alignment."""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from scripts.probe_planning import laser_regions, main, trace_cases
from agent.forward_model import Scene, TimedLaser


class ProbePlanningTests(unittest.TestCase):
    def test_cli_outputs_and_numbered_directory_preserve_previous_run(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            target = Path(directory) / 'probe'
            first = main(['--output', str(target), '--repeats', '1', '--beam-widths', '8'])
            original = (first / 'report.json').read_bytes()
            second = main(['--output', str(target), '--repeats', '1', '--beam-widths', '8'])
            self.assertNotEqual(first, second)
            self.assertEqual((first / 'report.json').read_bytes(), original)
            report = json.loads(original)
            self.assertEqual(len(report['cases']), 6)
            self.assertTrue((first / 'planning.console.log').exists())
            self.assertEqual(len(list(first.glob('*.svg'))), 6)
            self.assertEqual(len(list(first.glob('*.npz'))), 6)
            for case in report['cases']:
                self.assertTrue(ET.parse(first / (case['name'] + '.svg')).getroot().tag.endswith('svg'))
                with np.load(first / case['reachability_file']['file'], allow_pickle=False) as data:
                    reference = data['reference_reachable']
                    self.assertEqual(list(reference.sum(axis=(1, 2))), case['searches'][0]['layer_reachable_states'])
                    self.assertTrue(np.all(data['beam_8_reachable'] <= reference))

    def test_trace_uses_previous_transition_metadata_and_rejects_held_states(self):
        obs = {'player': [0, 0, 0, 0], 'bullets': [[0] * 7], 'lasers': np.zeros((64, 12)).tolist()}
        header = {'schema': 'touhourl-static-laser-state-v1', 'hold_frames': 4}
        prior = {'event': 'transition', 'episode': 0, 'step': 1, 'next_observation': obs,
                 'info': {'dropped_bullets': 7, 'bullet_count': 135}}
        selected = {'event': 'transition', 'episode': 0, 'step': 2, 'observation': obs,
                    'next_observation': obs, 'commitment_active': False,
                    'info': {'dropped_bullets': 19, 'elapsed_frames': 4, 'delta_frames': 2}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trace.jsonl'
            path.write_text('\n'.join(json.dumps(r) for r in (header, prior, selected)), encoding='utf-8')
            original = path.read_bytes()
            cases, digest = trace_cases(path, [(0, 2)], 16)
            self.assertEqual(digest, hashlib.sha256(original).hexdigest())
            self.assertTrue(cases[0]['metadata']['metadata_aligned'])
            self.assertEqual(cases[0]['metadata']['prior_dropped_bullets'], 7)
            self.assertEqual(path.read_bytes(), original)
            selected['commitment_active'] = True
            path.write_text('\n'.join(json.dumps(r) for r in (header, prior, selected)), encoding='utf-8')
            with self.assertRaises(ValueError):
                trace_cases(path, [(0, 2)], 16)

    def test_future_room_is_time_dependent_and_only_a_geometric_score(self):
        scene = Scene((8, 20), lasers=(TimedLaser(20, 0, 20, 40, 2, 8, 20),), bounds=(0, 40, 0, 40))
        before, _ = laser_regions(scene, 0)
        during, _ = laser_regions(scene, 12)
        after, _ = laser_regions(scene, 24)
        self.assertGreater(before(scene.player), during(scene.player))
        self.assertEqual(before(scene.player), after(scene.player))
