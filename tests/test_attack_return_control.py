"""Safety precedence, detour ranking, hold semantics and opt-in CLI guards."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from agent.attack_return_control import return_choice, AttackReturnCommitment
from tests.test_risk_control import scene
from tests.test_boundary_control import scores
from tests.test_control_evaluation import metadata
from tests.test_training import FakeTask
from tests.fakes import FakeKeyboard
from tests.test_ppo_backend import AVAILABLE


def target_scene(dx=-100.):
    obs = scene()
    obs['bullets'] = []
    obs['enemies'][0] = [dx / 384, -.5, 1., 1., 1., 1.]
    return obs


class AttackReturnTests(unittest.TestCase):
    def choose(self, obs, candidates, mode='safe'):
        return return_choice(obs, candidates, scores(3, 0), mode, 4, 24.)[0]

    def test_approach_beats_actor_retreat_but_respects_attack_band(self):
        self.assertEqual(self.choose(target_scene(), list(range(9))), 6)
        self.assertEqual(self.choose(target_scene(10.), list(range(9))), 3)
        self.assertEqual(self.choose(target_scene(100.), list(range(9))), 3)

    def test_vertical_detour_precedes_stay_and_retreat(self):
        self.assertIn(self.choose(target_scene(), [0, 1, 3, 5]), [1, 5])
        self.assertEqual(self.choose(target_scene(), [0, 3]), 0)
        self.assertEqual(self.choose(target_scene(), [3]), 3)

    def test_safety_candidates_and_emergency_take_precedence(self):
        self.assertEqual(self.choose(target_scene(), [3], 'safe'), 3)
        self.assertEqual(self.choose(target_scene(), [3, 7], 'emergency'), 3)
        obs = target_scene()
        obs['enemies'][0][4] = 0
        self.assertEqual(self.choose(obs, [3, 7]), 3)
        obs['enemies'][0][4] = 1
        obs['enemies'][0][1] = .1
        self.assertEqual(self.choose(obs, [3, 7]), 3)

    def test_boss_target_precedes_nearby_minion(self):
        obs = target_scene()
        obs['enemies'][1] = [80 / 384, -.2, 0., 1., 1., 1.]
        self.assertIn(self.choose(obs, [3, 7]), [7])

    def test_return_ranking_does_not_interrupt_hold(self):
        frame = [0]
        controller = AttackReturnCommitment(lambda _: scores(3, 7), lambda: frame[0])
        action = controller(target_scene())
        self.assertEqual(action, 7)
        row = controller.annotate({'event': 'transition', 'action': action})
        self.assertEqual(row['risk_filtered_action'], 3)
        self.assertEqual(row['attack_return_action'], 7)
        frame[0] = 2
        self.assertEqual(controller(target_scene(100.)), 7)
        row = controller.annotate({'event': 'transition', 'action': 7})
        self.assertTrue(row['commitment_active'])
        self.assertEqual(row['attack_return_action'], 3)
        self.assertFalse(row['boundary_hold_interrupted'])
        frame[0] = 4
        self.assertEqual(controller(target_scene(100.)), 3)
        controller.annotate({'event': 'transition', 'action': 3})
        self.assertEqual(controller.summary()['attack_return_applied_fresh_choices'], 1)

    def test_cli_rejects_missing_risk_flags_before_connection(self):
        from scripts.evaluate_control import main
        args = ['evaluate_control', '--resume', 'unused', '--task', 'source',
                '--output', 'unused', '--attack-return']
        with patch('sys.argv', args), patch('scripts.evaluate_control.connect') as connect, \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main()
        connect.assert_not_called()

    @unittest.skipUnless(AVAILABLE, 'Install training dependencies')
    def test_fake_native_cli_labels_override_and_preserves_archive(self):
        from scripts.evaluate_control import main
        class Task(FakeTask):
            capacity = 4
            @property
            def elapsed_frames(self): return self.position * 2
            def reset(self):
                _, info = super().reset()
                return target_scene(), info
            def step(self, action):
                _, reward, done, truncated, info = super().step(action)
                return target_scene(), reward, done, truncated, info
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def verify_training_boundary(self): self.boundary()
        class Backend:
            def __init__(self, *args, **kwargs): pass
            def action_logits(self, obs): return scores(3, 7)
            def update(self, *args, **kwargs): raise AssertionError('No training')
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            checkpoint = Path(directory) / 'source'; checkpoint.mkdir()
            archive = checkpoint / 'model.zip'; archive.write_bytes(b'frozen')
            output = Path(directory) / 'return.jsonl'
            args = ['evaluate_control', '--resume', str(checkpoint), '--task', 'source',
                    '--output', str(output), '--episodes', '2', '--boundary-mask',
                    '--risk-filter', '--attack-return']
            reader = SimpleNamespace(addresses={'SPELL_RESULT_ACTIVE': 1}, close=lambda: None)
            with patch('sys.argv', args), patch('scripts.evaluate_control.read_checkpoint', return_value=metadata()), \
                    patch('scripts.evaluate_control.connect', return_value=(reader, 1, 'test-build')), \
                    patch('scripts.evaluate_control.WindowsKeyboard', return_value=FakeKeyboard()), \
                    patch('scripts.evaluate_control.GameSession', return_value=Task(length=3)), \
                    patch('agent.ppo_backend.PPOBackend', Backend), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                main()
            rows = [json.loads(s) for s in output.read_text().splitlines()]
            self.assertEqual(rows[0]['collection'], 'attack_return_control_evaluation')
            self.assertFalse(rows[0]['attack_return_training'])
            transitions = [r for r in rows if r['event'] == 'transition']
            self.assertEqual([r['action'] for r in transitions], [7] * 6)
            self.assertTrue(all(r['risk_filtered_action'] == 3 for r in transitions))
            self.assertEqual(rows[-1]['attack_return_applied_fresh_choices'], 4)
            self.assertEqual(archive.read_bytes(), b'frozen')
