"""Deterministic combined-risk evaluation and fake-native provenance."""
import contextlib
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from agent.laser_contract import LASER_TASK
from agent.laser_evaluation import LaserEvaluationCommitment
from agent.laser_policy import encode_lasers
from tests.fakes import FakeKeyboard
from tests.test_laser_geometry import laser
from tests.test_training import FakeTask
from tests import test_train_laser


class LaserEvaluationTests(unittest.TestCase):
    def test_fake_native_evaluation_is_frozen_and_records_combined_candidates(self):
        from scripts.train_laser import main as prepare
        from scripts.evaluate_laser import main as evaluate
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent)as directory:
            root=Path(directory);source,obs=test_train_laser.LaserTrainingTests().make_source(root/'source')
            prep=root/'prepared'
            args=['train_laser','--resume',str(source),'--task',LASER_TASK,'--run-dir',str(prep),'--prepare-only']
            with patch('sys.argv',args),patch('env.windows.connect')as connect, \
                    contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):prepare()
            connect.assert_not_called()
            checkpoint=prep/'checkpoints/update-000000';before=hashlib.sha256((checkpoint/'model.zip').read_bytes()).hexdigest()
            obs={k:v.tolist()for k,v in obs.items()}
            beam=laser();beam['segment_candidate']=[[100.,200.],[280.,200.]]
            obs['lasers']=encode_lasers([beam])
            class Task(FakeTask):
                capacity=4
                @property
                def elapsed_frames(self):return self.position*2
                def reset(self):
                    _,info=super().reset();return deepcopy(obs),info
                def step(self,action):
                    _,reward,done,truncated,info=super().step(action);info['deaths']=0
                    return deepcopy(obs),reward,done,truncated,info
                def __enter__(self):return self
                def __exit__(self,*args):return False
                def verify_training_boundary(self):self.boundary()
            task=Task(length=4);reader=SimpleNamespace(addresses={'SPELL_RESULT_ACTIVE':1},close=lambda:None)
            output=root/'eval.jsonl'
            args=['evaluate_laser','--resume',str(checkpoint),'--task',LASER_TASK,'--episodes','2','--output',str(output)]
            with patch('sys.argv',args),patch('env.windows.connect',return_value=(reader,1,'test-build')), \
                    patch('env.windows.WindowsKeyboard',return_value=FakeKeyboard()), \
                    patch('env.laser_session.LaserReader',side_effect=lambda r,b:r), \
                    patch('env.laser_session.LaserSession',return_value=task), \
                    contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):evaluate()
            rows=[json.loads(line)for line in output.read_text().splitlines()]
            self.assertEqual(rows[0]['collection'],'laser_ppo_evaluation')
            self.assertFalse(rows[0]['training'])
            transitions=[r for r in rows if r['event']=='transition']
            self.assertEqual(len(transitions),8)
            self.assertTrue(all(r['action'] in r['boundary_execution_allowed_actions']for r in transitions))
            self.assertTrue(all(r['action'] in r['laser_risk_context']['candidates']
                                for r in transitions if not r['commitment_active']))
            self.assertEqual(before,hashlib.sha256((checkpoint/'model.zip').read_bytes()).hexdigest())
            alternative=root/'alternate.jsonl'
            args[args.index(str(output))]=str(alternative);args.extend(['--candidate-slack','4'])
            with patch('sys.argv',args),patch('env.windows.connect',return_value=(reader,1,'test-build')), \
                    patch('env.windows.WindowsKeyboard',return_value=FakeKeyboard()), \
                    patch('env.laser_session.LaserReader',side_effect=lambda r,b:r), \
                    patch('env.laser_session.LaserSession',return_value=Task(length=4)), \
                    contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):evaluate()
            altrows=[json.loads(line)for line in alternative.read_text().splitlines()]
            self.assertEqual(altrows[0]['collection'],'laser_ppo_external_slack_evaluation')
            self.assertTrue(altrows[0]['external_candidate_gate'])
            self.assertEqual(altrows[0]['candidate_slack'],4.)
            self.assertEqual(before,hashlib.sha256((checkpoint/'model.zip').read_bytes()).hexdigest())

    def test_evaluation_policy_matches_intrinsic_deterministic_mode(self):
        import gymnasium as gym
        import numpy as np
        from agent.laser_policy import LaserRiskPolicy
        from env.laser_session import laser_state_space
        policy=LaserRiskPolicy(laser_state_space(4),gym.spaces.Discrete(9),lambda _:1e-4)
        obs={k:np.zeros(space.shape,dtype=np.float32) for k,space in laser_state_space(4).spaces.items()}
        obs['player'][:2]=[0.,384/224-1]
        beam=laser();beam['segment_candidate']=[[100.,364.],[280.,364.]]
        obs['lasers']=encode_lasers([beam])
        def logits(state):
            tensor,_=policy.obs_to_tensor(state)
            return policy.unmasked_logits(tensor)[0].detach().tolist()
        choose=LaserEvaluationCommitment(logits,lambda:0)
        action=choose(obs)
        expected=int(policy(policy.obs_to_tensor(obs)[0],deterministic=True)[0].item())
        self.assertEqual(action,expected)
        row=choose.annotate({'event':'transition','action':action})
        self.assertIn(action,row['laser_risk_context']['candidates'])

    def test_wider_gate_changes_only_safe_mode_and_never_physical_bounds(self):
        from tests.test_risk_control import scene
        obs=scene();obs['bullets']=[];obs['lasers']=encode_lasers([])
        scores=[0.]*9;scores[3]=9.;scores[5]=1.
        detail={'allowed':list(range(9)),'combined_clearances':[5.,5.,5.,17.59,5.,21.41,5.,5.,5.],
                'bullet_clearances':[24.]*9,'laser_clearances':[24.]*9,
                'best':21.41,'threshold':20.41,'mode':'safe','candidates':[5]}
        with patch('agent.laser_evaluation.combined_candidates',side_effect=lambda *a,**k:deepcopy(detail)):
            baseline=LaserEvaluationCommitment(lambda _:scores,lambda:0)
            self.assertEqual(baseline(obs),5)
            baseline.annotate({'event':'transition','action':5})
            experiment=LaserEvaluationCommitment(lambda _:scores,lambda:0,candidate_slack=4.)
            self.assertEqual(experiment(obs),3)
            row=experiment.annotate({'event':'transition','action':3})
            self.assertIn(3,row['boundary_execution_allowed_actions'])
            self.assertAlmostEqual(row['laser_risk_context']['threshold'],17.41)
        detail.update(best=-1.,threshold=-1.,mode='emergency',candidates=[5])
        with patch('agent.laser_evaluation.combined_candidates',side_effect=lambda *a,**k:deepcopy(detail)):
            experiment=LaserEvaluationCommitment(lambda _:scores,lambda:0,candidate_slack=4.)
            self.assertEqual(experiment(obs),5)
            self.assertEqual(experiment.annotate({'event':'transition','action':5})['laser_risk_context']['candidates'],[5])
        obs['player'][0]=374/192-1
        experiment=LaserEvaluationCommitment(lambda _:scores,lambda:0,candidate_slack=4.)
        action=experiment(obs)
        row=experiment.annotate({'event':'transition','action':action})
        self.assertNotEqual(action,3)
        self.assertIn(action,row['boundary_execution_allowed_actions'])

    def test_invalid_slack_is_rejected(self):
        for value in (float('nan'),-1,9):
            with self.assertRaises(ValueError):LaserEvaluationCommitment(lambda _:[],lambda:0,candidate_slack=value)
