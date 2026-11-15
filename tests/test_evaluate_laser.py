"""Deterministic combined-risk evaluation and fake-native provenance."""
import contextlib
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from agent.laser_contract import LASER_TASK
from agent.laser_evaluation import LaserEvaluationCommitment
from agent.laser_policy import encode_lasers
from agent.laser_shadow import BUFFERED_LEAF_CONFIG, SHADOW_CONFIG, LaserBufferedReuseCommitment
from scripts.analyze_laser_shadow import analyze, first_partition
from tests.fakes import FakeKeyboard
from tests.test_laser_geometry import laser
from tests.test_training import FakeTask
from tests import test_train_laser


class LaserEvaluationTests(unittest.TestCase):
    def test_buffered_cache_discards_suffix_that_ends_in_a_warning_beam(self):
        frame = 2
        controller = LaserBufferedReuseCommitment(lambda _: [1.] + [0.] * 8, lambda: frame,
                                                   clock=lambda: 0.)
        controller.since = controller.held_action = 0
        beam = laser(phase=0, timer=20)
        beam['segment_candidate'] = [[192., 100.], [192., 400.]]
        obs = {'player': np.zeros(4, dtype=np.float32), 'lasers': encode_lasers([beam]),
               'bullets': np.zeros((1, 7), dtype=np.float32)}

        def seed_cache():
            controller.cached_frame, controller.cached_horizon = 0, 60
            controller.cached_actions = (0,) * 6

        seed_cache()
        self.assertEqual(controller(obs), 0)
        held = controller.annotate({'event': 'transition', 'action': 0,
                                    'info': {'elapsed_frames': 4, 'delta_frames': 2}})
        record = held['planner_shadow']
        self.assertEqual((record['source'], record['recheck']), ('hold', 'warning_ttc'))
        self.assertEqual(record['cache_warning_ttc_frames'], 18.)
        self.assertEqual(record['checked_prefix_frames'], 2)
        self.assertEqual(controller.cached_actions, ())

        frame = 4
        seed_cache()
        self.assertEqual(controller(obs), 0)
        record = controller.last['planner_shadow']
        self.assertEqual((record['source'], record['recheck']), ('search', 'warning_ttc'))

    def test_buffered_hold_rechecks_suffix_and_only_searches_on_collision(self):
        frame = 0
        controller = LaserBufferedReuseCommitment(lambda _: [1.] + [0.] * 8, lambda: frame,
                                                   clock=lambda: 0.)
        obs = {'player': np.zeros(4, dtype=np.float32), 'lasers': encode_lasers([]),
               'bullets': np.zeros((1, 7), dtype=np.float32)}

        def step():
            nonlocal frame
            action = controller(obs)
            frame += 2
            return controller.annotate({'event': 'transition', 'action': action,
                                        'info': {'elapsed_frames': frame, 'delta_frames': 2}})

        self.assertEqual(step()['planner_shadow']['source'], 'search')
        reused = step()['planner_shadow']
        self.assertTrue(reused['cache_reused'])
        self.assertEqual(reused['checked_prefix_frames'], 14)
        self.assertEqual(reused['route_durations'][0], 2)
        step()
        obs['bullets'][0] = [4. / 384, 0., -1. / 16, 0., 0., 0., 1.]
        interrupted = step()
        self.assertTrue(interrupted['commitment_active'])
        self.assertEqual(interrupted['planner_shadow']['source'], 'search')
        self.assertEqual(interrupted['planner_shadow']['recheck'], 'collision')
        self.assertEqual(interrupted['action'], 0)

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
                sample_received_at=property(lambda _:time.perf_counter())
                timing={'sample_read_seconds':.002}
                @property
                def elapsed_frames(self):return self.position*2
                def reset(self):
                    _,info=super().reset();return deepcopy(obs),info
                def step(self,action):
                    _,reward,done,truncated,info=super().step(action);info['deaths']=0
                    info['timing']={'applied_action':action,'observation_age_seconds':.01,
                                    'input_send_seconds':.001,'sample_read_seconds':.002}
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

            self.assertNotIn('planner_shadow',altrows[0])
            baseline_steps=[r for r in altrows if r['event']=='transition']
            out_index=args.index(str(alternative))
            args.append('')
            for flag, expected in (('--planner-shadow', SHADOW_CONFIG), ('--planner-buffered-shadow', BUFFERED_LEAF_CONFIG)):
                shadow=root/(expected['contract']+'.jsonl')
                args[out_index]=str(shadow)
                args[-1]=flag
                with patch('sys.argv',args),patch('env.windows.connect',return_value=(reader,1,'test-build')), \
                        patch('env.windows.WindowsKeyboard',return_value=FakeKeyboard()), \
                        patch('env.laser_session.LaserReader',side_effect=lambda r,b:r), \
                        patch('env.laser_session.LaserSession',return_value=Task(length=4)), \
                        contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):evaluate()
                shadow_rows=[json.loads(line)for line in shadow.read_text().splitlines()]
                self.assertEqual(shadow_rows[0]['planner_shadow'],expected)
                shadow_steps=[r for r in shadow_rows if r['event']=='transition']
                self.assertEqual([r['action'] for r in shadow_steps],[r['action'] for r in baseline_steps])
                buffered = expected == BUFFERED_LEAF_CONFIG
                self.assertEqual(['planner_shadow' in r for r in shadow_steps],
                                 [True] * 8 if buffered else [True,False,True,False]*2)
                if buffered:
                    self.assertTrue(all(r['observation']==previous['next_observation']
                                        for previous,r in zip(shadow_steps,shadow_steps[1:])
                                        if previous['episode']==r['episode']))
                    self.assertIn('json_write_seconds',shadow_steps[1]['previous_trace_timing'])
                report=analyze(shadow)
                self.assertEqual(report['episodes'][0]['counts']['shadow_samples'],4 if buffered else 2)
                timings=report['episodes'][1]['timing_ms']
                self.assertAlmostEqual(timings['fresh_observation_to_send_complete']['p95'],11.)
                self.assertAlmostEqual(timings['fresh_capture_to_send_complete']['p95'],13.)
                self.assertEqual(timings['fresh_capture_to_send_complete']['count'],1)
                self.assertEqual(before,hashlib.sha256((checkpoint/'model.zip').read_bytes()).hexdigest())

            cached=root/'cached.jsonl'
            args[out_index]=str(cached)
            args[-1]='--planner-cache-shadow'
            reader.frame=lambda:9
            Task.sample=SimpleNamespace(frame=9)
            Task.sample_received_at=property(lambda _:time.perf_counter())
            with patch('sys.argv',args),patch('env.windows.connect',return_value=(reader,1,'test-build')), \
                    patch('env.windows.WindowsKeyboard',return_value=FakeKeyboard()), \
                    patch('env.laser_session.LaserReader',side_effect=lambda r,b:r), \
                    patch('env.laser_session.LaserSession',return_value=Task(length=4)), \
                    contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):evaluate()
            cached_rows=[json.loads(line)for line in cached.read_text().splitlines()]
            cached_steps=[r for r in cached_rows if r['event']=='transition']
            self.assertEqual([r['action'] for r in cached_steps],[r['action'] for r in baseline_steps])
            self.assertTrue(all(not r['planner_cached']['search_attempted'] for r in cached_steps if r['commitment_active']))
            self.assertEqual(analyze(cached)['planner_cache']['mode'],'shadow')
            self.assertEqual(before,hashlib.sha256((checkpoint/'model.zip').read_bytes()).hexdigest())

    def test_partition_waits_for_full_activation(self):
        def state(phases):
            return {'player': [0., .5, 0., 0.], 'lasers': encode_lasers([laser(phase) for phase in phases])}

        samples = [(10, state([0, 0])), (12, state([1, 0])), (14, state([1, 1]))]
        self.assertIsNone(first_partition(samples[:2])['full_activation_frame'])
        room = first_partition(samples)
        self.assertEqual(room['full_activation_frame'], 14)
        self.assertEqual(room['active_lasers'], 2)
        self.assertGreater(room['area_pixels2'], 0)

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
