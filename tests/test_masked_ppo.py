"""Masked likelihood consistency, conversion, actual updates and held interruption."""
from copy import deepcopy
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tests.test_ppo_backend import AVAILABLE
from agent.held_training import HeldTransition
from agent.training import TrainingError, Decision
from agent.masked_training import collect_masked_episodes, survival_reward
from tests.test_training import FakeTask


class MaskedCollectionTests(unittest.TestCase):
    def test_reward_distinguishes_death_timeout_and_success(self):
        def reward(reason, deaths=0):
            return survival_reward({'delta_frames': 2, 'reason': reason, 'deaths': deaths}, 1)
        self.assertAlmostEqual(reward('spell_success'), 5.001)
        self.assertAlmostEqual(reward('spell_failure', 1), -4.999)
        self.assertAlmostEqual(reward('spell_failure'), -.999)

    def test_unsafe_remaining_hold_ends_before_new_sample(self):
        task = FakeTask(length=3)
        def obs(x):
            return {'player': [x/192-1, 300/224-1, 0, 0]}
        task.reset = lambda: (obs(368), {})
        original = task.step
        actions = []
        def step(a):
            actions.append(a)
            _, reward, done, truncated, info = original(a)
            info['delta_frames'] = 3 if task.position == 1 else 2
            return obs(374 if task.position == 1 else 370), reward, done, truncated, info
        task.step = step
        class Learner:
            model = SimpleNamespace(policy=SimpleNamespace(mask_hold_frames=4))
            def act(self, state):
                return Decision(3 if state['player'][0] < .93 else 7, .1, -.2)
        # One complete episode has two fresh decisions; no action is replaced.
        rows, _ = collect_masked_episodes(task, Learner(), 2, lambda r: None)
        self.assertEqual(actions, [3, 7, 7])
        self.assertEqual([r.duration_frames for r in rows], [3, 4])
        self.assertEqual([r.action for r in rows], [3, 7])
        self.assertTrue(all(r.log_prob == -.2 for r in rows))


@unittest.skipUnless(AVAILABLE, 'Install training dependencies')
class MaskedPPOTests(unittest.TestCase):
    def make(self):
        import gymnasium as gym
        import numpy as np
        from env.touhou_env import state_space
        from agent.masked_backend import MaskedPPOBackend
        from scripts.train_masked import masked_config
        from tests.test_held_training import clone_metadata
        class Offline(gym.Env):
            observation_space = state_space(4)
            action_space = gym.spaces.Discrete(9)
        learner = MaskedPPOBackend(Offline(), features='hazard-zone-v1', feature_options={'half_width':24})
        config = masked_config(clone_metadata())
        config.update(capacity=4, max_frames=3600, seed=0)
        state = {k: np.zeros(s.shape, dtype=s.dtype) for k,s in Offline.observation_space.spaces.items()}
        return learner, config, state, Offline

    def test_conversion_preserves_parameters_and_mask_matches_scalar_geometry(self):
        import torch
        from agent.boundary_control import feasible_actions
        learner, config, obs, _ = self.make()
        before = deepcopy(learner.model.policy.state_dict())
        learner.enable_mask(4)
        for k,v in before.items():torch.testing.assert_close(v, learner.model.policy.state_dict()[k])
        for x,y in [(8,16),(376,432),(370,300),(366,300),(192,224)]:
            obs['player'][:2] = [x/192-1,y/224-1]
            tensor,_ = learner.model.policy.obs_to_tensor(obs)
            mask = learner.model.policy.action_mask(tensor)[0].tolist()
            self.assertEqual([i for i,v in enumerate(mask) if v], feasible_actions(obs,4))
            p = learner.model.policy.get_distribution(tensor).distribution.probs[0]
            self.assertAlmostEqual(float(p.sum().detach()),1., places=6)
            self.assertTrue(all(float(p[i].detach()) == 0 for i in range(9) if not mask[i]))

    def test_sampling_evaluation_update_and_optimizer_reload(self):
        import torch
        from agent.masked_backend import MaskedPPOBackend
        learner, config, obs, Env = self.make()
        learner.enable_mask(4);learner.configure(config['held_ppo'])
        obs['player'][0] = 376/192-1
        def batch():
            result=[]
            for i in range(8):
                d=learner.act(obs)
                tensor,_=learner.model.policy.obs_to_tensor(obs)
                _,logp,_=learner.model.policy.evaluate_actions(tensor,torch.tensor([d.action]))
                self.assertAlmostEqual(d.log_prob,float(logp[0].detach()),places=6)
                self.assertNotIn(d.action,[2,3,4])
                result.append(HeldTransition(deepcopy(obs),d.action,.1+i*.01,i==0,d.value,d.log_prob,4))
            return result
        before=deepcopy(learner.model.policy.state_dict());learner.update(batch(),16)
        self.assertTrue(any(k.startswith(('action_net.','mlp_extractor.policy_net.')) and
                            not torch.equal(v,learner.model.policy.state_dict()[k])for k,v in before.items()))
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            p=Path(directory)/'model.zip';learner.save(p)
            restored=MaskedPPOBackend(Env(),checkpoint=str(p));restored.enable_mask(4);restored.configure(config['held_ppo'])
            self.assertEqual(restored.timesteps,8)
            self.assertEqual(len(restored.model.policy.optimizer.state),len(learner.model.policy.optimizer.state))
            original_optimizer=learner.model.policy.optimizer.state_dict()['state']
            restored_optimizer=restored.model.policy.optimizer.state_dict()['state']
            for key,values in original_optimizer.items():
                for name,value in values.items():
                    torch.testing.assert_close(value,restored_optimizer[key][name])
            for k,v in learner.model.policy.state_dict().items():torch.testing.assert_close(v,restored.model.policy.state_dict()[k])
            learner=restored;learner.update(batch(),24)
            self.assertEqual(learner.timesteps,16)

    def test_unmasked_likelihood_is_rejected_before_update(self):
        from dataclasses import replace
        learner, config, obs, _ = self.make();learner.enable_mask(4);learner.configure(config['held_ppo'])
        obs['player'][0]=376/192-1;d=learner.act(obs)
        row=HeldTransition(obs,d.action,.1,True,d.value,d.log_prob+1,4)
        with self.assertRaises(TrainingError):learner.update([row],2)
        self.assertEqual(learner.timesteps,0)

    def test_offline_prepare_and_task_guards(self):
        from agent.checkpoints import save_checkpoint,read_checkpoint
        from scripts.train_masked import main
        from agent.masked_contract import MASKED_REWARD_PROFILE
        learner,config,obs,_=self.make()
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            original_config={k:v for k,v in config.items() if k not in ('ppo_contract','held_ppo')}
            source=save_checkpoint(learner,Path(directory)/'source',{'updates':0,'timesteps':0,
                'initialization':{'identity':{'tasks':['a','b']}}},original_config,'test')
            original=(source/'model.zip').read_bytes()
            args=['train_masked','--resume',str(source),'--task','a','--prepare-only','--run-dir',str(Path(directory)/'prepared')]
            with patch('sys.argv',args),patch('env.windows.connect') as connect,contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                main();connect.assert_not_called()
            prepared=Path(directory)/'prepared/checkpoints/update-000000'
            self.assertEqual(read_checkpoint(prepared)['reward_profile'],MASKED_REWARD_PROFILE)
            from scripts.train_held import held_config
            with self.assertRaises(ValueError):held_config(read_checkpoint(prepared))
            from scripts.evaluate_control import main as evaluate
            eval_args=['evaluate_control','--resume',str(prepared),'--task','a','--output','unused']
            with patch('sys.argv',eval_args),patch('scripts.evaluate_control.connect') as connect, \
                    contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):evaluate()
            connect.assert_not_called()
            self.assertEqual((source/'model.zip').read_bytes(),original)
            args[args.index('a')]='unseen'
            with patch('sys.argv',args),patch('env.windows.connect') as connect,contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):main()
            connect.assert_not_called()

    def test_native_entry_uses_complete_episodes_and_publishes_reloadable_update(self):
        from agent.checkpoints import save_checkpoint, read_checkpoint
        from scripts.train_masked import main
        from agent.masked_backend import MaskedPPOBackend
        learner,config,state,Env=self.make()
        class Task(FakeTask):
            capacity=4
            def reset(self):
                _,info=super().reset()
                return deepcopy(state),info
            def step(self,action):
                _,reward,done,truncated,info=super().step(action)
                info['deaths']=0
                return deepcopy(state),reward,done,truncated,info
            def __enter__(self):return self
            def __exit__(self,*args):return False
            def verify_training_boundary(self):self.boundary()
        task=Task(length=2)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            initial_config={k:v for k,v in config.items()if k not in ('ppo_contract','held_ppo')}
            source=save_checkpoint(learner,Path(directory)/'source',{'updates':0,'timesteps':0,
                'initialization':{'identity':{'tasks':['a','b']}}},initial_config,'test')
            before=(source/'model.zip').read_bytes();run=Path(directory)/'run'
            args=['train_masked','--resume',str(source),'--task','a','--decisions','2',
                  '--rollout-decisions','2','--run-dir',str(run)]
            reader=SimpleNamespace(addresses={'SPELL_RESULT_ACTIVE':1},close=lambda:None)
            from tests.fakes import FakeKeyboard
            with patch('sys.argv',args),patch('env.windows.connect',return_value=(reader,1,'test')), \
                    patch('env.windows.WindowsKeyboard',return_value=FakeKeyboard()), \
                    patch('env.session.GameSession',return_value=task), \
                    contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):main()
            checkpoint=run/'checkpoints/update-000001'
            meta=read_checkpoint(checkpoint)
            restored=MaskedPPOBackend(Env(),checkpoint=str(checkpoint/'model.zip'))
            self.assertEqual((meta['updates'],restored.timesteps),(1,2))
            self.assertEqual(meta['fine_tuning']['task'],'a')
            self.assertEqual(meta['initialization']['identity']['tasks'],['a','b'])
            self.assertEqual((source/'model.zip').read_bytes(),before)
            self.assertEqual(task.resets,2)
            self.assertTrue(task.menu)
