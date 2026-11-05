"""Fake-environment held PPO migration, update and reload with laser input."""
import tempfile
from pathlib import Path
import unittest

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO

from agent.hazard_features import AttackZoneFeatures
from agent.held_training import HeldTransition
from agent.laser_backend import LaserPPOBackend
from agent.laser_policy import encode_lasers
from agent.risk_policy import RiskActorCriticPolicy
from env.laser_session import laser_state_space
from env.touhou_env import state_space
from tests.test_laser_geometry import laser


class Offline(gym.Env):
    action_space=gym.spaces.Discrete(9)
    def __init__(self,lasers=False):
        self.observation_space=laser_state_space(4) if lasers else state_space(4)
    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        obs={k:np.zeros(s.shape,dtype=np.float32)for k,s in self.observation_space.spaces.items()}
        obs['player'][:2]=[0,384/224-1]
        if 'lasers'in obs:
            beam=laser();beam['segment_candidate']=[[100.,364.],[280.,364.]]
            obs['lasers']=encode_lasers([beam])
        return obs,{}
    def step(self,action):
        return self.reset()[0],float(action==5),True,False,{}


class LaserBackendTests(unittest.TestCase):
    def test_held_update_and_reload_retain_combined_policy(self):
        torch.set_num_threads(1)
        oldenv=Offline()
        source=PPO(RiskActorCriticPolicy,oldenv,n_steps=8,batch_size=4,n_epochs=1,
            policy_kwargs={'features_extractor_class':AttackZoneFeatures,
                           'features_extractor_kwargs':{'half_width':24.},
                           'net_arch':{'pi':[128,128],'vf':[128,128]}},seed=4,device='cpu')
        source.learn(total_timesteps=8)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent)as directory:
            initial=Path(directory)/'source.zip';source.save(initial)
            backend=LaserPPOBackend(oldenv,checkpoint=str(initial))
            backend.enable_laser(laser_state_space(4),4)
            config={'learning_rate':1e-5,'epochs':1,'batch_size':4,'entropy_coefficient':0.,
                    'clip_range':.1,'target_kl':.005,'gamma_frame':.999**.5,
                    'lambda_frame':.95**.5,'hold_frames':4}
            backend.configure(config)
            env=Offline(True);obs=env.reset()[0];rows=[]
            for i in range(8):
                decision=backend.act(obs)
                self.assertIn(decision.action,backend.decision_context['candidates'])
                rows.append(HeldTransition(obs,decision.action,float(i%3)*.1,i%4==0,
                                           decision.value,decision.log_prob,4))
            before=backend.model.policy.action_net.weight.detach().clone()
            backend.update(rows,16)
            self.assertFalse(torch.equal(before,backend.model.policy.action_net.weight))
            checkpoint=Path(directory)/'laser.zip';backend.save(checkpoint)
            restored=LaserPPOBackend(env,checkpoint=str(checkpoint));restored.enable_laser(laser_state_space(4),4)
            restored.configure(config)
            self.assertEqual(restored.timesteps,16)
            self.assertEqual(len(restored.model.policy.optimizer.state),len(backend.model.policy.optimizer.state))
            self.assertIn(restored.act(obs).action,restored.decision_context['candidates'])
