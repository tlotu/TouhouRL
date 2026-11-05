"""Laser feature and PPO distribution prototype runs only on fake observations."""
import unittest
import tempfile
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch

from agent.laser_policy import encode_lasers,decode_lasers,LaserRiskPolicy,LASER_CAPACITY,LASER_FIELDS,migrate_risk_policy
from agent.laser_risk import combined_candidates
from env.touhou_env import state_space
from tests.test_laser_geometry import laser


class LaserPolicyTests(unittest.TestCase):
    def test_encoding_preserves_phase_and_rejects_overflow(self):
        row=laser(0,55)
        self.assertEqual(decode_lasers(encode_lasers([row]))[0]['timer_raw'],55)
        with self.assertRaises(ValueError):encode_lasers([row]*65)
        moving=laser();moving['speed']=1
        with self.assertRaises(ValueError):encode_lasers([moving])

    def test_sample_and_likelihood_use_same_combined_distribution(self):
        spaces=dict(state_space(4).spaces)
        spaces['lasers']=gym.spaces.Box(-np.inf,np.inf,shape=(LASER_CAPACITY,LASER_FIELDS),dtype=np.float32)
        policy=LaserRiskPolicy(gym.spaces.Dict(spaces),gym.spaces.Discrete(9),lambda _:1e-4)
        obs={k:np.zeros(s.shape,dtype=np.float32)for k,s in spaces.items()}
        obs['player'][:2]=[0,384/224-1]
        beam=laser();beam['segment_candidate']=[[100.,364.],[280.,364.]]
        obs['lasers']=encode_lasers([beam])
        tensor,_=policy.obs_to_tensor(obs)
        expected=combined_candidates(obs,decode_lasers(obs['lasers']))['candidates']
        self.assertEqual(torch.where(policy.action_mask(tensor)[0])[0].tolist(),expected)
        self.assertEqual(policy.extract_features(tensor).shape[1],71)
        actions,_,logp=policy(tensor)
        self.assertIn(int(actions.item()),expected)
        _,repeat,entropy=policy.evaluate_actions(tensor,actions)
        torch.testing.assert_close(logp,repeat)
        self.assertTrue(torch.isfinite(entropy).all())
        (-repeat.mean()).backward()
        self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() for p in policy.action_net.parameters()))

    def test_features_change_with_laser(self):
        spaces=dict(state_space(4).spaces)
        spaces['lasers']=gym.spaces.Box(-np.inf,np.inf,shape=(LASER_CAPACITY,LASER_FIELDS),dtype=np.float32)
        policy=LaserRiskPolicy(gym.spaces.Dict(spaces),gym.spaces.Discrete(9),lambda _:1e-4)
        obs={k:np.zeros(s.shape,dtype=np.float32)for k,s in spaces.items()}
        obs['player'][:2]=[0,384/224-1]
        before=policy.extract_features(policy.obs_to_tensor(obs)[0])
        beam=laser();beam['segment_candidate']=[[100.,364.],[280.,364.]]
        obs['lasers']=encode_lasers([beam])
        after=policy.extract_features(policy.obs_to_tensor(obs)[0])
        self.assertFalse(torch.equal(before[:,62:],after[:,62:]))

    def test_fake_ppo_update_and_checkpoint_reload(self):
        from stable_baselines3 import PPO
        torch.set_num_threads(1)
        class FakeLaserEnv(gym.Env):
            action_space=gym.spaces.Discrete(9)
            observation_space=gym.spaces.Dict({**dict(state_space(4).spaces),
                'lasers':gym.spaces.Box(-np.inf,np.inf,shape=(LASER_CAPACITY,LASER_FIELDS),dtype=np.float32)})
            def __init__(self):self.count=0
            def observation(self):
                obs={k:np.zeros(s.shape,dtype=np.float32)for k,s in self.observation_space.spaces.items()}
                obs['player'][:2]=[0,384/224-1]
                beam=laser();beam['segment_candidate']=[[100.,364.],[280.,364.]]
                obs['lasers']=encode_lasers([beam]);return obs
            def reset(self,*,seed=None,options=None):
                super().reset(seed=seed);self.count=0;return self.observation(),{}
            def step(self,action):
                self.count+=1
                return self.observation(),float(action==5),self.count>=4,False,{}
        env=FakeLaserEnv()
        model=PPO(LaserRiskPolicy,env,n_steps=8,batch_size=4,n_epochs=1,learning_rate=1e-3,seed=7,device='cpu')
        before=model.policy.action_net.weight.detach().clone()
        model.learn(total_timesteps=8)
        self.assertFalse(torch.equal(before,model.policy.action_net.weight))
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent)as directory:
            file=Path(directory)/'fake-laser.zip';model.save(file)
            restored=PPO.load(file,env=env,device='cpu')
            self.assertEqual(restored.num_timesteps,8)
            self.assertTrue(torch.equal(model.policy.action_net.weight,restored.policy.action_net.weight))
            restored.learn(total_timesteps=8,reset_num_timesteps=False)
            self.assertEqual(restored.num_timesteps,16)

    def test_migration_preserves_logits_and_nonempty_adam_without_lasers(self):
        from agent.risk_policy import RiskActorCriticPolicy
        from agent.hazard_features import AttackZoneFeatures
        oldspace=state_space(4)
        kwargs={'features_extractor_class':AttackZoneFeatures,'features_extractor_kwargs':{'half_width':24.},
                'net_arch':{'pi':[128,128],'vf':[128,128]}}
        old=RiskActorCriticPolicy(oldspace,gym.spaces.Discrete(9),lambda _:1e-4,**kwargs)
        obs={k:np.zeros(s.shape,dtype=np.float32)for k,s in oldspace.spaces.items()}
        obs['player'][:2]=[0,384/224-1]
        tensor,_=old.obs_to_tensor(obs)
        old.optimizer.zero_grad()
        (old.unmasked_logits(tensor).sum()+old(tensor)[1].sum()).backward()
        old.optimizer.step()
        self.assertTrue(old.optimizer.state)
        spaces=dict(oldspace.spaces)
        spaces['lasers']=gym.spaces.Box(-np.inf,np.inf,shape=(LASER_CAPACITY,LASER_FIELDS),dtype=np.float32)
        target=migrate_risk_policy(old,gym.spaces.Dict(spaces),lambda _:1e-4,kwargs)
        newobs={**obs,'lasers':encode_lasers([])};newtensor,_=target.obs_to_tensor(newobs)
        torch.testing.assert_close(old.unmasked_logits(tensor),target.unmasked_logits(newtensor))
        self.assertEqual(old.action_mask(tensor).tolist(),target.action_mask(newtensor).tolist())
        for name,oldparam in old.named_parameters():
            if oldparam not in old.optimizer.state:continue
            newparam=dict(target.named_parameters())[name]
            for key,value in old.optimizer.state[oldparam].items():
                candidate=target.optimizer.state[newparam][key]
                if torch.is_tensor(value) and value.ndim==2 and value.shape!=candidate.shape:
                    torch.testing.assert_close(value,candidate[:,:62])
                    self.assertEqual(torch.count_nonzero(candidate[:,62:]).item(),0)
                elif torch.is_tensor(value):torch.testing.assert_close(value,candidate)
