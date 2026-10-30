"""SB3 PPO optimizer with a complete-episode, single-instance collector.

Do not call model.learn(): its vector wrapper resets terminal games immediately.
Only buffer construction is customized; SB3 implements PPO and GAE.
"""
import math
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import DictRolloutBuffer
from stable_baselines3.common.logger import Logger

from .training import Decision


class PPOBackend:
    def __init__(self, env, seed=0, rollout_steps=512, checkpoint=None, features="flat"):
        torch.set_num_threads(1)
        if checkpoint:
            self.model = PPO.load(checkpoint, env=env, device="cpu")
        else:
            policy_kwargs = {"net_arch": {"pi": [128, 128], "vf": [128, 128]}}
            if features == "hazard-v1":
                from .hazard_features import HazardFeatures
                policy_kwargs["features_extractor_class"] = HazardFeatures
            elif features != "flat":
                raise ValueError("Unknown feature extractor")
            self.model = PPO("MultiInputPolicy", env, device="cpu", seed=seed,
                             n_steps=rollout_steps, batch_size=64, n_epochs=4,
                             learning_rate=3e-4, gamma=0.999, gae_lambda=0.95,
                             clip_range=0.2, ent_coef=0.01, target_kl=0.03,
                             policy_kwargs=policy_kwargs)
        self.model.set_logger(Logger(folder=None, output_formats=[]))
        self.model.policy.set_training_mode(False)
        # Warm up tensor/inference initialization before attaching a live episode.
        self.act({k: np.zeros(s.shape, dtype=s.dtype) for k, s in env.observation_space.spaces.items()})

    @property
    def timesteps(self):
        return self.model.num_timesteps

    def act(self, obs):
        self.model.policy.set_training_mode(False)
        tensor, _ = self.model.policy.obs_to_tensor(obs)
        with torch.no_grad():
            action, value, log_prob = self.model.policy(tensor)
        return Decision(int(action.item()), float(value.item()), float(log_prob.item()))

    def update(self, transitions, target_timesteps):
        model = self.model
        buffer = DictRolloutBuffer(len(transitions), model.observation_space, model.action_space,
                                   device=model.device, gamma=model.gamma,
                                   gae_lambda=model.gae_lambda, n_envs=1)
        for row in transitions:
            buffer.add(row.observation, np.array([row.action]), np.array([row.reward]),
                       np.array([row.episode_start]), torch.tensor([row.value]),
                       torch.tensor([row.log_prob]))
        # Every collected episode is truly terminal; no artificial truncations.
        buffer.compute_returns_and_advantage(torch.zeros(1, device=model.device), np.array([True]))
        model.rollout_buffer = buffer
        model.num_timesteps += len(transitions)
        model._current_progress_remaining = max(0.0, 1 - model.num_timesteps / target_timesteps)
        model.train()
        metrics = {}
        for key, value in model.logger.name_to_value.items():
            if isinstance(value, (int, float, np.number)):
                number = float(value)
                if not math.isfinite(number):
                    if key == "train/explained_variance":
                        metrics[key] = None  # Undefined when target variance is zero.
                        continue
                    raise RuntimeError(f"Non-finite PPO metric: {key}")
                metrics[key] = number
        if any(not torch.isfinite(parameter).all().item() for parameter in model.policy.parameters()):
            raise RuntimeError("Non-finite PPO parameters; checkpoint not saved")
        model.policy.set_training_mode(False)
        return metrics

    def deterministic_action(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return int(np.asarray(action).item())

    def save(self, path):
        # SB3 includes policy and optimizer state; native game state is excluded.
        self.model.save(path)
