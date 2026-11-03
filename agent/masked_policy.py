"""State-derived boundary masks shared by sampling and PPO likelihoods."""
import math
import torch
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.distributions import CategoricalDistribution
from env.action_emulator import DIRECTIONS


class BoundaryActorCriticPolicy(MultiInputActorCriticPolicy):
    def __init__(self, *args, mask_hold_frames=4, **kwargs):
        if type(mask_hold_frames) is not int or not 2 <= mask_hold_frames <= 16 or mask_hold_frames % 2:
            raise ValueError("Invalid masked policy horizon")
        self.mask_hold_frames = mask_hold_frames
        super().__init__(*args, **kwargs)
        if self.action_space.n != 9:
            raise ValueError("Boundary policy requires nine discrete actions")

    def action_mask(self, obs):
        p = obs["player"].double()
        x, y = (p[:, 0] + 1) * 192, (p[:, 1] + 1) * 224
        inside = torch.isfinite(p[:, :2]).all(1) & (x >= 8-1e-4) & (x <= 376+1e-4) & (y >= 16-1e-4) & (y <= 432+1e-4)
        if not inside.all():
            raise ValueError("Player position outside masked policy bounds")
        velocities = []
        for keys in DIRECTIONS:
            dx = int("right" in keys)-int("left" in keys)
            dy = int("down" in keys)-int("up" in keys)
            speed = 2/max(1., math.hypot(dx, dy))
            velocities.append((dx*speed, dy*speed))
        velocity = p.new_tensor(velocities) * self.mask_hold_frames
        ex, ey = x[:, None]+velocity[:, 0], y[:, None]+velocity[:, 1]
        return (ex >= 8-1e-4) & (ex <= 376+1e-4) & (ey >= 16-1e-4) & (ey <= 432+1e-4)

    def _latents(self, obs):
        features = self.extract_features(obs)
        if self.share_features_extractor:
            return self.mlp_extractor(features)
        return (self.mlp_extractor.forward_actor(features[0]),
                self.mlp_extractor.forward_critic(features[1]))

    def unmasked_logits(self, obs):
        pi, _ = self._latents(obs)
        return self.action_net(pi)

    def _masked_distribution(self, obs, pi):
        logits = self.action_net(pi).masked_fill(~self.action_mask(obs), -torch.inf)
        return CategoricalDistribution(9).proba_distribution(action_logits=logits)

    def forward(self, obs, deterministic=False):
        pi, vf = self._latents(obs)
        dist = self._masked_distribution(obs, pi)
        actions = dist.get_actions(deterministic=deterministic)
        return actions.reshape((-1, *self.action_space.shape)), self.value_net(vf), dist.log_prob(actions)

    def evaluate_actions(self, obs, actions):
        if not self.action_mask(obs).gather(1, actions.long().reshape(-1, 1)).all():
            raise ValueError("Rollout contains an infeasible sampled action")
        pi, vf = self._latents(obs)
        dist = self._masked_distribution(obs, pi)
        return self.value_net(vf), dist.log_prob(actions), dist.entropy()

    def get_distribution(self, obs):
        pi, _ = self._latents(obs)
        return self._masked_distribution(obs, pi)
