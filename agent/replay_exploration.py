"""PPO exploration with the same mixture likelihood at rollout and update."""
import torch
from stable_baselines3.common.policies import ActorCriticPolicy

from .replay_hrl import OPTIONS


class ReplayExplorationPolicy(ActorCriticPolicy):
    def __init__(self, *args, exploration=0., **kwargs):
        self.exploration = exploration
        super().__init__(*args, **kwargs)

    def _exploratory_distribution(self, obs, latent):
        logits = self.action_net(latent)
        # The final field of each option is its normalized first direction.
        requests = obs[:, -OPTIONS * 7 + 6::7]
        nonrule = requests != requests[:, :1]
        earlier = torch.ones((OPTIONS, OPTIONS), dtype=torch.bool, device=obs.device).tril(-1)
        duplicate = ((requests[:, :, None] == requests[:, None, :]) & earlier).any(dim=-1)
        candidates = nonrule & ~duplicate
        total = candidates.sum(1, keepdim=True)
        mixture = self.exploration * (total > 0)
        probs = (1 - mixture) * logits.softmax(dim=-1) + mixture * candidates / total.clamp_min(1)
        return self.action_dist.proba_distribution(action_logits=probs.clamp_min(1e-30).log())

    def forward(self, obs, deterministic=False):
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        dist = (self._get_action_dist_from_latent(latent_pi) if deterministic
                else self._exploratory_distribution(obs, latent_pi))
        actions = dist.get_actions(deterministic=deterministic)
        return actions.reshape((-1, *self.action_space.shape)), self.value_net(latent_vf), dist.log_prob(actions)

    def evaluate_actions(self, obs, actions):
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        dist = self._exploratory_distribution(obs, latent_pi)
        return self.value_net(latent_vf), dist.log_prob(actions), dist.entropy()
