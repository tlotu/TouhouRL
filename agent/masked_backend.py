"""Source-preserving policy conversion and duration-aware masked PPO updates."""
from copy import deepcopy
import numpy as np
import torch
from .held_backend import HeldPPOBackend
from .masked_policy import BoundaryActorCriticPolicy
from .held_training import HeldTransition
from .training import TrainingError


class MaskedPPOBackend(HeldPPOBackend):
    def enable_mask(self, hold_frames):
        old = self.model.policy
        if isinstance(old, BoundaryActorCriticPolicy):
            if old.mask_hold_frames != hold_frames:
                raise ValueError("Saved policy mask horizon differs")
            return
        kwargs = dict(self.model.policy_kwargs, mask_hold_frames=hold_frames)
        policy = BoundaryActorCriticPolicy(self.model.observation_space, self.model.action_space,
                                           self.model.lr_schedule, **kwargs).to(self.model.device)
        policy.load_state_dict(old.state_dict(), strict=True)
        policy.optimizer.load_state_dict(deepcopy(old.optimizer.state_dict()))
        policy.set_training_mode(False)
        self.model.policy = policy
        self.model.policy_class = BoundaryActorCriticPolicy
        self.model.policy_kwargs = kwargs

    def update(self, transitions, target_timesteps):
        if not isinstance(self.model.policy, BoundaryActorCriticPolicy):
            raise ValueError("Masked PPO requires a masked policy")
        if not transitions or not all(isinstance(r, HeldTransition) for r in transitions):
            raise TrainingError("Masked PPO requires held transitions")
        obs = {k: np.stack([r.observation[k] for r in transitions]) for k in transitions[0].observation}
        tensor, _ = self.model.policy.obs_to_tensor(obs)
        actions = torch.as_tensor([r.action for r in transitions], device=self.model.device)
        with torch.no_grad():
            _, probabilities, _ = self.model.policy.evaluate_actions(tensor, actions)
        old = probabilities.new_tensor([r.log_prob for r in transitions])
        if not torch.isfinite(old).all() or not torch.allclose(old, probabilities, atol=1e-4, rtol=1e-5):
            raise TrainingError("Stored likelihoods differ from the current masked sampling policy")
        return super().update(transitions, target_timesteps)
