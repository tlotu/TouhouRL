"""One shared risk candidate algorithm for sampling, likelihoods and evaluation."""
import torch

from .masked_policy import BoundaryActorCriticPolicy
from .boundary_control import feasible_actions
from .risk_control import risk_candidates


class RiskActorCriticPolicy(BoundaryActorCriticPolicy):
    def action_mask(self, obs):
        # Masks are non-differentiable state-derived constraints. Reusing the
        # scalar predictor avoids a second, numerically different gate algorithm.
        arrays = {k: v.detach().cpu().float().numpy() for k, v in obs.items()}
        masks = []
        for i in range(len(arrays["player"])):
            state = {k: v[i] for k, v in arrays.items()}
            physical = feasible_actions(state, self.mask_hold_frames)
            allowed = risk_candidates(state, physical)["risk_allowed_actions"]
            masks.append([a in allowed for a in range(9)])
        return torch.tensor(masks, dtype=torch.bool, device=obs["player"].device)
