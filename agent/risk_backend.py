"""Explicit migration from physical-mask PPO, preserving learned tensors and Adam."""
from copy import deepcopy

from .masked_backend import MaskedPPOBackend
from .masked_policy import BoundaryActorCriticPolicy
from .risk_policy import RiskActorCriticPolicy
from .risk_control import canonical_risk_observation, risk_candidates
from .boundary_control import feasible_actions


class RiskPPOBackend(MaskedPPOBackend):
    def act(self, obs, deterministic=False):
        decision = super().act(obs, deterministic=deterministic)
        self.decision_context = {}
        if isinstance(self.model.policy, RiskActorCriticPolicy):
            state = canonical_risk_observation(obs)
            allowed = feasible_actions(state, self.model.policy.mask_hold_frames)
            self.decision_context = risk_candidates(state, allowed)
            self.decision_context["player"] = [float(v) for v in state["player"]]
            if decision.action not in self.decision_context["risk_allowed_actions"]:
                raise RuntimeError("Sampled action differs from its risk candidates")
        return decision

    def enable_mask(self, hold_frames):
        old = self.model.policy
        if isinstance(old, RiskActorCriticPolicy):
            if old.mask_hold_frames != hold_frames:
                raise ValueError("Saved risk mask horizon differs")
            return
        if type(old) is not BoundaryActorCriticPolicy or old.mask_hold_frames != hold_frames:
            raise ValueError("Risk migration requires a matching physical-mask policy")
        kwargs = dict(self.model.policy_kwargs)
        policy = RiskActorCriticPolicy(self.model.observation_space, self.model.action_space,
                                      self.model.lr_schedule, **kwargs).to(self.model.device)
        policy.load_state_dict(old.state_dict(), strict=True)
        policy.optimizer.load_state_dict(deepcopy(old.optimizer.state_dict()))
        policy.set_training_mode(False)
        self.model.policy, self.model.policy_class = policy, RiskActorCriticPolicy
        self.model.policy_kwargs = kwargs
