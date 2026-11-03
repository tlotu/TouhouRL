"""Masked held collection; death and non-death native failure stay distinct."""
from .boundary_control import feasible_actions
from .held_training import collect_held_episodes
from .training import TrainingError


def survival_reward(info, gamma):
    frames = info["delta_frames"]
    if type(frames) is not int or frames < 1 or not 0 < gamma <= 1:
        raise TrainingError("Invalid masked reward duration")
    bonus = (5 if info["reason"] == "spell_success" else
             -5 if info["reason"] == "spell_failure" and info.get("deaths", 0) > 0 else
             -1 if info["reason"] == "spell_failure" else 0)
    return .0005*sum(gamma**i for i in range(frames)) + gamma**(frames-1)*bonus


def collect_masked_episodes(env, learner, minimum_steps, emit, hold_frames=4, **kwargs):
    if learner.model.policy.mask_hold_frames != hold_frames:
        raise TrainingError("Collector and policy mask horizons differ")
    return collect_held_episodes(env, learner, minimum_steps, emit, hold_frames=hold_frames,
                                 reward_fn=survival_reward,
                                 action_feasible=lambda obs, a, n: a in feasible_actions(obs, n), **kwargs)
