"""Complete-episode collection; updates never reset or advance the game."""
from copy import deepcopy
from dataclasses import dataclass
import math
import time


REWARD_PROFILE = "spell-v2"


class TrainingError(RuntimeError):
    pass


@dataclass(frozen=True)
class Decision:
    action: int
    value: float
    log_prob: float


@dataclass(frozen=True)
class Transition:
    observation: dict
    action: int
    reward: float
    episode_start: bool
    value: float
    log_prob: float


def training_reward(info):
    """Reward survival and verified spell completion."""
    frames = info["delta_frames"]
    if not isinstance(frames, int) or frames < 0:
        raise TrainingError("Invalid elapsed frame count")
    bonus = 5.0 if info["reason"] == "spell_success" else -1.0 if info["reason"] == "spell_failure" else 0.0
    return bonus + 0.0005 * frames


def collect_episodes(env, learner, minimum_steps, emit):
    """Keep one fixed policy until a batch ends at a real spell result."""
    if minimum_steps < 2:
        raise ValueError("A PPO batch requires at least two transitions")
    transitions = []
    episodes = successes = frames = 0
    while len(transitions) < minimum_steps:
        obs, _ = env.reset()
        start = True
        total_reward = env_return = 0.0
        while True:
            decision = learner.act(obs)
            if not (0 <= decision.action < 9 and math.isfinite(decision.value)
                    and math.isfinite(decision.log_prob)):
                raise TrainingError("Policy produced an invalid decision")
            before = deepcopy(obs)
            obs, env_reward, terminated, truncated, info = env.step(decision.action)
            if truncated or (terminated and info["reason"] not in ("spell_success", "spell_failure")):
                raise TrainingError(f"Unrecognized training ending: {info['reason']}; batch discarded")
            reward = training_reward(info)
            transitions.append(Transition(before, decision.action, reward, start,
                                          decision.value, decision.log_prob))
            start = False
            total_reward += reward
            env_return += env_reward
            frames += info["delta_frames"]
            if terminated:
                episodes += 1
                successes += int(info["reason"] == "spell_success")
                emit({"event": "training_episode", "episode": episodes,
                      "training_return": total_reward, "environment_return": env_return,
                      "batch_steps": len(transitions), **info})
                break
    return transitions, {"episodes": episodes, "successes": successes, "logic_frames": frames}


def train_episodic(env, learner, boundary, checkpoint, emit, additional_steps,
                   rollout_steps=512, completed_updates=0, clock=time.monotonic):
    if additional_steps < 2 or rollout_steps < 2:
        raise ValueError("Step budgets must be at least two")
    start_steps = learner.timesteps
    target = start_steps + additional_steps
    updates = completed_updates
    while learner.timesteps < target:
        started = clock()
        count = max(2, min(rollout_steps, target - learner.timesteps))
        batch, stats = collect_episodes(env, learner, count,
                                       lambda row: emit({"update": updates + 1, **row}))
        sampled = clock()
        boundary()
        metrics = learner.update(batch, target)
        updated = clock()
        boundary()
        updates += 1
        state = {"updates": updates, "timesteps": learner.timesteps}
        saved = checkpoint(state)
        emit({"event": "ppo_update", **state, **stats, "batch_steps": len(batch),
              "sample_seconds": sampled - started, "update_seconds": updated - sampled,
              "sampling_steps_per_second": len(batch) / max(sampled - started, 1e-9),
              "metrics": metrics, "checkpoint": str(saved)})
    emit({"event": "training_complete", "timesteps": learner.timesteps,
          "additional_steps": learner.timesteps - start_steps, "updates": updates})
