"""Complete-episode held decisions with frame-discounted rewards and GAE."""
from copy import deepcopy
from dataclasses import dataclass
import math

from .training import Transition, TrainingError

PPO_CONTRACT = "held-smdp-v1"
HELD_REWARD_PROFILE = "spell-v2-frame-discounted"
DEFAULT_GAMMA_FRAME = 0.999**0.5
DEFAULT_LAMBDA_FRAME = 0.95**0.5


@dataclass(frozen=True)
class HeldTransition(Transition):
    duration_frames: int


def frame_reward(info, gamma):
    frames = info["delta_frames"]
    if type(frames) is not int or frames < 1 or not 0 < gamma <= 1:
        raise TrainingError("Invalid frame count or frame discount")
    survival = 0.0005 * sum(gamma**i for i in range(frames))
    bonus = 5 if info["reason"] == "spell_success" else -1 if info["reason"] == "spell_failure" else 0
    return survival + gamma**(frames-1) * bonus


def duration_gae(rows, gamma_frame, lambda_frame):
    """Rows end at a true terminal; episode starts prevent cross-episode traces."""
    if not rows or not 0 < gamma_frame <= 1 or not 0 < lambda_frame <= 1 or not rows[0].episode_start:
        raise TrainingError("Invalid held GAE input")
    advantages = [0.] * len(rows)
    following = 0.
    for i in reversed(range(len(rows))):
        row = rows[i]
        if type(row.duration_frames) is not int or row.duration_frames < 1:
            raise TrainingError("Invalid held duration")
        continuation = i+1 < len(rows) and not rows[i+1].episode_start
        next_value = rows[i+1].value if continuation else 0.
        discount = gamma_frame**row.duration_frames
        delta = row.reward + discount*next_value - row.value
        following = delta + discount * lambda_frame**row.duration_frames * following * continuation
        if not math.isfinite(following):
            raise TrainingError("Nonfinite held advantage")
        advantages[i] = following
    return advantages, [a+r.value for a, r in zip(advantages, rows)]


def collect_held_episodes(env, learner, minimum_steps, emit, hold_frames=4,
                          gamma_frame=DEFAULT_GAMMA_FRAME):
    if (minimum_steps < 2 or type(hold_frames) is not int or not 2 <= hold_frames <= 16
            or hold_frames % 2 or not 0 < gamma_frame <= 1):
        raise TrainingError("Invalid held collection budget")
    rows = []
    episodes = successes = total_frames = total_substeps = 0
    while len(rows) < minimum_steps:
        obs, _ = env.reset()
        start = True
        episode_reward = env_return = 0.
        episode_frames = episode_decisions = episode_substeps = 0
        while True:
            before = deepcopy(obs)
            decision = learner.act(obs)
            if (type(decision.action) is not int or not 0 <= decision.action < 9
                    or not math.isfinite(decision.value) or not math.isfinite(decision.log_prob)):
                raise TrainingError("Invalid sampled held decision")
            duration = 0
            reward = 0.
            while duration < hold_frames:
                obs, diagnostic_reward, terminated, truncated, info = env.step(decision.action)
                if truncated or (terminated and info["reason"] not in ("spell_success", "spell_failure")):
                    raise TrainingError("Held collection requires recognized complete episodes")
                frames = info["delta_frames"]
                reward += gamma_frame**duration * frame_reward(info, gamma_frame)
                duration += frames
                env_return += diagnostic_reward
                total_substeps += 1
                episode_substeps += 1
                if terminated:
                    break
            rows.append(HeldTransition(before, decision.action, reward, start,
                                       decision.value, decision.log_prob, duration))
            start = False
            total_frames += duration
            episode_frames += duration
            episode_decisions += 1
            episode_reward += reward
            emit({"event": "held_decision", "episode": episodes, "decision": episode_decisions,
                  "action": decision.action, "log_prob": decision.log_prob, "value": decision.value,
                  "reward": reward, "duration_frames": duration, "terminated": terminated})
            if terminated:
                episodes += 1
                successes += int(info["reason"] == "spell_success")
                emit({"event": "training_episode", "episode": episodes, "training_return": episode_reward,
                      "environment_return": env_return, "episode_decisions": episode_decisions,
                      "episode_substeps": episode_substeps, "episode_frames": episode_frames,
                      "batch_steps": len(rows), **info})
                break
    return rows, {"episodes": episodes, "successes": successes, "logic_frames": total_frames,
                  "substeps": total_substeps}
