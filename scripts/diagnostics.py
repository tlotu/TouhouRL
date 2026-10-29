"""Repeatable episode checks and append-free JSONL evidence."""
import json
from pathlib import Path
import sys
from agent.baselines import make_policy


class DiagnosticLog:
    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.file = None

    def __enter__(self):
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            requested = self.path
            for index in range(10000):
                candidate = (requested if index == 0 else
                             requested.with_name(f"{requested.stem}-{index:03d}{requested.suffix}"))
                try:
                    self.file = candidate.open("x", encoding="utf-8")
                except FileExistsError:
                    continue
                self.path = candidate
                break
            else:
                raise FileExistsError(f"No unused log name available for {requested}")
            print(f"Diagnostic log: {self.path}", file=sys.stderr)
        return self

    def emit(self, record, console=True):
        line = json.dumps(record, allow_nan=False)
        if self.file:
            self.file.write(line + "\n")
            self.file.flush()
        if console:
            print(line, flush=True)

    def __exit__(self, kind, error, traceback):
        try:
            if error is not None:
                self.emit({"event": "error", "type": type(error).__name__, "message": str(error)})
        finally:
            if self.file:
                self.file.close()


def run_episodes(session, episodes, emit, seed=0, policy="random", trace=False, action_fn=None):
    """Report every reset and terminal event, including partially failed runs."""
    choose = make_policy(policy, seed) if action_fn is None else action_fn
    resets = completed = failures = successes = classified_failures = 0
    total_frames = 0
    reason = None
    transitions = fallback_steps = 0
    max_bullets = max_dropped = max_fallback = 0
    try:
        for episode in range(episodes):
            obs, info = session.reset()
            resets += 1
            emit({"event": "reset", "episode": episode, **info})
            steps = 0
            episode_return = 0.0
            while True:
                action = choose(obs)
                next_obs, reward, terminated, truncated, info = session.step(action)
                transitions += 1
                max_bullets = max(max_bullets, info.get("bullet_count", 0))
                max_dropped = max(max_dropped, info.get("dropped_bullets", 0))
                fallback_count = info.get("radius_fallback_count", 0)
                max_fallback = max(max_fallback, fallback_count)
                fallback_steps += int(fallback_count > 0)
                if trace:
                    emit({"event": "transition", "episode": episode, "step": steps,
                          "observation": obs, "action": action, "reward": reward,
                          "next_observation": next_obs, "terminated": terminated,
                          "truncated": truncated, "info": info})
                obs = next_obs
                steps += 1
                episode_return += reward
                if steps % 30 == 0 or terminated or truncated:
                    emit({"event": "step", "episode": episode, "action": action,
                          "reward": reward, "terminated": terminated, "truncated": truncated, **info})
                if terminated or truncated:
                    completed += 1
                    reason = info["reason"]
                    failures += int(reason in ("death", "spell_failure"))
                    successes += int(reason == "spell_success")
                    classified_failures += int(reason == "spell_failure")
                    total_frames += info["elapsed_frames"]
                    emit({"event": "episode_end", "episode": episode, "steps": steps,
                          "return": episode_return, **info})
                    if reason == "quiet_after_combat":
                        emit({"event": "manual_review", "episode": episode,
                              "message": "Combat became quiet. Check the result screen; success is not inferred. "
                                         "Select the spell manually before starting a new run."})
                    break
            if reason not in ("death", "spell_success", "spell_failure"):
                break
    finally:
        emit({"event": "summary", "requested_episodes": episodes, "resets": resets,
              "policy": policy, "transitions": transitions,
              "max_bullet_count": max_bullets, "max_dropped_bullets": max_dropped,
              "radius_fallback_steps": fallback_steps, "max_radius_fallback_count": max_fallback,
              "completed_episodes": completed, "death_or_retry_episodes": failures,
              "successful_episodes": successes, "classified_failure_episodes": classified_failures,
              "unresolved_episodes": completed - failures - successes,
              "success_rate": successes / completed if completed and successes + classified_failures == completed else None,
              "observed_retries": max(0, resets - 1), "completed_frames": total_frames,
              "all_requested_completed": completed == episodes, "last_reason": reason})
