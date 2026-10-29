"""Repeatable episode checks and append-free JSONL evidence."""
import json
from pathlib import Path
import random


class DiagnosticLog:
    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.file = None

    def __enter__(self):
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.file = self.path.open("x", encoding="utf-8")
        return self

    def emit(self, record):
        line = json.dumps(record, allow_nan=False)
        if self.file:
            self.file.write(line + "\n")
            self.file.flush()
        print(line, flush=True)

    def __exit__(self, kind, error, traceback):
        try:
            if error is not None:
                self.emit({"event": "error", "type": type(error).__name__, "message": str(error)})
        finally:
            if self.file:
                self.file.close()


def run_episodes(session, episodes, emit, seed=0):
    """Report every reset and terminal event, including partially failed runs."""
    rng = random.Random(seed)
    resets = completed = failures = 0
    total_frames = 0
    reason = None
    try:
        for episode in range(episodes):
            _, info = session.reset()
            resets += 1
            emit({"event": "reset", "episode": episode, **info})
            steps = 0
            episode_return = 0.0
            while True:
                action = rng.randrange(9)
                _, reward, terminated, truncated, info = session.step(action)
                steps += 1
                episode_return += reward
                if steps % 30 == 0 or terminated or truncated:
                    emit({"event": "step", "episode": episode, "action": action,
                          "reward": reward, "terminated": terminated, "truncated": truncated, **info})
                if terminated or truncated:
                    completed += 1
                    reason = info["reason"]
                    failures += int(reason == "death")
                    total_frames += info["elapsed_frames"]
                    emit({"event": "episode_end", "episode": episode, "steps": steps,
                          "return": episode_return, **info})
                    break
            if reason != "death":
                break
    finally:
        emit({"event": "summary", "requested_episodes": episodes, "resets": resets,
              "completed_episodes": completed, "failed_episodes": failures,
              "observed_retries": max(0, resets - 1), "completed_frames": total_frames,
              "all_requested_completed": completed == episodes, "last_reason": reason})
