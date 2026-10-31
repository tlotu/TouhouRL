"""Validate teacher traces and split complete episodes before fitting."""
import hashlib
import json
import random
from pathlib import Path

import numpy as np

from env.session import OBSERVATION_SCHEMA, ENEMY_CAPACITY


def _read_lines(path):
    with path.open("rb") as stream:
        yield from stream


def validate_observation(obs, capacity):
    shapes = {"player": (4,), "bullets": (capacity, 7), "enemies": (ENEMY_CAPACITY, 6)}
    if not isinstance(obs, dict) or set(obs) != set(shapes):
        raise ValueError("Invalid demonstration observation keys")
    result = {}
    for key, shape in shapes.items():
        value = np.asarray(obs[key], dtype=np.float32)
        if value.shape != shape or not np.isfinite(value).all() or (np.abs(value) > 1).any():
            raise ValueError(f"Invalid demonstration observation: {key}")
        result[key] = value
    return result


def load_demonstrations(paths, include_failures=False):
    """Drop incomplete tails; reject malformed, mismatched, or misaligned data."""
    episodes, sources = [], []
    identity = None
    discarded = failures = 0
    for path in map(Path, paths):
        digest = hashlib.sha256()
        header = pending = None
        correction = False
        ended = False
        for line_number, raw in enumerate(_read_lines(path), 1):
            digest.update(raw)
            try:
                row = json.loads(raw)
                event = row["event"]
                if event == "connected":
                    if header is not None:
                        raise ValueError("Repeated trace header")
                    header = row
                    correction = row.get("collection") == "corrections"
                    valid_policy = (row.get("policy") == "mixed_teacher_student"
                                    and row.get("teacher") == "reactive_attack"
                                    and 0 <= row.get("teacher_probability", -1) <= 1
                                    and bool(row.get("student_checkpoint"))) if correction else row.get("policy") == "reactive_attack"
                    if (row.get("schema") != OBSERVATION_SCHEMA or not valid_policy
                            or row.get("repeat") != 2 or row.get("trace") is not True
                            or row.get("run") is not True or not row.get("task")
                            or not row.get("build_md5") or not 1 <= row["capacity"] <= 640):
                        raise ValueError("Requires a labeled state-v2 reactive_attack trace with repeat=2")
                    current = {k: row[k] for k in ("schema", "task", "build_md5", "capacity", "max_frames")}
                    if identity is not None and current != identity:
                        raise ValueError("Demonstration task/build/config mismatch")
                    identity = current
                elif event == "reset":
                    if header is None:
                        raise ValueError("Missing trace header")
                    if pending is not None:
                        discarded += 1
                    pending = {"id": f"{path.resolve()}:{row['episode']}", "episode": row["episode"],
                               "observations": [], "actions": [], "disagreements": [], "frames": []}
                    ended = False
                elif event == "transition":
                    if pending is None or ended or row["episode"] != pending["episode"]:
                        raise ValueError("Transition outside an active episode")
                    if row["step"] != len(pending["actions"]):
                        raise ValueError("Noncontiguous transition sequence")
                    executed = row["action"]
                    action = row["teacher_action"] if correction else executed
                    if type(executed) is not int or not 0 <= executed < 9:
                        raise ValueError("Invalid executed action")
                    if correction:
                        student, source = row["student_action"], row["executed_by"]
                        if (type(student) is not int or not 0 <= student < 9
                                or source not in ("teacher", "student")
                                or executed != (action if source == "teacher" else student)):
                            raise ValueError("Invalid correction action provenance")
                    if type(action) is not int or not 0 <= action < 9:
                        raise ValueError("Invalid teacher action")
                    obs = validate_observation(row["observation"], identity["capacity"])
                    from .baselines import reactive_attack
                    if reactive_attack(row["observation"]) != action:
                        raise ValueError("Teacher action does not match its observation")
                    pending["observations"].append(obs)
                    pending["actions"].append(action)
                    pending["disagreements"].append(bool(correction and row["student_action"] != action))
                    pending["frames"].append(row["info"].get("elapsed_frames", 2 * (row["step"] + 1)))
                    ended = bool(row["terminated"] or row["truncated"])
                    pending["reason"] = row["info"]["reason"] if ended else None
                    pending["terminal"] = bool(row["terminated"] and not row["truncated"])
                elif event == "episode_end":
                    if (pending is None or not ended or row["episode"] != pending["episode"]
                            or row["steps"] != len(pending["actions"])
                            or row["reason"] != pending["reason"]):
                        raise ValueError("Unverified episode boundary")
                    reason = row["reason"]
                    if pending["terminal"] and (reason == "spell_success" or
                                                ((include_failures or correction) and reason == "spell_failure")):
                        pending["reason"] = reason
                        episodes.append(pending)
                    else:
                        failures += 1
                    pending = None
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
        if header is None:
            raise ValueError(f"Missing trace header: {path}")
        discarded += int(pending is not None)
        sources.append({"path": str(path.resolve()), "sha256": digest.hexdigest(),
                        "collection": "corrections" if correction else "teacher",
                        "student_checkpoint": header.get("student_checkpoint"),
                        "teacher_probability": header.get("teacher_probability")})
    if len({ep["id"] for ep in episodes}) != len(episodes):
        raise ValueError("Duplicate demonstration episodes")
    if len({s["sha256"] for s in sources}) != len(sources):
        raise ValueError("Duplicate demonstration files")
    return episodes, {"identity": identity, "sources": sources, "incomplete_episodes": discarded,
                      "excluded_episodes": failures, "include_failures": include_failures}


def split_episodes(episodes, seed=0, fraction=0.2):
    if len(episodes) < 2 or not 0 < fraction < 1:
        raise ValueError("Need at least two complete eligible episodes for an episode split")
    shuffled = list(episodes)
    random.Random(seed).shuffle(shuffled)
    count = max(1, min(len(shuffled) - 1, round(len(shuffled) * fraction)))
    return shuffled[count:], shuffled[:count]


def stack_episodes(episodes):
    observations = [obs for episode in episodes for obs in episode["observations"]]
    return ({key: np.stack([obs[key] for obs in observations]) for key in observations[0]},
            np.asarray([action for episode in episodes for action in episode["actions"]], dtype=np.int64))


def deduplicate_episodes(episodes):
    """Keep exact observation/label trajectories in a single split group."""
    kept, signatures, duplicates = [], {}, []
    for episode in episodes:
        digest = hashlib.sha256()
        for obs, action in zip(episode["observations"], episode["actions"]):
            for key in sorted(obs):
                digest.update(obs[key].tobytes())
            digest.update(bytes([action]))
        signature = digest.hexdigest()
        if signature in signatures:
            duplicates.append({"kept": signatures[signature], "removed": episode["id"]})
        else:
            signatures[signature] = episode["id"]
            kept.append(episode)
    return kept, duplicates


def imitation_weights(episodes, profile="uniform"):
    """Bounded emphasis on disagreements, especially near failed endings."""
    if profile not in ("uniform", "correction-v1"):
        raise ValueError("Unknown imitation weighting")
    weights = []
    for episode in episodes:
        end = episode["frames"][-1]
        for disagreement, frame in zip(episode["disagreements"], episode["frames"]):
            value = 1.0
            if profile == "correction-v1" and disagreement:
                value = 20.0 if episode["reason"] == "spell_failure" and end - frame <= 64 else 3.0
            weights.append(value)
    return np.asarray(weights, dtype=np.float32)
