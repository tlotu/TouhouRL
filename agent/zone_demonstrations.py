"""Explicitly validate held attack-zone traces and retain fresh decision labels."""
import hashlib
import json
from pathlib import Path

import numpy as np

from .action_commitment import ActionCommitment
from .attack_zone import AttackZoneConfig, rank_attack_zone
from .demonstrations import validate_observation, split_episodes
from env.session import OBSERVATION_SCHEMA
from env.movement_probe import BASELINE_BUILD


CONTRACT = "attack-zone-held-v1"


def load_zone_demonstrations(paths, include_failures=False):
    episodes, sources, identity = [], [], None
    discarded = excluded = 0
    for path in map(Path, paths):
        data = path.read_bytes()
        header = pending = previous_next = None
        elapsed = count = 0
        ended = False
        kept_decisions = 0
        for line_number, line in enumerate(data.splitlines(), 1):
            try:
                row = json.loads(line)
                event = row["event"]
                if event == "connected":
                    if header is not None:
                        raise ValueError("Repeated header")
                    header = row
                    if (row.get("policy") != "attack_zone" or row.get("collection") != "attack_zone_evaluation"
                            or row.get("schema") != OBSERVATION_SCHEMA or row.get("repeat") != 2
                            or row.get("run") is not True or row.get("trace") is not True
                            or row.get("prediction_frames") != 12 or row.get("aim_projection_frames") != 4
                            or not row.get("task") or row.get("build_md5") != BASELINE_BUILD
                            or type(row.get("capacity")) is not int or not 1 <= row["capacity"] <= 640):
                        raise ValueError("Requires a complete labeled attack-zone evaluation trace")
                    config = AttackZoneConfig(**row["attack_zone"])
                    current = {k: row[k] for k in ("schema", "task", "build_md5", "capacity", "max_frames", "hold_frames", "attack_zone")}
                    current["decision_contract"] = CONTRACT
                    if identity is not None and current != identity:
                        raise ValueError("Zone task, teacher or control settings differ")
                    identity = current
                    holding = ActionCommitment(lambda obs: rank_attack_zone(obs, config)["action"],
                                               lambda: elapsed, row["hold_frames"])
                elif event == "reset":
                    if header is None:
                        raise ValueError("Missing header")
                    discarded += int(pending is not None)
                    elapsed = count = 0
                    previous_next = None
                    ended = False
                    holding.annotate(row)
                    pending = {"id": f"{path.resolve()}:{row['episode']}", "episode": row["episode"],
                               "observations": [], "actions": [], "frames": [], "disagreements": []}
                elif event == "transition":
                    if pending is None or ended or row["episode"] != pending["episode"] or row["step"] != count:
                        raise ValueError("Noncontiguous zone transition")
                    obs = validate_observation(row["observation"], identity["capacity"])
                    next_obs = validate_observation(row["next_observation"], identity["capacity"])
                    if previous_next is not None and any(not np.array_equal(obs[k], previous_next[k]) for k in obs):
                        raise ValueError("Observation chain is broken")
                    previous_next = next_obs
                    ranking = rank_attack_zone(row["observation"], config)
                    expected_action = holding(row["observation"])
                    expected = holding.annotate({"event": "transition", "action": expected_action})
                    if (row["rule_ranking"] != ranking or row["raw_rule_action"] != ranking["action"]
                            or type(row["action"]) is not int or row["action"] != expected_action
                            or any(row[k] != expected[k] for k in
                                   ("committed_action", "commitment_active", "action_overridden", "hold_age_frames"))):
                        raise ValueError("Rule label or held action provenance is invalid")
                    # Require evidence of intended slow input on playable transitions.
                    if row["info"].get("player_x") is not None and not (row["terminated"] or row["truncated"]):
                        telemetry = row["movement_diagnostic"]
                        candidate = telemetry["game_candidates"]
                        if (telemetry.get("post_step_cleanup") or telemetry["requested_slow"] is not True
                                or "shift" not in telemetry["owned_keys"] or telemetry["os_shift"]["shift"] is not True
                                or not candidate.get("available") or not candidate.get("plausible")
                                or not candidate.get("same_transition_frame") or not candidate.get("slow_input_bit")
                                or candidate.get("focus_raw") != 1
                                or candidate.get("frame_before") != row["info"]["frame"]
                                or candidate.get("frame_after") != row["info"]["frame"]
                                or not (candidate["input_mask"] & 4)
                                or abs(candidate["speed_parameters"]["slow"] - 2) > 1e-4
                                or abs(candidate["speed_parameters"]["diagonal_slow_axis"] - 2**0.5) > 1e-4):
                            raise ValueError("Slow-mode telemetry is missing, stale or inconsistent")
                    frames = row["info"]["delta_frames"]
                    if type(frames) is not int or frames < 1 or frames > 32 or row["info"]["elapsed_frames"] != elapsed + frames:
                        raise ValueError("Invalid elapsed frame sequence")
                    if not expected["commitment_active"]:
                        pending["observations"].append(obs)
                        pending["actions"].append(ranking["action"])
                        pending["frames"].append(elapsed)
                        pending["disagreements"].append(False)
                        kept_decisions += 1
                    elapsed += frames
                    count += 1
                    ended = bool(row["terminated"] or row["truncated"])
                    pending["reason"] = row["info"]["reason"] if ended else None
                    pending["terminal"] = bool(row["terminated"] and not row["truncated"])
                elif event == "episode_end":
                    if (pending is None or not ended or row["episode"] != pending["episode"]
                            or row["steps"] != count or row["reason"] != pending["reason"]):
                        raise ValueError("Unverified zone episode boundary")
                    if pending["terminal"] and (row["reason"] == "spell_success" or
                            include_failures and row["reason"] == "spell_failure"):
                        episodes.append(pending)
                    else:
                        excluded += 1
                    pending = None
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
        if header is None:
            raise ValueError("Missing zone trace header")
        discarded += int(pending is not None)
        sources.append({"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest(),
                        "collection": "validated_attack_zone_decisions", "original_collection": header["collection"],
                        "fresh_decisions_seen": kept_decisions})
    if len({s["sha256"] for s in sources}) != len(sources) or len({e["id"] for e in episodes}) != len(episodes):
        raise ValueError("Duplicate zone demonstration files or episodes")
    return episodes, {"identity": identity, "sources": sources, "incomplete_episodes": discarded,
                      "excluded_episodes": excluded, "include_failures": include_failures,
                      "label_contract": "fresh decisions only; held intermediate frames excluded"}


def load_multitask_zone_demonstrations(paths, include_failures=False):
    """Validate each original task before combining compatible held-zone data."""
    episodes, sources, tasks = [], [], set()
    shared = None
    incomplete = excluded = 0
    for path in paths:
        current, provenance = load_zone_demonstrations([path], include_failures)
        identity = dict(provenance["identity"])
        task = identity.pop("task")
        if shared is not None and identity != shared:
            raise ValueError("Multi-task zone teacher, build or control settings differ")
        shared = identity
        tasks.add(task)
        episodes.extend({**ep, "task": task} for ep in current)
        sources.extend({**source, "task": task} for source in provenance["sources"])
        incomplete += provenance["incomplete_episodes"]
        excluded += provenance["excluded_episodes"]
    if len(tasks) < 2:
        raise ValueError("Multi-task cloning requires at least two declared tasks")
    if len({s["sha256"] for s in sources}) != len(sources):
        raise ValueError("Duplicate multi-task demonstration files")
    return episodes, {"identity": {**shared, "tasks": sorted(tasks)}, "sources": sources,
                      "dataset_contract": "attack-zone-multitask-v1",
                      "incomplete_episodes": incomplete, "excluded_episodes": excluded,
                      "include_failures": include_failures,
                      "label_contract": "fresh decisions only; original task per episode and source"}


def split_zone_tasks(episodes, tasks, seed=0):
    """Hold out complete episodes within every declared task."""
    groups = {task: [] for task in tasks}
    for episode in episodes:
        groups[episode["task"]].append(episode)
    train, validation, manifest = [], [], {}
    for task in sorted(groups):
        if len(groups[task]) < 2:
            raise ValueError(f"Task {task!r} needs two unique eligible episodes after deduplication")
        task_train, task_validation = split_episodes(groups[task], seed)
        train.extend(task_train)
        validation.extend(task_validation)
        manifest[task] = {"train_episode_ids": [ep["id"] for ep in task_train],
                          "validation_episode_ids": [ep["id"] for ep in task_validation],
                          "train_samples": sum(len(ep["actions"]) for ep in task_train),
                          "validation_samples": sum(len(ep["actions"]) for ep in task_validation)}
    return train, validation, manifest
