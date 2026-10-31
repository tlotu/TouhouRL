"""Inspect full traces offline; motion fits do not identify engine input latency."""
import argparse
import json
import math
from pathlib import Path
import statistics

from agent.baselines import action_candidates
from env.action_emulator import DIRECTIONS


def velocity(action):
    keys = DIRECTIONS[action]
    x = int("right" in keys) - int("left" in keys)
    y = int("down" in keys) - int("up" in keys)
    scale = 2 / max(1, math.hypot(x, y))
    return x * scale, y * scale


def position(obs):
    return [(obs["player"][i] + 1) * scale for i, scale in enumerate((192, 224))]


def analyze(rows, tail=48):
    episodes = {}
    for row in rows:
        if row.get("event") == "transition":
            episodes.setdefault(row["episode"], []).append(row)
    if not episodes:
        raise ValueError("A full transition trace is required")
    reports = []
    for episode, transitions in episodes.items():
        # Identical rows for every lag; exclude death, resets, walls and padding.
        eligible = []
        for i in range(2, len(transitions)):
            row = transitions[i]
            if row["terminated"] or row["truncated"] or row["info"].get("dying"):
                continue
            if [r["step"] for r in transitions[i-2:i+1]] != list(range(row["step"]-2, row["step"]+1)):
                continue
            frames = row["info"]["delta_frames"]
            before, after = position(row["observation"]), position(row["next_observation"])
            if frames not in (1, 2) or not all(16 < p[0] < 368 and 24 < p[1] < 424 for p in (before, after)):
                continue
            # Avoid inflating agreement with the long stationary spell intro.
            if not any(r["action"] for r in transitions[i-2:i+1]):
                continue
            eligible.append((i, frames, [b-a for a, b in zip(before, after)]))
        fits = []
        for lag in range(3):
            errors = []
            for i, frames, displacement in eligible:
                predicted = velocity(transitions[i-lag]["action"])
                errors.append(sum((d-v*frames)**2 for d, v in zip(displacement, predicted)))
            fits.append({"lag_decisions": lag, "samples": len(errors),
                         "mean_squared_displacement_error": sum(errors)/len(errors) if errors else None})
        timing_values = {}
        for row in transitions:
            values = {**row["info"].get("timing", {}),
                      **{key: row[key] for key in ("student_inference_seconds", "teacher_inference_seconds")
                         if key in row}}
            if "decision_seconds" in row["info"]:
                values["decision_seconds"] = row["info"]["decision_seconds"]
            for key, value in values.items():
                if key.endswith("_seconds") and isinstance(value, (int, float)) and math.isfinite(value):
                    timing_values.setdefault(key, []).append(value)
        timing_summary = {key: {"samples": len(values), "median": statistics.median(values),
                                "max": max(values)} for key, values in timing_values.items()}
        detail = []
        for row in transitions[-tail:]:
            before, after = position(row["observation"]), position(row["next_observation"])
            gaps = {c[0]: c[3] for c in action_candidates(row["observation"])}
            detail.append({"step": row["step"], "position": before,
                           "action": row["action"], "teacher_action": row.get("teacher_action"),
                           "displacement": [b-a for a, b in zip(before, after)] if not row["terminated"] else None,
                           "candidate_clearance": gaps, "dying": row["info"].get("dying"),
                           "timing": row["info"].get("timing", {}),
                           "elapsed_frames": row["info"]["elapsed_frames"]})
        reports.append({"episode": episode, "transitions": len(transitions),
                        "disagreements": sum(r["action"] != r["teacher_action"] for r in transitions
                                             if "teacher_action" in r),
                        "timing_summary": timing_summary, "motion_fits": fits, "tail": detail})
    return {"episodes": reports, "interpretation":
            "Lag fits compare constant-speed hypotheses, not measured input latency. "
            "Snapshots, input polling and policy computation may affect displacement. "
            "Teacher labels and predicted clearances do not prove counterfactual survival."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace")
    parser.add_argument("--tail", type=int, default=48)
    parser.add_argument("--output", help="New JSON report; existing files are never overwritten")
    args = parser.parse_args()
    if args.tail < 1:
        parser.error("tail must be positive")
    with Path(args.trace).open(encoding="utf-8") as source:
        report = analyze([json.loads(line) for line in source if line.strip()], args.tail)
    result = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        with Path(args.output).open("x", encoding="utf-8") as target:
            target.write(result + "\n")
        print(args.output)
    else:
        print(result)


if __name__ == "__main__":
    main()
