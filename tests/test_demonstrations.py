"""Trace boundaries and episode-level validation isolation."""
import json
from pathlib import Path
import tempfile
import unittest

from agent.baselines import reactive_attack
from agent.demonstrations import load_demonstrations, split_episodes, stack_episodes
from env.session import OBSERVATION_SCHEMA


def trace_rows():
    yield {"event": "connected", "schema": OBSERVATION_SCHEMA, "policy": "reactive_attack",
           "repeat": 2, "trace": True, "run": True, "task": "synthetic-test",
           "build_md5": "test-build", "capacity": 2, "max_frames": 3600}
    for episode in range(3):
        yield {"event": "reset", "episode": episode}
        for step in range(3):
            obs = {"player": [0, 0.5, 0, 0], "bullets": [[0.] * 7 for _ in range(2)],
                   "enemies": [[0.] * 6 for _ in range(16)]}
            obs["enemies"][0] = [0.3, -0.5, 1, 1, 1, 1]
            done = step == 2
            yield {"event": "transition", "episode": episode, "step": step,
                   "observation": obs, "action": reactive_attack(obs),
                   "terminated": done, "truncated": False,
                   "info": {"reason": "spell_success" if done else None}}
        yield {"event": "episode_end", "episode": episode, "steps": 3, "reason": "spell_success"}


def write_trace(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class DemonstrationTests(unittest.TestCase):
    def test_complete_episodes_split_without_frame_leakage(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / "trace.jsonl"
            write_trace(path, trace_rows())
            episodes, meta = load_demonstrations([path])
            train, validation = split_episodes(episodes)
            self.assertFalse({e["id"] for e in train} & {e["id"] for e in validation})
            self.assertEqual(len(stack_episodes(train)[1]), 6)
            self.assertEqual(len(meta["sources"][0]["sha256"]), 64)
            with self.assertRaises(ValueError):
                load_demonstrations([path, path])

    def test_incomplete_tail_is_discarded(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / "trace.jsonl"
            write_trace(path, list(trace_rows())[:-1])
            episodes, meta = load_demonstrations([path])
            self.assertEqual(len(episodes), 2)
            self.assertEqual(meta["incomplete_episodes"], 1)

    def test_failures_require_explicit_opt_in(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / "trace.jsonl"
            rows = list(trace_rows())
            for row in rows:
                if row.get("reason") == "spell_success":
                    row["reason"] = "spell_failure"
                if row.get("info", {}).get("reason") == "spell_success":
                    row["info"]["reason"] = "spell_failure"
            write_trace(path, rows)
            self.assertEqual(len(load_demonstrations([path])[0]), 0)
            self.assertEqual(len(load_demonstrations([path], include_failures=True)[0]), 3)

    def test_wrong_action_shape_sequence_and_task_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / "trace.jsonl"
            for mutation in ("action", "shape", "step", "task"):
                rows = list(trace_rows())
                if mutation == "action":
                    rows[2]["action"] = (rows[2]["action"] + 1) % 9
                elif mutation == "shape":
                    rows[2]["observation"]["bullets"] = []
                elif mutation == "step":
                    rows[2]["step"] = 5
                else:
                    rows[0]["task"] = None
                write_trace(path, rows)
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    load_demonstrations([path])

    def test_single_episode_cannot_be_split(self):
        with self.assertRaises(ValueError):
            split_episodes([{"id": "one"}])
