"""Teacher labels stay distinct from actions executed by the student."""
from pathlib import Path
import tempfile
import unittest

from agent.corrections import CorrectionPolicy
from agent.demonstrations import load_demonstrations
from tests.test_demonstrations import trace_rows, write_trace


def correction_rows():
    rows = list(trace_rows())
    rows[0].update({"collection": "corrections", "policy": "mixed_teacher_student",
                    "teacher": "reactive_attack", "teacher_probability": 0.5,
                    "student_checkpoint": "synthetic-checkpoint"})
    for row in rows:
        if row["event"] == "transition":
            row["teacher_action"] = row["action"]
            row["student_action"] = (row["action"] + 1) % 9
            row["action"] = row["student_action"]
            row["executed_by"] = "student"
            if row["terminated"]:
                row["info"]["reason"] = "spell_failure"
        elif row["event"] == "episode_end":
            row["reason"] = "spell_failure"
    return rows


class CorrectionTests(unittest.TestCase):
    def test_trace_callback_keeps_labels_aligned_through_retries(self):
        from tests.test_training import FakeTask
        from scripts.diagnostics import run_episodes
        policy = CorrectionPolicy(lambda obs: 3, 0, teacher=lambda obs: 7)
        rows = []
        run_episodes(FakeTask(length=2), 2, lambda row: rows.append(policy.annotate(row)),
                     trace=True, policy="mixed_teacher_student", action_fn=policy)
        transitions = [row for row in rows if row["event"] == "transition"]
        self.assertEqual(len(transitions), 4)
        self.assertEqual([row["action"] for row in transitions], [3] * 4)
        self.assertEqual([row["teacher_action"] for row in transitions], [7] * 4)
        self.assertEqual([row["observation"]["player"][0] for row in transitions], [0, 1, 0, 1])

    def test_execution_probability_and_annotation(self):
        for probability, expected, source in ((0, 3, "student"), (1, 7, "teacher")):
            policy = CorrectionPolicy(lambda obs: 3, probability, teacher=lambda obs: 7)
            self.assertEqual(policy({}), expected)
            row = policy.annotate({"event": "transition", "action": expected})
            self.assertEqual(row["teacher_action"], 7)
            self.assertEqual(row["student_action"], 3)
            self.assertEqual(row["executed_by"], source)
            self.assertEqual(policy.summary()["disagreement_rate"], 1)
            with self.assertRaises(RuntimeError):
                policy.annotate({"event": "transition", "action": expected})

    def test_seed_reproduces_mixture_and_mismatch_is_rejected(self):
        first = CorrectionPolicy(lambda obs: 1, seed=42, teacher=lambda obs: 2)
        second = CorrectionPolicy(lambda obs: 1, seed=42, teacher=lambda obs: 2)
        self.assertEqual([first({}) for _ in range(20)], [second({}) for _ in range(20)])
        with self.assertRaises(RuntimeError):
            first.annotate({"event": "transition", "action": 8})
        with self.assertRaises(ValueError):
            CorrectionPolicy(lambda obs: 0, teacher_probability=2)

    def test_failed_student_episodes_use_teacher_labels(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / "corrections.jsonl"
            rows = correction_rows()
            write_trace(path, rows)
            episodes, metadata = load_demonstrations([path])
            self.assertEqual(len(episodes), 3)
            label = rows[2]["teacher_action"]
            self.assertNotEqual(label, rows[2]["action"])
            self.assertEqual(episodes[0]["actions"], [label] * 3)
            self.assertEqual(metadata["sources"][0]["collection"], "corrections")

    def test_teacher_and_correction_sources_can_be_aggregated(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            teacher, corrections = Path(directory) / "teacher.jsonl", Path(directory) / "corrections.jsonl"
            write_trace(teacher, trace_rows())
            write_trace(corrections, correction_rows())
            episodes, _ = load_demonstrations([teacher, corrections])
            self.assertEqual(len(episodes), 6)
            rows = correction_rows()
            rows[0]["task"] = "different-spell"
            write_trace(corrections, rows)
            with self.assertRaises(ValueError):
                load_demonstrations([teacher, corrections])

    def test_corrupt_teacher_label_and_execution_source_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / "corrections.jsonl"
            for corrupt in ("label", "source"):
                rows = correction_rows()
                if corrupt == "label":
                    rows[2]["teacher_action"] = (rows[2]["teacher_action"] + 2) % 9
                else:
                    rows[2]["executed_by"] = "teacher"
                write_trace(path, rows)
                with self.subTest(corrupt=corrupt), self.assertRaises(ValueError):
                    load_demonstrations([path])
