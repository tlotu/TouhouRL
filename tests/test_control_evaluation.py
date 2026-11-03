"""Evaluation task transfer preserves provenance and frozen action execution."""
from copy import deepcopy
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.evaluate_control import evaluation_identity, main
from tests.fakes import FakeKeyboard
from tests.test_ppo_backend import AVAILABLE
from tests.test_training import FakeTask


def metadata():
    return {"build_md5": "test-build", "initialization": {"identity": {"task": "source"}},
            "config": {"capacity": 4, "max_frames": 3600,
                       "decision_contract": "attack-zone-held-v1", "hold_frames": 4}}


class ControlEvaluationTests(unittest.TestCase):
    def test_source_identity_survives_explicit_transfer(self):
        saved = metadata()
        before = deepcopy(saved)
        same = evaluation_identity(saved, "source")
        self.assertFalse(same["task_transfer"])
        moved = evaluation_identity(saved, "target", allow_task_transfer=True,
                                    transfer_reason=" Faster, denser straight bullets ")
        self.assertEqual((moved["training_task"], moved["evaluation_task"]), ("source", "target"))
        self.assertEqual(moved["transfer_reason"], "Faster, denser straight bullets")
        self.assertEqual(moved["task_verification"], "manual")
        self.assertEqual(saved, before)

    def test_transfer_requires_different_known_task_and_reason(self):
        cases = [(metadata(), "target", {}),
                 (metadata(), "target", {"transfer_reason": "test"}),
                 (metadata(), "target", {"allow_task_transfer": True}),
                 (metadata(), "target", {"allow_task_transfer": True, "transfer_reason": " "}),
                 (metadata(), "source", {"allow_task_transfer": True, "transfer_reason": "test"}),
                 ({}, "target", {"allow_task_transfer": True, "transfer_reason": "test"}),
                 (metadata(), " ", {})]
        for saved, task, options in cases:
            with self.subTest(task=task, options=options), self.assertRaises(ValueError):
                evaluation_identity(saved, task, **options)
        # Legacy checkpoints without demonstration identities retain same-task access.
        self.assertIsNone(evaluation_identity({}, "legacy")["training_task"])

    def test_cli_rejects_implicit_transfer_before_game_connection(self):
        args = ["evaluate_control", "--resume", "unused", "--task", "target", "--output", "unused"]
        with patch("sys.argv", args), patch("scripts.evaluate_control.read_checkpoint", return_value=metadata()), \
                patch("scripts.evaluate_control.connect") as connect, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as stopped:
                main()
            self.assertEqual(stopped.exception.code, 2)
            connect.assert_not_called()

    @unittest.skipUnless(AVAILABLE, "Install training dependencies")
    def test_transfer_does_not_bypass_saved_hold_duration(self):
        args = ["evaluate_control", "--resume", "unused", "--task", "target", "--output", "unused",
                "--allow-task-transfer", "--transfer-reason", "test", "--hold-frames", "2"]
        with patch("sys.argv", args), patch("scripts.evaluate_control.read_checkpoint", return_value=metadata()), \
                patch("scripts.evaluate_control.connect") as connect, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as stopped:
                main()
            self.assertEqual(stopped.exception.code, 2)
            connect.assert_not_called()

    @unittest.skipUnless(AVAILABLE, "Install training dependencies")
    def test_frozen_transfer_logs_identity_hash_and_preserves_holding_across_retries(self):
        class Task(FakeTask):
            capacity = 4
            @property
            def elapsed_frames(self):
                return self.position * 2
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def verify_training_boundary(self):
                self.boundary()

        task = Task(length=3)
        class Backend:
            def __init__(self, env, checkpoint):
                self.checkpoint = checkpoint
            def deterministic_action(self, obs):
                return 7 if task.position == 0 else 3
            def update(self, *args, **kwargs):
                raise AssertionError("Frozen evaluation cannot train")
            def save(self, *args, **kwargs):
                raise AssertionError("Frozen evaluation cannot publish models")

        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            source = Path(directory) / "checkpoint"
            source.mkdir()
            archive = source / "model.zip"
            archive.write_bytes(b"frozen archive")
            marker = source / "metadata.json"
            marker.write_text(json.dumps(metadata()), encoding="utf-8")
            original_files = {p.name: p.read_bytes() for p in source.iterdir()}
            output = Path(directory) / "transfer.jsonl"
            args = ["evaluate_control", "--resume", str(source), "--task", "target", "--output", str(output),
                    "--episodes", "2", "--allow-task-transfer", "--transfer-reason", "denser straight bullets"]
            reader = SimpleNamespace(addresses={"SPELL_RESULT_ACTIVE": 1}, close=lambda: None)
            with patch("sys.argv", args), patch("scripts.evaluate_control.read_checkpoint", return_value=metadata()), \
                    patch("scripts.evaluate_control.connect", return_value=(reader, 1, "test-build")), \
                    patch("scripts.evaluate_control.WindowsKeyboard", return_value=FakeKeyboard()), \
                    patch("scripts.evaluate_control.GameSession", return_value=task), \
                    patch("agent.ppo_backend.PPOBackend", Backend), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                main()
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            header = rows[0]
            self.assertEqual((header["training_task"], header["evaluation_task"]), ("source", "target"))
            self.assertTrue(header["task_transfer"])
            self.assertEqual(header["task"], "target")
            self.assertEqual(header["collection"], "control_evaluation")
            self.assertEqual(header["student_model_sha256"], hashlib.sha256(b"frozen archive").hexdigest())
            self.assertEqual([r["action"] for r in rows if r["event"] == "transition"], [7, 7, 3, 7, 7, 3])
            self.assertEqual(next(r for r in rows if r["event"] == "summary")["successful_episodes"], 2)
            self.assertEqual({p.name: p.read_bytes() for p in source.iterdir()}, original_files)
            self.assertIn("Training task: source; evaluation task: target",
                          output.with_suffix(".console.log").read_text(encoding="utf-8"))
