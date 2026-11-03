"""Held endpoint masks preserve actor provenance and remain a separate experiment."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from agent.action_commitment import ActionCommitment
from agent.boundary_control import BoundaryCommitment, feasible_actions, BOUNDARY_CONTRACT
from tests.fakes import FakeKeyboard
from tests.test_control_evaluation import metadata
from tests.test_ppo_backend import AVAILABLE
from tests.test_training import FakeTask


def observation(x=192., y=384.):
    return {"player": [x / 192 - 1, y / 224 - 1, 0., 0.]}


def scores(action, runner_up=0):
    values = [-10.] * 9
    values[runner_up], values[action] = 8., 9.
    return values


class BoundaryControlTests(unittest.TestCase):
    def test_all_edges_corners_and_tangential_motion(self):
        self.assertEqual(set(feasible_actions(observation(376, 300), 4)), {0, 1, 5, 6, 7, 8})
        self.assertEqual(set(feasible_actions(observation(8, 300), 4)), {0, 1, 2, 3, 4, 5})
        self.assertEqual(set(feasible_actions(observation(192, 16), 4)), {0, 3, 4, 5, 6, 7})
        self.assertEqual(set(feasible_actions(observation(192, 432), 4)), {0, 1, 2, 3, 7, 8})
        self.assertEqual(set(feasible_actions(observation(376, 432), 4)), {0, 1, 7, 8})

    def test_horizon_matches_hold_not_teacher_forecast(self):
        self.assertIn(3, feasible_actions(observation(366, 300), 4))
        self.assertNotIn(3, feasible_actions(observation(366, 300), 12))
        self.assertIn(3, feasible_actions(observation(370, 300), 2))
        self.assertNotIn(3, feasible_actions(observation(370, 300), 4))
        self.assertIn(3, feasible_actions(observation(368, 300), 4))

    def test_mask_replaces_outward_choice_and_preserves_actual_raw_argmax(self):
        policy = BoundaryCommitment(lambda obs: scores(3, 5), lambda: 0)
        self.assertEqual(policy(observation(376, 300)), 5)
        row = policy.annotate({"event": "transition", "action": 5})
        self.assertEqual((row["raw_student_action"], row["masked_student_action"]), (3, 5))
        self.assertTrue(row["boundary_mask_overridden"])
        self.assertTrue(row["action_overridden"])
        self.assertFalse(row["commitment_overridden"])
        self.assertIn(5, row["boundary_execution_allowed_actions"])

    def test_safe_remaining_hold_is_kept_even_when_full_new_hold_would_cross(self):
        frame = [0]
        policy = BoundaryCommitment(lambda obs: scores(3, 7), lambda: frame[0])
        self.assertEqual(policy(observation(368, 300)), 3)
        policy.annotate({"event": "transition", "action": 3})
        frame[0] = 2
        self.assertEqual(policy(observation(372, 300)), 3)
        row = policy.annotate({"event": "transition", "action": 3})
        self.assertTrue(row["commitment_active"])
        self.assertEqual(row["masked_student_action"], 7)
        self.assertEqual(row["boundary_execution_horizon_frames"], 2)
        self.assertTrue(row["commitment_overridden"])
        self.assertFalse(row["action_overridden"])

    def test_overshoot_interrupts_unsafe_hold_and_reset_requires_annotation(self):
        frame = [0]
        policy = BoundaryCommitment(lambda obs: scores(3, 7), lambda: frame[0])
        policy(observation(368, 300))
        policy.annotate({"event": "transition", "action": 3})
        frame[0] = 3
        self.assertEqual(policy(observation(374, 300)), 7)
        row = policy.annotate({"event": "transition", "action": 7})
        self.assertTrue(row["boundary_hold_interrupted"])
        self.assertEqual(row["boundary_previous_hold_age"], 3)
        self.assertEqual(row["hold_age_frames"], 0)
        self.assertEqual(policy.summary()["boundary_hold_interruptions"], 1)
        frame[0] = 0
        with self.assertRaises(RuntimeError):
            policy(observation())
        policy.annotate({"event": "reset"})
        self.assertEqual(policy(observation()), 3)

    def test_interior_actions_match_unchanged_commitment(self):
        raw = [7, 3, 3, 0, 0, 7]
        a, b, frame = iter(raw), iter(raw), [0]
        baseline = ActionCommitment(lambda obs: next(a), lambda: frame[0])
        masked = BoundaryCommitment(lambda obs: scores(next(b)), lambda: frame[0])
        for elapsed in range(0, 12, 2):
            frame[0] = elapsed
            action = baseline(observation())
            self.assertEqual(masked(observation()), action)
            baseline.annotate({"event": "transition", "action": action})
            masked.annotate({"event": "transition", "action": action})
        self.assertEqual(masked.summary()["mask_overridden_decisions"], 0)

    def test_invalid_inputs_do_not_produce_an_action(self):
        for x, y, frames in [(0, 300, 4), (192, float("nan"), 4), (192, 300, 0), (192, 300, 17)]:
            with self.subTest(x=x, y=y, frames=frames), self.assertRaises(ValueError):
                feasible_actions(observation(x, y), frames)
        for values in ([0.] * 8, [float("inf")] + [0.] * 8, [float("nan")] * 9):
            with self.subTest(values=values), self.assertRaises(ValueError):
                BoundaryCommitment(lambda obs: values, lambda: 0)(observation())

    @unittest.skipUnless(AVAILABLE, "Install training dependencies")
    def test_cli_records_separate_controller_without_changing_checkpoint(self):
        from scripts.evaluate_control import main
        from agent.demonstrations import load_demonstrations
        class Task(FakeTask):
            capacity = 4
            @property
            def elapsed_frames(self):
                return self.position * 2
            def reset(self):
                _, info = super().reset()
                return observation(376), info
            def step(self, action):
                _, reward, done, truncated, info = super().step(action)
                return observation(376 - 4 * self.position), reward, done, truncated, info
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def verify_training_boundary(self):
                self.boundary()
        task = Task(length=3)
        class Backend:
            def __init__(self, *args, **kwargs):
                pass
            def action_logits(self, obs):
                return scores(3, 7)
            def deterministic_action(self, obs):
                raise AssertionError("Mask selection must use full logits")
            def update(self, *args, **kwargs):
                raise AssertionError("No training")
            def save(self, *args, **kwargs):
                raise AssertionError("No checkpoint changes")
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            checkpoint = Path(directory) / "source"
            checkpoint.mkdir()
            (checkpoint / "model.zip").write_bytes(b"frozen")
            (checkpoint / "metadata.json").write_text(json.dumps(metadata()))
            before = {p.name: p.read_bytes() for p in checkpoint.iterdir()}
            output = Path(directory) / "masked.jsonl"
            args = ["evaluate_control", "--resume", str(checkpoint), "--task", "source",
                    "--output", str(output), "--episodes", "2", "--boundary-mask"]
            reader = SimpleNamespace(addresses={"SPELL_RESULT_ACTIVE": 1}, close=lambda: None)
            with patch("sys.argv", args), patch("scripts.evaluate_control.read_checkpoint", return_value=metadata()), \
                    patch("scripts.evaluate_control.connect", return_value=(reader, 1, "test-build")), \
                    patch("scripts.evaluate_control.WindowsKeyboard", return_value=FakeKeyboard()), \
                    patch("scripts.evaluate_control.GameSession", return_value=task), \
                    patch("agent.ppo_backend.PPOBackend", Backend), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                main()
            rows = [json.loads(s) for s in output.read_text().splitlines()]
            self.assertEqual(rows[0]["collection"], "boundary_control_evaluation")
            self.assertEqual(rows[0]["boundary_contract"], BOUNDARY_CONTRACT)
            transitions = [r for r in rows if r["event"] == "transition"]
            self.assertEqual([r["action"] for r in transitions], [7, 7, 3, 7, 7, 3])
            self.assertTrue(all(r["raw_student_action"] == 3 for r in transitions))
            self.assertEqual({p.name: p.read_bytes() for p in checkpoint.iterdir()}, before)
            self.assertTrue(output.with_suffix(".console.log").exists())
            with self.assertRaises(ValueError):
                load_demonstrations([output])
