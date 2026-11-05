"""Clearance filtering, action provenance and explicit evaluation-only CLI."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from agent.risk_control import RiskBoundaryCommitment, forecast_clearances, RISK_CONTRACT
from tests.test_boundary_control import observation, scores
from tests.test_control_evaluation import metadata
from tests.test_ppo_backend import AVAILABLE
from tests.test_training import FakeTask
from tests.fakes import FakeKeyboard


def scene(distance=12., x=192., capacity=1):
    return {**observation(x), "bullets": [[0., distance / 448, 0., 0., 2 / 32, .25, 1.]] + [[0.] * 7 for _ in range(capacity - 1)],
            "enemies": [[0.] * 6 for _ in range(16)]}


class RiskControlTests(unittest.TestCase):
    def test_real_geometry_rejects_collision_course_and_keeps_safe_actor_choice(self):
        obs = scene()
        gaps = forecast_clearances(obs)
        self.assertAlmostEqual(gaps[5], -3.25)
        self.assertAlmostEqual(gaps[7], 8.75)
        policy = RiskBoundaryCommitment(lambda _: scores(5, 7), lambda: 0)
        self.assertEqual(policy(obs), 7)
        row = policy.annotate({"event": "transition", "action": 7})
        self.assertEqual((row["raw_student_action"], row["boundary_student_action"],
                          row["risk_filtered_action"]), (5, 5, 7))
        self.assertNotIn(5, row["risk_allowed_actions"])
        self.assertEqual(row["risk_candidate_mode"], "safe")
        self.assertTrue(row["risk_mask_overridden"])
        self.assertFalse(row["boundary_mask_overridden"])
        self.assertFalse(row["commitment_overridden"])

    def test_emergency_uses_best_available_gap_without_claiming_safety(self):
        policy = RiskBoundaryCommitment(lambda _: scores(5, 7), lambda: 0)
        action = policy(scene(3.))
        row = policy.annotate({"event": "transition", "action": action})
        self.assertEqual(row["risk_candidate_mode"], "emergency")
        self.assertLess(row["risk_best_clearance"], 0)
        self.assertTrue(all(row["risk_clearances"][a] == row["risk_best_clearance"]
                            for a in row["risk_allowed_actions"]))
        self.assertNotEqual(action, 5)

    def test_physical_horizon_stays_four_frames_even_with_twelve_frame_forecast(self):
        obs = scene(); obs["bullets"] = []
        obs["player"] = observation(366)["player"]
        policy = RiskBoundaryCommitment(lambda _: scores(3, 7), lambda: 0)
        self.assertEqual(policy(obs), 3)
        policy.annotate({"event": "transition", "action": 3})
        policy.annotate({"event": "reset"})
        obs["player"] = observation(374)["player"]
        self.assertEqual(policy(obs), 7)
        row = policy.annotate({"event": "transition", "action": 7})
        self.assertTrue(row["boundary_mask_overridden"])
        self.assertFalse(row["risk_mask_overridden"])

    def test_risk_change_does_not_replace_a_held_action_under_its_old_decision(self):
        frame = [0]
        policy = RiskBoundaryCommitment(lambda _: scores(5, 7), lambda: frame[0])
        obs = scene(); obs["bullets"] = []
        self.assertEqual(policy(obs), 5)
        policy.annotate({"event": "transition", "action": 5})
        frame[0] = 2
        self.assertEqual(policy(scene()), 5)
        row = policy.annotate({"event": "transition", "action": 5})
        self.assertEqual(row["risk_filtered_action"], 7)
        self.assertTrue(row["commitment_active"])
        self.assertTrue(row["commitment_overridden"])
        self.assertFalse(row["boundary_hold_interrupted"])
        frame[0] = 4
        self.assertEqual(policy(scene()), 7)
        policy.annotate({"event": "transition", "action": 7})
        self.assertEqual(policy.summary()["risk_applied_fresh_choices"], 1)

    @unittest.skipUnless(AVAILABLE, "Install training dependencies")
    def test_geometry_matches_existing_feature_gaps_and_rejects_nonfinite_rows(self):
        import torch
        from agent.hazard_features import AttackZoneFeatures
        from env.touhou_env import state_space
        obs = scene()
        tensor = {k: torch.tensor([v], dtype=torch.float32) for k, v in obs.items()}
        feature = AttackZoneFeatures(state_space(1))(tensor)[0, 8:].reshape(9, 6)
        expected = torch.tensor(forecast_clearances(obs))
        torch.testing.assert_close(feature[:, 1] * 24, expected, atol=1e-5, rtol=1e-5)
        obs["bullets"][0][0] = float("nan")
        with self.assertRaises(ValueError):
            forecast_clearances(obs)

    def test_cli_requires_explicit_boundary_flag_before_any_connection(self):
        from scripts.evaluate_control import main
        args = ["evaluate_control", "--resume", "unused", "--task", "source",
                "--output", "unused", "--risk-filter"]
        with patch("sys.argv", args), patch("scripts.evaluate_control.connect") as connect, \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main()
        connect.assert_not_called()

    @unittest.skipUnless(AVAILABLE, "Install training dependencies")
    def test_fake_native_cli_uses_separate_collection_and_preserves_source(self):
        from scripts.evaluate_control import main
        from agent.demonstrations import load_demonstrations
        from agent.zone_demonstrations import load_zone_demonstrations
        class Task(FakeTask):
            capacity = 4
            @property
            def elapsed_frames(self): return self.position * 2
            def reset(self):
                _, info = super().reset()
                return scene(capacity=4), info
            def step(self, action):
                _, reward, done, truncated, info = super().step(action)
                return scene(capacity=4), reward, done, truncated, info
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def verify_training_boundary(self): self.boundary()
        class Backend:
            def __init__(self, *args, **kwargs): pass
            def action_logits(self, obs): return scores(5, 7)
            def update(self, *args, **kwargs): raise AssertionError("No training")
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            checkpoint = Path(directory) / "source"; checkpoint.mkdir()
            (checkpoint / "model.zip").write_bytes(b"frozen")
            (checkpoint / "metadata.json").write_text(json.dumps(metadata()))
            before = {p.name: p.read_bytes() for p in checkpoint.iterdir()}
            output = Path(directory) / "risk.jsonl"
            args = ["evaluate_control", "--resume", str(checkpoint), "--task", "source",
                    "--output", str(output), "--episodes", "2", "--boundary-mask", "--risk-filter"]
            reader = SimpleNamespace(addresses={"SPELL_RESULT_ACTIVE": 1}, close=lambda: None)
            with patch("sys.argv", args), patch("scripts.evaluate_control.read_checkpoint", return_value=metadata()), \
                    patch("scripts.evaluate_control.connect", return_value=(reader, 1, "test-build")), \
                    patch("scripts.evaluate_control.WindowsKeyboard", return_value=FakeKeyboard()), \
                    patch("scripts.evaluate_control.GameSession", return_value=Task(length=3)), \
                    patch("agent.ppo_backend.PPOBackend", Backend), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                main()
            rows = [json.loads(s) for s in output.read_text().splitlines()]
            self.assertEqual(rows[0]["collection"], "risk_control_evaluation")
            self.assertEqual(rows[0]["risk_contract"], RISK_CONTRACT)
            self.assertFalse(rows[0]["risk_interrupts_hold"])
            transitions = [r for r in rows if r["event"] == "transition"]
            self.assertEqual([r["action"] for r in transitions], [7] * 6)
            self.assertTrue(all(r["boundary_student_action"] == 5 for r in transitions))
            self.assertEqual(rows[-1]["risk_applied_fresh_choices"], 4)
            self.assertEqual({p.name: p.read_bytes() for p in checkpoint.iterdir()}, before)
            for loader in (load_demonstrations, load_zone_demonstrations):
                with self.assertRaises(ValueError): loader([output])
