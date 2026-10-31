"""Action commitment uses actual elapsed frames and keeps action provenance."""
import unittest
from pathlib import Path
import tempfile

from agent.action_commitment import ActionCommitment
from tests.fakes import FakeClock


class ActionCommitmentTests(unittest.TestCase):
    def test_holds_reversals_and_neutral_but_logs_every_raw_prediction(self):
        frame = [0]
        raw = iter((7, 3, 3, 0, 0, 7))
        policy = ActionCommitment(lambda obs: next(raw), lambda: frame[0], 4)
        actions, records = [], []
        for elapsed in range(0, 12, 2):
            frame[0] = elapsed
            action = policy({})
            actions.append(action)
            records.append(policy.annotate({"event": "transition", "action": action}))
        self.assertEqual(actions, [7, 7, 3, 3, 0, 0])
        self.assertEqual([r["raw_student_action"] for r in records], [7, 3, 3, 0, 0, 7])
        self.assertEqual(policy.summary()["overridden_decisions"], 3)

    def test_overshoot_uses_actual_frames_and_reset_drops_previous_hold(self):
        frame = [0]
        raw = [7]
        policy = ActionCommitment(lambda obs: raw[0], lambda: frame[0], 4)
        def step():
            action = policy({})
            policy.annotate({"event": "transition", "action": action})
            return action
        self.assertEqual(step(), 7)
        frame[0], raw[0] = 5, 3
        self.assertEqual(step(), 3)
        frame[0], raw[0] = 0, 0
        with self.assertRaises(RuntimeError):
            step()
        policy.annotate({"event": "reset"})
        self.assertEqual(step(), 0)

    def test_two_frame_control_has_no_overrides_and_times_inference(self):
        clock = FakeClock()
        frame = [0]
        def student(obs):
            clock.sleep(0.001)
            return 3 if frame[0] == 0 else 7
        policy = ActionCommitment(student, lambda: frame[0], 2, clock=clock)
        for elapsed, expected in ((0, 3), (2, 7)):
            frame[0] = elapsed
            self.assertEqual(policy({}), expected)
            row = policy.annotate({"event": "transition", "action": expected})
            self.assertFalse(row["action_overridden"])
            self.assertAlmostEqual(row["student_inference_seconds"], 0.001)

    def test_invalid_actions_and_misaligned_annotations_fail(self):
        with self.assertRaises(ValueError):
            ActionCommitment(lambda obs: 0, lambda: 0, 3)
        with self.assertRaises(ValueError):
            ActionCommitment(lambda obs: 9, lambda: 0)({})
        policy = ActionCommitment(lambda obs: 3, lambda: 0)
        policy({})
        with self.assertRaises(RuntimeError):
            policy({})
        with self.assertRaises(RuntimeError):
            policy.annotate({"event": "transition", "action": 7})

    def test_episode_runner_resets_commitment_and_preserves_both_actions(self):
        from scripts.diagnostics import run_episodes
        from tests.test_training import FakeTask
        task = FakeTask(length=3)
        policy = ActionCommitment(lambda obs: 7 if task.position == 0 else 3,
                                  lambda: task.position * 2, 4)
        rows = []
        run_episodes(task, 2, lambda row: rows.append(policy.annotate(row)),
                     policy="ppo_action_commitment", trace=True, action_fn=policy)
        trace = [r for r in rows if r["event"] == "transition"]
        self.assertEqual([r["action"] for r in trace], [7, 7, 3, 7, 7, 3])
        self.assertEqual([r["raw_student_action"] for r in trace], [7, 3, 3, 7, 3, 3])
        self.assertTrue(task.menu)

    def test_control_evaluation_cannot_be_loaded_as_teacher_demonstrations(self):
        from agent.demonstrations import load_demonstrations
        from tests.test_demonstrations import trace_rows, write_trace
        rows = list(trace_rows())
        rows[0].update(policy="ppo_action_commitment", collection="control_evaluation")
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / "control.jsonl"
            write_trace(path, rows)
            with self.assertRaises(ValueError):
                load_demonstrations([path])
