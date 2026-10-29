"""Actual supervised fitting and PPO-compatible publication without native input."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.test_ppo_backend import AVAILABLE


@unittest.skipUnless(AVAILABLE, "Install requirements-training.txt")
class CloningTests(unittest.TestCase):
    def test_offline_cli_fits_actor_and_publishes_resumable_checkpoint(self):
        from tests.test_demonstrations import trace_rows, write_trace
        from scripts.clone import main
        from agent.checkpoints import read_checkpoint
        from agent.ppo_backend import PPOBackend
        import gymnasium as gym
        from env.touhou_env import state_space
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / "teacher.jsonl"
            output = Path(directory) / "clone"
            write_trace(path, trace_rows())
            with patch("sys.argv", ["clone", "--demos", str(path), "--run-dir", str(output),
                                    "--epochs", "20"]), contextlib.redirect_stdout(io.StringIO()):
                main()
            saved = output / "checkpoints" / "update-000000"
            metadata = read_checkpoint(saved)
            result = metadata["initialization"]["result"]
            self.assertLess(result["validation"]["loss"], result["initial_validation"]["loss"])
            self.assertEqual(result["validation"]["accuracy"], 1.0)
            self.assertEqual(metadata["timesteps"], 0)
            class OfflineEnv(gym.Env):
                observation_space = state_space(2)
                action_space = gym.spaces.Discrete(9)
                def reset(self, **kwargs):
                    raise AssertionError("Native reset during cloning")
                def step(self, action):
                    raise AssertionError("Native step during cloning")
            learner = PPOBackend(OfflineEnv(), checkpoint=str(saved / "model.zip"))
            self.assertEqual(len(learner.model.policy.optimizer.state), 0)
            row = next(row for row in trace_rows() if row["event"] == "transition")
            self.assertEqual(learner.deterministic_action(row["observation"]), row["action"])
            # The cloned archive can immediately receive an actual PPO update.
            from agent.training import Transition
            decision = learner.act(row["observation"])
            batch = [Transition(row["observation"], decision.action, 0.01, start,
                                decision.value, decision.log_prob) for start in (True, False)]
            learner.update(batch, 2)
            self.assertEqual(learner.timesteps, 2)
