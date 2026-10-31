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
    def test_weighted_loss_prioritizes_rare_correction_and_validates_weights(self):
        import gymnasium as gym
        import numpy as np
        from agent.cloning import fit_actor
        from agent.ppo_backend import PPOBackend
        class Offline(gym.Env):
            observation_space = gym.spaces.Dict({"x": gym.spaces.Box(-1, 1, (2,), dtype=np.float32)})
            action_space = gym.spaces.Discrete(9)
        learner = PPOBackend(Offline(), rollout_steps=64)
        data = ({"x": np.zeros((2, 2), dtype=np.float32)}, np.array([3, 7], dtype=np.int64))
        weights = np.array([9., 1.], dtype=np.float32)
        result = fit_actor(learner, data, data, lambda row: None, epochs=30,
                           learning_rate=0.003, train_weights=weights, validation_weights=weights)
        self.assertEqual(learner.deterministic_action({"x": [0., 0.]}), 3)
        self.assertEqual(result["validation"]["priority_accuracy"], 1)
        self.assertLess(result["validation"]["weighted_loss"], result["validation"]["loss"])
        self.assertFalse(learner.model.policy.optimizer.state)
        with self.assertRaises(ValueError):
            fit_actor(learner, data, data, lambda row: None, train_weights=[-1., 1.])

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
