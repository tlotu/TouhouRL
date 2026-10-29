"""Actual optimizer checks when optional training dependencies are installed."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

AVAILABLE = bool(importlib.util.find_spec("torch") and importlib.util.find_spec("stable_baselines3"))


@unittest.skipUnless(AVAILABLE, "Install requirements-training.txt to test the actual PPO optimizer")
class PPOBackendTests(unittest.TestCase):
    def test_real_optimizer_and_checkpoint_roundtrip(self):
        import gymnasium as gym
        from gymnasium import spaces
        import numpy as np
        import torch
        from agent.ppo_backend import PPOBackend
        from agent.training import Transition

        class TinyEnv(gym.Env):
            observation_space = spaces.Dict({"vector": spaces.Box(-1, 1, (2,), dtype=np.float32)})
            action_space = spaces.Discrete(9)
            def reset(self, *, seed=None, options=None):
                raise AssertionError("Backend must not reset a native game")
            def step(self, action):
                raise AssertionError("Backend must not collect through an auto-reset wrapper")

        env = TinyEnv()
        backend = PPOBackend(env, rollout_steps=64)
        before = [p.detach().clone() for p in backend.model.policy.parameters()]
        rows = []
        for i in range(4):
            obs = {"vector": np.array([0.1 * i, 0.2], dtype=np.float32)}
            decision = backend.act(obs)
            rows.append(Transition(obs, decision.action, 1.0 if i % 2 else -1.0,
                                   i % 2 == 0, decision.value, decision.log_prob))
        metrics = backend.update(rows, 4)
        self.assertEqual(backend.timesteps, 4)
        self.assertIn("train/policy_gradient_loss", metrics)
        self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, backend.model.policy.parameters())))
        self.assertTrue(backend.model.policy.optimizer.state_dict()["state"])
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            checkpoint = Path(directory) / "model.zip"
            backend.save(checkpoint)
            restored = PPOBackend(env, checkpoint=str(checkpoint))
            self.assertEqual(restored.timesteps, 4)
            self.assertEqual(len(restored.model.policy.optimizer.state_dict()["state"]),
                             len(backend.model.policy.optimizer.state_dict()["state"]))
            for old, new in zip(backend.model.policy.parameters(), restored.model.policy.parameters()):
                self.assertTrue(torch.equal(old, new))

    def test_gae_does_not_cross_episode_boundaries(self):
        import numpy as np
        import torch
        from gymnasium import spaces
        from stable_baselines3.common.buffers import DictRolloutBuffer
        space = spaces.Dict({"x": spaces.Box(-1, 1, (1,), dtype=np.float32)})
        buffer = DictRolloutBuffer(4, space, spaces.Discrete(9), device="cpu", gamma=1, gae_lambda=1)
        for reward, start in zip([1, 1, 10, 10], [True, False, True, False]):
            buffer.add({"x": np.zeros(1, dtype=np.float32)}, np.array([0]),
                       np.array([reward]), np.array([start]), torch.zeros(1), torch.zeros(1))
        buffer.compute_returns_and_advantage(torch.zeros(1), np.array([True]))
        np.testing.assert_allclose(buffer.returns.flatten(), [2, 1, 20, 10])
