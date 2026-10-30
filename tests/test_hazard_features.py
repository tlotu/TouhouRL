"""Geometry inputs match the baseline approximation without choosing an action."""
from pathlib import Path
import tempfile
import unittest

from tests.test_ppo_backend import AVAILABLE


@unittest.skipUnless(AVAILABLE, "Install requirements-training.txt")
class HazardFeatureTests(unittest.TestCase):
    def make_data(self):
        import torch
        return {"player": torch.tensor([[0., 0.5, 0., 0.]]),
                "bullets": torch.zeros(1, 4, 7), "enemies": torch.zeros(1, 16, 6)}

    def test_clearances_and_wall_validity_match_baseline(self):
        import torch
        from agent.hazard_features import HazardFeatures
        from agent.baselines import action_candidates
        from env.touhou_env import state_space
        data = self.make_data()
        data["bullets"][0, 0] = torch.tensor([0.02, -0.04, 0.01, 0.08, 0.1, 0.25, 1.])
        features = HazardFeatures(state_space(4))(data)
        self.assertEqual(features.shape, (1, 62))
        self.assertTrue(torch.isfinite(features).all())
        candidates = features[0, 8:].reshape(9, 6)
        for action, _, _, gap, _ in action_candidates({k: v[0].tolist() for k, v in data.items()}):
            self.assertEqual(candidates[action, 0].item(), 1)
            self.assertAlmostEqual(candidates[action, 1].item() * 24, gap, places=4)
        data["player"][0, :2] = torch.tensor([-0.95, 0.92])
        output = HazardFeatures(state_space(4))(data)[0, 8:].reshape(9, 6)
        valid = {c[0] for c in action_candidates({k: v[0].tolist() for k, v in data.items()})}
        self.assertEqual({i for i in range(9) if output[i, 0] == 1}, valid)

    def test_padding_bullet_permutation_and_target_filtering(self):
        import torch
        from agent.hazard_features import HazardFeatures
        from env.touhou_env import state_space
        data = self.make_data()
        extractor = HazardFeatures(state_space(4))
        empty = extractor(data)
        data["bullets"][0, 0, :6] = 0.8
        torch.testing.assert_close(empty, extractor(data))
        data["bullets"][0, 0, 6] = 1
        before = extractor(data)
        data["bullets"] = data["bullets"][:, [3, 2, 1, 0]]
        torch.testing.assert_close(before, extractor(data))
        data["enemies"][0, 0] = torch.tensor([0.2, -0.5, 1., 1., 1., 1.])
        target = extractor(data)
        self.assertAlmostEqual(target[0, 4].item(), 0.2, places=6)
        self.assertEqual(target[0, 6].item(), 1)
        data["enemies"][0, 0, 4] = 0
        self.assertEqual(extractor(data)[0, 6].item(), 0)

    def test_custom_extractor_roundtrip_and_actual_ppo_update(self):
        import gymnasium as gym
        from agent.ppo_backend import PPOBackend
        from agent.training import Transition
        from env.touhou_env import state_space
        class Offline(gym.Env):
            observation_space = state_space(4)
            action_space = gym.spaces.Discrete(9)
        learner = PPOBackend(Offline(), features="hazard-v1", rollout_steps=64)
        obs = {k: v[0].numpy() for k, v in self.make_data().items()}
        decision = learner.act(obs)
        learner.update([Transition(obs, decision.action, 0.01, start, decision.value,
                                   decision.log_prob) for start in (True, False)], 2)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / "model.zip"
            learner.save(path)
            restored = PPOBackend(Offline(), checkpoint=str(path))
            self.assertEqual(restored.model.policy.features_extractor.features_dim, 62)
            self.assertEqual(restored.deterministic_action(obs), learner.deterministic_action(obs))
