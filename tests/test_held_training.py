"""Held action likelihoods, frame returns, terminal boundaries and optimizer resume."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import contextlib
import io
import json
from types import SimpleNamespace
from unittest.mock import patch

from agent.training import Decision, TrainingError, train_episodic
from agent.held_training import (HeldTransition, collect_held_episodes, duration_gae,
                                 frame_reward, HELD_REWARD_PROFILE)
from scripts.train_held import held_config
from tests.test_training import FakeTask, FakeLearner
from tests.test_ppo_backend import AVAILABLE


class ChoosingLearner(FakeLearner):
    def __init__(self, task):
        super().__init__(task)
        self.calls = []

    def act(self, obs):
        self.calls.append(deepcopy(obs))
        n = len(self.calls) % 9
        return Decision(n, .25, -float(n))


def clone_metadata():
    return {"config": {"decision_contract": "attack-zone-held-v1", "features": "hazard-zone-v1",
                        "hold_frames": 4, "feature_options": {"half_width": 24}, "rollout_steps": 512},
            "reward_profile": "spell-v2", "timesteps": 0, "updates": 0}


class HeldTrainingTests(unittest.TestCase):
    def test_one_sample_controls_substeps_and_preserves_its_likelihood(self):
        task = FakeTask(length=5)
        actions = []
        step = task.step
        def record(action):
            actions.append(action)
            return step(action)
        task.step = record
        learner = ChoosingLearner(task)
        rows, stats = collect_held_episodes(task, learner, 2, lambda r: None)
        self.assertEqual(actions, [1, 1, 2, 2, 3])
        self.assertEqual([r.duration_frames for r in rows], [4, 4, 2])
        self.assertEqual([r.log_prob for r in rows], [-1, -2, -3])
        self.assertEqual([r.observation["player"][0] for r in rows], [0, 2, 4])
        self.assertEqual(stats["logic_frames"], 10)
        self.assertEqual(stats["substeps"], 5)
        self.assertTrue(task.menu)

    def test_early_terminal_resets_hold_and_does_not_step_again(self):
        task = FakeTask(length=1)
        learner = ChoosingLearner(task)
        rows, _ = collect_held_episodes(task, learner, 2, lambda r: None)
        self.assertEqual(task.resets, 2)
        self.assertEqual([r.duration_frames for r in rows], [2, 2])
        self.assertTrue(all(r.episode_start for r in rows))

    def test_overshoot_uses_measured_duration(self):
        task = FakeTask(length=3)
        original = task.step
        def step(action):
            obs, reward, done, truncated, info = original(action)
            info["delta_frames"] = 3
            return obs, reward, done, truncated, info
        task.step = step
        rows, stats = collect_held_episodes(task, ChoosingLearner(task), 2, lambda r: None)
        self.assertEqual([r.duration_frames for r in rows], [6, 3])
        self.assertEqual(stats["logic_frames"], 9)

    def test_frame_reward_and_intra_hold_discount(self):
        self.assertAlmostEqual(frame_reward({"delta_frames": 2, "reason": "spell_success"}, .5), .00075+2.5)
        task = FakeTask(length=2)
        rows, _ = collect_held_episodes(task, ChoosingLearner(task), 2, lambda r: None, gamma_frame=.5)
        self.assertAlmostEqual(rows[0].reward, .0005*(1+.5+.25+.125)+5*.125)
        with self.assertRaises(TrainingError):
            frame_reward({"delta_frames": 0, "reason": None}, .5)

    def test_duration_gae_with_different_lengths_and_episode_boundaries(self):
        rows = [HeldTransition({}, 0, r, start, v, -1, frames)
                for r, start, v, frames in ((1, True, .2, 2), (4, False, .3, 1),
                                            (10, True, 1, 2), (10, False, 2, 3))]
        advantages, returns = duration_gae(rows, .5, 1.)
        self.assertEqual(returns, [2., 4., 12.5, 10.])
        self.assertAlmostEqual(advantages[0], 1.8)

    def test_bad_or_truncated_batch_never_updates(self):
        for reason, truncated in (("scene_exit", False), ("time_limit", True)):
            task = FakeTask()
            task.step = lambda action: ({"player": [1]}, 0, not truncated, truncated,
                                        {"reason": reason, "delta_frames": 2})
            learner = ChoosingLearner(task)
            with self.assertRaises(TrainingError):
                train_episodic(task, learner, task.boundary, lambda s: None, lambda r: None,
                               2, collector=collect_held_episodes)
            self.assertFalse(learner.batches)

    def test_incomplete_hold_failure_does_not_update(self):
        task = FakeTask(length=4)
        original = task.step
        def step(action):
            if task.position == 1:
                raise RuntimeError("read failed")
            return original(action)
        task.step = step
        learner = ChoosingLearner(task)
        with self.assertRaisesRegex(RuntimeError, "read failed"):
            train_episodic(task, learner, task.boundary, lambda s: None, lambda r: None,
                           2, collector=collect_held_episodes)
        self.assertFalse(learner.batches)

    def test_held_training_menu_boundaries_and_additional_budget(self):
        task = FakeTask(length=3)
        learner = ChoosingLearner(task)
        learner.timesteps = 10
        states = []
        def save(state):
            task.boundary()
            states.append(state)
            return "held-checkpoint"
        train_episodic(task, learner, task.boundary, save, lambda r: None,
                       4, rollout_steps=2, completed_updates=3, collector=collect_held_episodes)
        self.assertEqual(learner.timesteps, 14)
        self.assertEqual(states[-1], {"updates": 5, "timesteps": 14})
        self.assertTrue(task.menu)

    def test_resume_rejects_unmarked_updates_or_changed_hyperparameters(self):
        saved = clone_metadata()
        config = held_config(saved)
        self.assertNotIn("rollout_steps", config)
        for invalid in (dict(saved, timesteps=1), dict(saved, config={**config, "held_ppo": {}})):
            with self.assertRaises(ValueError):
                held_config(invalid)
        continued = dict(saved, config=config, reward_profile=HELD_REWARD_PROFILE, timesteps=4, updates=1)
        self.assertEqual(held_config(continued), config)

    @unittest.skipUnless(AVAILABLE, "Install training dependencies")
    def test_cli_publishes_separate_held_checkpoints_and_preserves_source(self):
        from scripts import train_held
        from tests.fakes import FakeKeyboard
        class ContextTask(FakeTask):
            capacity = 4
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def verify_training_boundary(self):
                self.boundary()
        task = ContextTask(length=3)
        saved = clone_metadata()
        saved.update(build_md5="test-build", initialization={"method": "behavior_cloning",
                     "identity": {"task": "unit-held"}})
        saved["config"].update(capacity=4, max_frames=3600, seed=0)
        class Backend(ChoosingLearner):
            def __init__(self, env, **kwargs):
                super().__init__(task)
                self.model = SimpleNamespace(policy=SimpleNamespace(features_extractor=SimpleNamespace(half_width=24)))
            def configure(self, config):
                self.settings = config
            def save(self, path):
                Path(path).write_bytes(b"new model")
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            source = Path(directory)/"source"
            source.mkdir()
            (source/"model.zip").write_bytes(b"frozen source")
            output = Path(directory)/"new-run"
            reader = SimpleNamespace(addresses={"SPELL_RESULT_ACTIVE": 1}, close=lambda: None)
            args = ["train_held", "--resume", str(source), "--task", "unit-held",
                    "--decisions", "2", "--rollout-decisions", "2", "--run-dir", str(output)]
            with patch("sys.argv", args), patch("scripts.train_held.read_checkpoint", return_value=saved), \
                    patch("scripts.train_held.connect", return_value=(reader, 1, "test-build")), \
                    patch("scripts.train_held.WindowsKeyboard", return_value=FakeKeyboard()), \
                    patch("scripts.train_held.GameSession", return_value=task), \
                    patch("agent.held_backend.HeldPPOBackend", Backend), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                train_held.main()
            self.assertEqual((source/"model.zip").read_bytes(), b"frozen source")
            meta = json.loads((output/"checkpoints/update-000001/metadata.json").read_text())
            self.assertEqual(meta["timesteps"], 2)
            self.assertEqual(meta["reward_profile"], HELD_REWARD_PROFILE)
            self.assertEqual(meta["initialization"]["method"], "behavior_cloning")
            self.assertTrue((output/"training.console.log").exists())

    @unittest.skipUnless(AVAILABLE, "Install training dependencies")
    def test_actual_update_frame_returns_and_optimizer_checkpoint_resume(self):
        import gymnasium as gym
        import numpy as np
        import torch
        from agent.held_backend import HeldPPOBackend
        from agent.checkpoints import save_checkpoint, read_checkpoint
        from env.touhou_env import state_space
        class Offline(gym.Env):
            observation_space = state_space(4)
            action_space = gym.spaces.Discrete(9)
            def reset(self, **kwargs):
                raise AssertionError("No native reset")
        config = held_config(clone_metadata())
        learner = HeldPPOBackend(Offline(), features="hazard-zone-v1", feature_options={"half_width": 24})
        learner.configure(config["held_ppo"])
        rows = []
        for i in range(8):
            obs = {k: np.zeros(s.shape, dtype=s.dtype) for k, s in Offline.observation_space.spaces.items()}
            obs["player"][0] = i*.01
            d = learner.act(obs)
            rows.append(HeldTransition(obs, d.action, .1 if i%2 else -.1, i%4 == 0,
                                       d.value, d.log_prob, 2 if i%4 == 3 else 4))
            with torch.no_grad():
                tensor, _ = learner.model.policy.obs_to_tensor(obs)
                _, log_prob, _ = learner.model.policy.evaluate_actions(tensor, torch.tensor([d.action]))
            self.assertAlmostEqual(float(log_prob.item()), d.log_prob, places=5)
        initial = [p.detach().clone() for p in learner.model.policy.parameters()]
        _, expected = duration_gae(rows, config["held_ppo"]["gamma_frame"], config["held_ppo"]["lambda_frame"])
        learner.update(rows, 8)
        np.testing.assert_allclose(learner.model.rollout_buffer.returns.flatten(), expected, rtol=1e-5)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(initial, learner.model.policy.parameters())))
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = save_checkpoint(learner, Path(directory), {"timesteps": 8, "updates": 1,
                                   "reward_profile": HELD_REWARD_PROFILE}, config, "test-build")
            meta = read_checkpoint(path)
            self.assertEqual(meta["reward_profile"], HELD_REWARD_PROFILE)
            restored = HeldPPOBackend(Offline(), checkpoint=str(path/"model.zip"))
            optimizer = deepcopy(restored.model.policy.optimizer.state_dict())
            restored.configure(config["held_ppo"])
            self.assertEqual(len(restored.model.policy.optimizer.state_dict()["state"]), len(optimizer["state"]))
            self.assertEqual(restored.timesteps, 8)
            self.assertEqual(restored.model.policy.features_extractor.half_width, 24)
            self.assertEqual(restored.model.lr_schedule(1), 1e-5)
            self.assertEqual(restored.deterministic_action(rows[0].observation), learner.deterministic_action(rows[0].observation))
            restored.update(rows, 16)
            self.assertEqual(restored.timesteps, 16)
