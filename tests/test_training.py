import unittest

from agent.training import Decision, TrainingError, collect_episodes, train_episodic, training_reward


class FakeTask:
    def __init__(self, length=3, reason="spell_success"):
        self.menu = True
        self.length, self.reason = length, reason
        self.resets = self.position = 0

    def reset(self):
        if not self.menu:
            raise RuntimeError("Reset during combat")
        self.menu = False
        self.resets += 1
        self.position = 0
        return {"player": [0.0]}, {}

    def step(self, action):
        self.position += 1
        done = self.position == self.length
        self.menu = done
        info = {"reason": self.reason if done else None,
                "delta_frames": 2, "elapsed_frames": self.position * 2}
        return {"player": [float(self.position)]}, 0.1, done, False, info

    def boundary(self):
        if not self.menu:
            raise RuntimeError("Update during combat")


class FakeLearner:
    def __init__(self, task):
        self.task = task
        self.timesteps = 0
        self.batches = []

    def act(self, obs):
        return Decision(0, 0.5, -1.5)

    def update(self, transitions, target):
        self.task.boundary()
        self.batches.append(transitions)
        self.timesteps += len(transitions)
        return {"test_loss": 0.1}


class TrainingTests(unittest.TestCase):
    def test_evaluation_callback_uses_no_optimizer_updates(self):
        from scripts.diagnostics import run_episodes
        task = FakeTask(length=2)
        learner = FakeLearner(task)
        rows = []
        run_episodes(task, 1, rows.append, policy="ppo_deterministic", action_fn=lambda obs: 3)
        self.assertFalse(learner.batches)
        self.assertEqual(rows[-1]["policy"], "ppo_deterministic")
        self.assertEqual(rows[-1]["successful_episodes"], 1)
        ended = next(row for row in rows if row["event"] == "episode_end")
        self.assertEqual(ended["action_counts"], [0, 0, 0, 2, 0, 0, 0, 0, 0])

    def test_reward_prioritizes_spell_outcome(self):
        self.assertAlmostEqual(training_reward({"delta_frames": 2, "reason": "spell_success"}), 5.001)
        self.assertAlmostEqual(training_reward({"delta_frames": 2, "reason": "spell_failure"}), -0.999)
        self.assertEqual(training_reward({"delta_frames": 2, "reason": None}), 0.001)

    def test_longer_survival_improves_failed_episode_return(self):
        def failed_return(frames):
            return sum(training_reward({"delta_frames": 2, "reason": None})
                       for _ in range(frames // 2 - 1)) + training_reward(
                           {"delta_frames": 2, "reason": "spell_failure"})
        self.assertGreater(failed_return(900), failed_return(438))
        # Even the longest default failure earns less than an immediate success.
        self.assertLess(failed_return(3600), training_reward(
            {"delta_frames": 0, "reason": "spell_success"}))

    def test_collection_finishes_episode_and_marks_boundaries(self):
        task = FakeTask()
        learner = FakeLearner(task)
        batch, stats = collect_episodes(task, learner, 4, lambda row: None)
        self.assertEqual(len(batch), 6)
        self.assertEqual([r.episode_start for r in batch], [True, False, False, True, False, False])
        self.assertEqual([r.observation["player"][0] for r in batch], [0, 1, 2, 0, 1, 2])
        self.assertEqual(stats["successes"], 2)
        self.assertTrue(task.menu)
        self.assertEqual(learner.timesteps, 0)

    def test_training_updates_only_at_menus_and_stops_without_reset(self):
        task = FakeTask()
        learner = FakeLearner(task)
        checkpoints, events = [], []
        def save(state):
            task.boundary()
            checkpoints.append(dict(state))
            return "fake-checkpoint"
        train_episodic(task, learner, task.boundary, save, events.append,
                       additional_steps=8, rollout_steps=4)
        self.assertEqual(learner.timesteps, 9)
        self.assertEqual([len(b) for b in learner.batches], [6, 3])
        self.assertEqual(task.resets, 3)
        self.assertEqual(checkpoints[-1], {"updates": 2, "timesteps": 9})
        self.assertTrue(task.menu)

    def test_unresolved_ending_is_not_trained(self):
        task = FakeTask(reason="scene_exit")
        learner = FakeLearner(task)
        with self.assertRaises(TrainingError):
            train_episodic(task, learner, task.boundary, lambda state: None,
                           lambda row: None, 4)
        self.assertFalse(learner.batches)

    def test_truncation_is_not_mislabeled_as_terminal(self):
        task = FakeTask()
        task.step = lambda action: ({"player": [1]}, 0, False, True,
                                    {"reason": "time_limit", "delta_frames": 2})
        with self.assertRaises(TrainingError):
            collect_episodes(task, FakeLearner(task), 2, lambda row: None)

    def test_missing_boundary_prevents_optimizer_update(self):
        task = FakeTask()
        learner = FakeLearner(task)
        def invalid():
            raise RuntimeError("menu lost")
        with self.assertRaisesRegex(RuntimeError, "menu lost"):
            train_episodic(task, learner, invalid, lambda state: None, lambda row: None, 4)
        self.assertFalse(learner.batches)

    def test_resume_budget_is_additional(self):
        task = FakeTask(length=2)
        learner = FakeLearner(task)
        learner.timesteps = 100
        events = []
        train_episodic(task, learner, task.boundary, lambda state: "checkpoint", events.append,
                       4, rollout_steps=4, completed_updates=5)
        self.assertEqual(learner.timesteps, 104)
        self.assertEqual(events[-1]["updates"], 6)

    def test_menu_lost_during_update_prevents_checkpoint_publication(self):
        task = FakeTask(length=2)
        class ChangedMenu(FakeLearner):
            def update(self, transitions, target):
                metrics = super().update(transitions, target)
                self.task.menu = False
                return metrics
        learner = ChangedMenu(task)
        published = []
        with self.assertRaises(RuntimeError):
            train_episodic(task, learner, task.boundary, published.append,
                           lambda row: None, additional_steps=2, rollout_steps=2)
        self.assertFalse(published)
