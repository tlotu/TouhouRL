"""Held teacher provenance, fresh labels and versioned feature round trips."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import contextlib
import io
from unittest.mock import patch

from agent.action_commitment import ActionCommitment
from agent.attack_zone import AttackZoneConfig, rank_attack_zone
from agent.zone_demonstrations import load_zone_demonstrations
from env.session import OBSERVATION_SCHEMA
from env.movement_probe import BASELINE_BUILD
from tests.test_demonstrations import write_trace
from tests.test_ppo_backend import AVAILABLE


def zone_rows(width=24):
    config = AttackZoneConfig(width)
    rows = [{"event": "connected", "schema": OBSERVATION_SCHEMA, "policy": "attack_zone",
             "collection": "attack_zone_evaluation", "repeat": 2, "trace": True, "run": True,
             "task": "synthetic-zone", "build_md5": BASELINE_BUILD, "capacity": 2,
             "max_frames": 3600, "hold_frames": 4, "prediction_frames": 12,
             "aim_projection_frames": 4, "attack_zone": vars(config)}]
    elapsed = [0]
    holding = ActionCommitment(lambda obs: rank_attack_zone(obs, config)["action"], lambda: elapsed[0])
    for episode in range(2):
        reset = {"event": "reset", "episode": episode}
        rows.append(holding.annotate(reset))
        observations = []
        for dx in (80, -80, -80, 20, 20):
            obs = {"player": [0, .7, 0, 0], "bullets": [[0.]*7 for _ in range(2)],
                   "enemies": [[0.]*6 for _ in range(16)]}
            obs["enemies"][0] = [dx/384, -.5, 1, 1, 1, 1]
            observations.append(obs)
        for step, obs in enumerate(observations[:-1]):
            elapsed[0] = step*2
            action = holding(obs)
            row = holding.annotate({"event": "transition", "episode": episode, "step": step,
                                   "observation": obs, "next_observation": observations[step+1],
                                   "action": action, "terminated": step == 3, "truncated": False,
                                   "info": {"reason": "spell_success" if step == 3 else None,
                                            "delta_frames": 2, "elapsed_frames": (step+1)*2,
                                            "frame": step+1, "player_x": 192}})
            row["raw_rule_action"] = row.pop("raw_student_action")
            row["rule_ranking"] = rank_attack_zone(obs, config)
            row["movement_diagnostic"] = {"requested_slow": True, "owned_keys": ["shift"],
                 "os_shift": {"shift": True}, "game_candidates": {"available": True,
                 "plausible": True, "same_transition_frame": True, "slow_input_bit": True,
                 "focus_raw": 1, "frame_before": step+1, "frame_after": step+1,
                 "input_mask": 4, "speed_parameters": {"slow": 2, "diagonal_slow_axis": 2**.5}}}
            rows.append(row)
        rows.append({"event": "episode_end", "episode": episode, "steps": 4, "reason": "spell_success"})
    return rows


class ZoneDemonstrationTests(unittest.TestCase):
    def load(self, rows):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory)/"zone.jsonl"
            write_trace(path, rows)
            return load_zone_demonstrations([path])

    def test_only_fresh_decisions_are_labels(self):
        episodes, metadata = self.load(zone_rows())
        self.assertEqual([e["actions"] for e in episodes], [[3, 7], [3, 7]])
        self.assertEqual(episodes[0]["frames"], [0, 4])
        self.assertEqual(metadata["identity"]["hold_frames"], 4)
        self.assertEqual(metadata["sources"][0]["original_collection"], "attack_zone_evaluation")

    def test_corrupt_held_provenance_or_teacher_is_rejected(self):
        for key, value in (("action", 7), ("raw_rule_action", 3), ("hold_age_frames", 0)):
            rows = zone_rows()
            rows[3][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.load(rows)

    def test_missing_or_stale_slow_telemetry_is_rejected(self):
        for mode in ("missing", "stale", "fast", "build"):
            rows = zone_rows()
            if mode == "missing":
                del rows[2]["movement_diagnostic"]
            elif mode == "stale":
                rows[2]["movement_diagnostic"]["game_candidates"]["frame_after"] = 99
            elif mode == "fast":
                rows[2]["movement_diagnostic"]["game_candidates"]["speed_parameters"]["slow"] = 4
            else:
                rows[0]["build_md5"] = "other-build"
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.load(rows)

    def test_broken_chain_and_incomplete_boundary(self):
        rows = zone_rows()
        rows[3]["observation"] = deepcopy(rows[3]["observation"])
        rows[3]["observation"]["player"][0] = .1
        with self.assertRaises(ValueError):
            self.load(rows)
        episodes, meta = self.load(zone_rows()[:-1])
        self.assertEqual(len(episodes), 1)
        self.assertEqual(meta["incomplete_episodes"], 1)

    def test_duplicate_and_different_widths_cannot_mix(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            paths = [Path(directory)/f"zone{i}.jsonl" for i in range(2)]
            for path, width in zip(paths, (24, 6)):
                write_trace(path, zone_rows(width))
            with self.assertRaises(ValueError):
                load_zone_demonstrations(paths)
            with self.assertRaises(ValueError):
                load_zone_demonstrations([paths[0], paths[0]])

    def test_legacy_ppo_entry_rejects_zone_contract_before_connecting(self):
        from scripts import train
        saved = {"config": {"decision_contract": "attack-zone-held-v1"}}
        with patch("sys.argv", ["train", "--resume", "dummy", "--evaluate"]), \
                patch("scripts.train.read_checkpoint", return_value=saved), \
                patch("scripts.train.connect") as connect, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as exc:
                train.main()
            self.assertEqual(exc.exception.code, 2)
            connect.assert_not_called()

    @unittest.skipUnless(AVAILABLE, "Install training dependencies")
    def test_actual_fitting_feature_width_and_archive_reload(self):
        import gymnasium as gym
        import numpy as np
        import torch
        from agent.cloning import fit_actor
        from agent.demonstrations import stack_episodes
        from agent.ppo_backend import PPOBackend
        from env.touhou_env import state_space
        from agent.hazard_features import HazardFeatures
        class Offline(gym.Env):
            observation_space = state_space(2)
            action_space = gym.spaces.Discrete(9)
        episodes, _ = self.load(zone_rows())
        data = stack_episodes(episodes)
        learner = PPOBackend(Offline(), features="hazard-zone-v1", feature_options={"half_width": 24})
        obs = {k: np.asarray(v[0]) for k, v in data[0].items()}
        tensors = {k: torch.tensor(v[:1]) for k, v in data[0].items()}
        new_features = learner.model.policy.features_extractor(tensors)
        old_features = HazardFeatures(Offline.observation_space)(tensors)
        self.assertFalse(torch.equal(new_features, old_features))
        initial = deepcopy(learner.model.policy.state_dict())
        fit_actor(learner, data, data, lambda row: None, epochs=3)
        self.assertTrue(any(not torch.equal(initial[k], v) for k, v in learner.model.policy.state_dict().items()))
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory)/"model.zip"
            learner.save(path)
            restored = PPOBackend(Offline(), checkpoint=str(path))
            self.assertEqual(restored.model.policy.features_extractor.half_width, 24)
            self.assertEqual(restored.deterministic_action(obs), learner.deterministic_action(obs))
