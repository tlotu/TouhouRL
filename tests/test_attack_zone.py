"""Attack bands change aiming preferences only within explicit clearance gates."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agent.attack_zone import AttackZoneConfig, AttackZonePolicy, rank_attack_zone
from agent.baselines import action_candidates, attack_target, reactive_attack
from env.memory_reader import Bullet
from env.session import observation
from tests.fakes import sample
from tests.test_attack_policy import boss


class AttackZoneTests(unittest.TestCase):
    def obs(self, dx=80):
        return observation(sample(y=384, enemies=(boss(x=192+dx),)))

    def test_inside_band_does_not_prefer_center_and_tracks_both_sides(self):
        for dx in (-20, 20):
            result = rank_attack_zone(self.obs(dx), AttackZoneConfig(24))
            self.assertEqual(result["action"], 0)
            stay = next(c for c in result["candidates"] if c["action"] == 0)
            self.assertEqual(stay["zone_excess"], 0)
            self.assertNotEqual(reactive_attack(self.obs(dx)), 0)
        self.assertEqual(rank_attack_zone(self.obs(80), AttackZoneConfig())["action"], 3)
        self.assertEqual(rank_attack_zone(self.obs(-80), AttackZoneConfig())["action"], 7)

    def test_safe_second_best_clearance_can_approach(self):
        candidates = [(7, -2, 0, 24, 23.85), (3, 2, 0, 23.5, 23.35)]
        with patch("agent.attack_zone.action_candidates", return_value=candidates):
            result = rank_attack_zone(self.obs(), AttackZoneConfig())
        self.assertEqual(result["action"], 3)
        self.assertEqual(result["threshold"], 23)

    def test_relative_and_absolute_gates_each_reject_approach(self):
        for candidates in ([(7, -2, 0, 20, 20), (3, 2, 0, 18, 18)],
                           [(7, -2, 0, 4.5, 4.5), (3, 2, 0, 3.9, 3.9)]):
            with self.subTest(candidates=candidates):
                with patch("agent.attack_zone.action_candidates", return_value=candidates):
                    result = rank_attack_zone(self.obs(), AttackZoneConfig())
                self.assertEqual(result["action"], 7)
                self.assertFalse(result["candidates"][1]["eligible"])

    def test_all_dangerous_candidates_ignore_attack_preference(self):
        with patch("agent.attack_zone.action_candidates", return_value=[
                (7, -2, 0, 2, 2), (3, 2, 0, 1.9, 1.9)]):
            result = rank_attack_zone(self.obs(), AttackZoneConfig())
        self.assertEqual(result["mode"], "escape")
        self.assertEqual(result["action"], 7)

    def test_zero_loss_requires_best_clearance(self):
        with patch("agent.attack_zone.action_candidates", return_value=[
                (7, -2, 0, 24, 24), (3, 2, 0, 23.99, 23.99)]):
            result = rank_attack_zone(self.obs(), AttackZoneConfig(max_clearance_loss=0))
        self.assertEqual(result["action"], 7)

    def test_wall_and_missing_target_and_empty_candidates(self):
        obs = observation(sample(x=8, y=384))
        result = rank_attack_zone(obs, AttackZoneConfig())
        self.assertEqual(result["mode"], "no_target")
        self.assertNotIn(7, [c["action"] for c in result["candidates"]])
        with patch("agent.attack_zone.action_candidates", return_value=[]):
            result = rank_attack_zone(obs, AttackZoneConfig())
        self.assertEqual(result["mode"], "no_valid_direction")
        self.assertEqual(result["action"], 0)

    def test_legacy_width_matches_old_teacher_for_targeted_geometry(self):
        for px in (8, 120, 192, 376):
            for bx in (50, 192, 300):
                obs = observation(sample(x=px, y=384, enemies=(boss(x=bx),),
                                  bullets=(Bullet(0, 194, 360, 0, 2, 3, 1),)))
                self.assertIsNotNone(attack_target(obs))
                result = rank_attack_zone(obs, AttackZoneConfig(6))
                self.assertEqual(result["action"], reactive_attack(obs))
                best = max(c[3] for c in action_candidates(obs))
                chosen = next(c for c in result["candidates"] if c["action"] == result["action"])
                self.assertGreaterEqual(chosen["clearance"], max(4, best-1) if best >= 4 else best)

    def test_invalid_configuration_and_rule_callback(self):
        for kwargs in ({"half_width": -1}, {"half_width": float("nan")},
                       {"min_clearance": 25}, {"max_clearance_loss": -1}):
            with self.assertRaises(ValueError):
                AttackZoneConfig(**kwargs)
        policy = AttackZonePolicy(AttackZoneConfig())
        self.assertEqual(policy(self.obs(20)), policy.last["action"])

    def test_new_evaluation_cannot_enter_legacy_cloning_as_teacher(self):
        from agent.demonstrations import load_demonstrations
        from tests.test_demonstrations import trace_rows, write_trace
        rows = list(trace_rows())
        rows[0].update(policy="attack_zone", collection="attack_zone_evaluation")
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / "zone.jsonl"
            write_trace(path, rows)
            with self.assertRaises(ValueError):
                load_demonstrations([path])
