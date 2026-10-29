import unittest
from unittest.mock import patch

from agent.baselines import attack_target, make_policy, reactive, reactive_attack
from env.action_emulator import DIRECTIONS
from env.memory_reader import Bullet, Enemy
from env.session import ENEMY_CAPACITY, observation
from tests.fakes import sample


def boss(x=260, y=100, flags2=10, flags3=0, slot=0):
    return Enemy(slot, x, y, 500, 128, flags2, flags3)


class AttackPolicyTests(unittest.TestCase):
    def test_no_target_matches_original_reactive(self):
        obs = observation(sample(y=224))
        self.assertEqual(reactive_attack(obs), reactive(obs))

    def test_aligns_both_directions_and_tracks_moving_target(self):
        for x, direction in ((260, "right"), (100, "left")):
            with self.subTest(x=x):
                obs = observation(sample(enemies=(boss(x=x),)))
                self.assertIn(direction, DIRECTIONS[reactive_attack(obs)])

    def test_stays_when_aligned(self):
        obs = observation(sample(enemies=(boss(x=192),)))
        self.assertEqual(reactive_attack(obs), 0)

    def test_hidden_noncollidable_and_rear_targets_are_ignored(self):
        for enemy in (boss(flags3=8), boss(flags2=8), boss(y=420)):
            with self.subTest(enemy=enemy):
                obs = observation(sample(enemies=(enemy,)))
                self.assertIsNone(attack_target(obs))
                self.assertEqual(reactive_attack(obs), reactive(obs))

    def test_boss_precedes_closer_regular_enemy(self):
        obs = observation(sample(enemies=(boss(x=193, flags2=2, slot=1), boss())))
        self.assertAlmostEqual(attack_target(obs)[0] * 384, 68)

    def test_attack_never_selects_a_path_outside_safety_tolerance(self):
        obs = observation(sample(enemies=(boss(),)))
        candidates = [(7, -2, 0, 20, 20), (3, 2, 0, 18, 18)]
        with patch("agent.baselines.action_candidates", return_value=candidates):
            self.assertEqual(reactive_attack(obs), 7)

    def test_danger_uses_clearance_instead_of_attacking(self):
        obs = observation(sample(enemies=(boss(),)))
        candidates = [(7, -2, 0, 2, 2), (3, 2, 0, -1, -1)]
        with patch("agent.baselines.action_candidates", return_value=candidates):
            self.assertEqual(reactive_attack(obs), 7)

    def test_selected_path_is_safe_under_the_same_bullet_model(self):
        from agent.baselines import action_candidates
        obs = observation(sample(y=224, enemies=(boss(x=260),),
                                 bullets=(Bullet(0, 204, 210, 0, 2, 3, 1),)))
        candidates = action_candidates(obs)
        selected = next(c for c in candidates if c[0] == reactive_attack(obs))
        best_gap = max(c[3] for c in candidates)
        self.assertGreaterEqual(selected[3], max(4, best_gap - 1) if best_gap >= 4 else best_gap)

    def test_enemy_padding_and_capacity(self):
        obs = observation(sample(enemies=(boss(),)))
        self.assertEqual(len(obs["enemies"]), ENEMY_CAPACITY)
        self.assertEqual(obs["enemies"][1], [0.0] * 6)
        self.assertEqual(make_policy("reactive_attack")(obs), reactive_attack(obs))

    def test_too_close_target_does_not_encourage_upward_chase(self):
        obs = observation(sample(y=224, enemies=(boss(x=192, y=210),)))
        self.assertIn("down", DIRECTIONS[reactive_attack(obs)])
