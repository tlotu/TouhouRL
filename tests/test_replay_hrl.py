"""Check feedback and observation timing in the log-driven environment."""
import unittest

import numpy as np
import torch
from gymnasium import spaces

from agent.direction_buffer import DirectionBuffer
from agent.forward_model import Circle, Scene, TimedLaser
from agent.replay_exploration import ReplayExplorationPolicy
from agent.replay_async import AsyncReplayPlanner, fallback_hold
from agent.replay_connectivity import component_area, contraction
from agent.replay_hrl import FEATURES, ReplayEpisode, ReplayManagerEnv, ReplaySample, ReplayWorker, rebase, routes_contract
from agent.short_planner import LONG_HORIZON, plan
from scripts.evaluate_replay_escape_gate import ContractionGate


class ReplayManagerTests(unittest.TestCase):
    def test_contraction_gate_checks_lead_and_releases_at_settling_frame(self):
        gate = ContractionGate(0)
        self.assertFalse(gate.update(0, lambda: {'start': 36, 'end': 48, 'reason': 'stable'}))
        self.assertFalse(gate.update(12, lambda: self.fail('Scan ran early')))
        self.assertTrue(gate.update(16, lambda: {'start': 12, 'end': 28, 'reason': 'stable'}))
        self.assertTrue(gate.update(40, lambda: self.fail('Active window was rescanned')))
        self.assertFalse(gate.update(44, lambda: self.fail('Gate did not release')))
        self.assertFalse(gate.update(48, lambda: {'start': 4, 'end': 400, 'reason': 'capped'}))

    def make_env(self, circle_frame=None):
        recorded = (192., 224.)
        obs = {'player': np.zeros(4, dtype=np.float32),
               'enemies': np.zeros((16, 6), dtype=np.float32),
               'bullets': np.zeros((128, 7), dtype=np.float32)}
        samples = {frame: ReplaySample(obs, Scene(recorded, (Circle(198., 224., 0., 0., 1.),)
                                                  if frame == circle_frame else ()), 0)
                   for frame in range(0, 12, 2)}
        episode = ReplayEpisode('test', 1, samples, {0: frozenset({DirectionBuffer()})}, 12)
        return ReplayManagerEnv([episode], ReplayWorker(lambda _: np.zeros(9)), verify=True)

    def test_next_search_starts_from_observed_simulated_endpoint(self):
        env = self.make_env()
        env.reset(options={'episode': 0, 'delay': 1})
        obs, _, done, _, info = env.step(3)
        self.assertFalse(done)
        self.assertEqual(info['history'][0]['request'], 3)
        self.assertEqual(env.scene.player, (198., 224.))
        self.assertAlmostEqual(obs[0], 198. / 192 - 1.)
        env.step(7)
        self.assertEqual(env.history[-1]['start'], [198., 224.])
        self.assertEqual(env.history[-1]['request'], 7)

    def test_hazard_refresh_inside_hold_can_terminate_verified_route(self):
        env = self.make_env(circle_frame=2)
        env.reset(options={'episode': 0, 'delay': 0})
        _, _, done, _, info = env.step(3)
        self.assertTrue(done)
        self.assertEqual(info['outcome'], 'collision')
        self.assertEqual(info['frames'], 2 + 1)

    def test_delayed_rewards_use_observed_continuation_and_margin(self):
        env = self.make_env()
        env.reset(options={'episode': 0, 'delay': 0})
        env.delayed_checks.append(0)
        env.forecasts.extend([0, 4])
        env.frame = 31
        self.assertEqual(env._long_reward(None), (0., 0.))
        env.frame = 32
        env.choices = ()
        self.assertEqual(env._long_reward(None), (-.06, 0.))
        env.frame = 60
        self.assertEqual(env._long_reward(None), (0., .04))
        env.frame = 64
        env.last_padded_contact = 63
        self.assertEqual(env._long_reward(None), (0., 0.))

    def test_route_contraction_and_laser_warning(self):
        self.assertFalse(routes_contract((8, 60.), (7, 50.), 100.))
        self.assertTrue(routes_contract((8, 60.), (6, 60.), 100.))
        self.assertTrue(routes_contract((8, 60.), (8, 48.), 100.))
        self.assertTrue(routes_contract(None, (3, 60.), 44.))
        self.assertFalse(routes_contract(None, (4, 60.), 44.))

    def test_long_search_keeps_the_requested_first_direction(self):
        result = plan(Scene((192., 224.)), width=9, horizon_override=LONG_HORIZON,
                      first_action=5, input_buffer=DirectionBuffer(), switch_delay=1)
        self.assertTrue(result.candidates)
        self.assertTrue(all(len(route.actions) == 15 and route.actions[0] == 5
                            for route in result.candidates))

    def test_early_exit_preserves_full_route_for_independent_verification(self):
        env = self.make_env()
        env.worker = ReplayWorker(lambda _: np.zeros(9), early_exit=True)
        env.reset(options={'episode': 0, 'delay': 1})
        self.assertEqual(env.edges, [0])
        self.assertGreaterEqual(len({c.route.actions[0] for c in env.choices}), 3)
        _, _, done, _, info = env.step(3)
        self.assertFalse(done)
        self.assertEqual(info['history'][0]['request'], 3)

    def test_direction_options_only_map_to_their_verified_first_action(self):
        env = self.make_env()
        env.reset(options={'episode': 0, 'delay': 0})
        obs = env.episode.samples[0].observation
        choices, _, _ = env.worker.options(env.scene, obs, env.belief, by_direction=True)
        self.assertEqual(set(choices), set(range(9)))
        self.assertTrue(all(choice.route.actions[0] == action for action, choice in choices.items()))
        waypoints, _, _ = env.worker.options(env.scene, obs, env.belief, waypoint_coverage=True)
        self.assertEqual(waypoints[0].route, env.choices[0].route)
        self.assertEqual({choice.route.actions[0] for choice in waypoints}, set(range(9)))
        self.assertEqual(len(waypoints), len(env.choices))

    def test_async_route_is_checked_against_new_hazards(self):
        env = self.make_env()
        env.reset(options={'episode': 0, 'delay': 0})
        obs = env.episode.samples[0].observation
        planner = AsyncReplayPlanner(env.worker)
        try:
            self.assertTrue(planner.submit(0, env.scene, obs, env.belief, 0))
            planner.pending[1].result(timeout=5)
            self.assertEqual(planner.consume(2, env.scene, obs, env.belief)[1], 'stale')
            blocked = Scene(env.scene.player, (Circle(*env.scene.player, 0., 0., 3.),))
            candidate, status = planner.consume(4, blocked, rebase(obs, blocked.player), env.belief)
            self.assertIsNone(candidate)
            self.assertEqual(status, 'recheck_failed')
        finally:
            planner.close()

    def test_fallback_checks_buffered_hold_without_a_worker(self):
        env = self.make_env(circle_frame=0)
        obs = env.episodes[0].samples[0].observation
        scene = env.episodes[0].samples[0].scene
        choice = fallback_hold(scene, obs, frozenset({DirectionBuffer()}), 3)
        self.assertIsNotNone(choice)
        self.assertEqual(len(choice.route.actions), 1)
        self.assertNotIn(choice.route.actions[0], (2, 3, 4))

    def test_connectivity_detects_barrier_between_free_grid_centers(self):
        scene = Scene((12., 24.), lasers=(TimedLaser(32., 0., 32., 64., 1., 8., 40.),),
                      bounds=(0., 64., 0., 64.))
        samples = component_area(scene, horizon=32)
        self.assertGreater(samples[0][1], samples[2][1])
        self.assertEqual(contraction(samples), {'start': 8, 'end': 20, 'reason': 'stable'})

    def test_exploration_reuses_rollout_likelihood_and_deduplicates_moves(self):
        policy = ReplayExplorationPolicy(spaces.Box(-1., 1., (FEATURES,), dtype=np.float32),
                                         spaces.Discrete(9), lambda _: .001, exploration=.2)
        policy.action_net.weight.data.zero_()
        policy.action_net.bias.data.zero_()
        obs = torch.zeros((2, FEATURES))
        obs[0, -63 + 6::7] = torch.tensor([0., 1., 1., .5, 0., 0., 0., 0., 0.])
        _, log_probs, _ = policy.evaluate_actions(obs, torch.tensor([1, 1]))
        torch.testing.assert_close(log_probs.exp(), torch.tensor([.8 / 9 + .1, 1 / 9]))
        actions, _, rollout_log_prob = policy(obs)
        _, update_log_prob, _ = policy.evaluate_actions(obs, actions)
        torch.testing.assert_close(rollout_log_prob, update_log_prob)
