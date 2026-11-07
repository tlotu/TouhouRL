"""Swept collisions, time semantics and observation coverage diagnostics."""
import unittest

import numpy as np

from agent.forward_model import Circle, ForwardModel, Scene, TimedLaser, scene_from_observation


class ForwardModelTests(unittest.TestCase):
    def test_fast_circle_hits_between_safe_endpoints(self):
        scene = Scene((20., 20.), (Circle(20., 0., 0., 10., 2.),), bounds=(0, 40, 0, 40))
        model = ForwardModel(scene)
        self.assertTrue(model.edges([[20, 20]], [[20, 20]], 0, 0)[0][0])
        self.assertTrue(model.edges([[20, 20]], [[20, 20]], 4, 0)[0][0])
        valid, gap = model.edges([[20, 20]], [[20, 20]], 0, 4)
        self.assertFalse(valid[0])
        self.assertAlmostEqual(gap[0], -3.25)

    def test_player_segment_cannot_tunnel_through_laser(self):
        laser = TimedLaser(20, 0, 20, 40, 1, 0, 100)
        model = ForwardModel(Scene((16, 20), lasers=(laser,), bounds=(0, 40, 0, 40)))
        self.assertFalse(model.edges([[16, 20]], [[24, 20]], 0, 4)[0][0])
        self.assertTrue(model.edges([[16, 20]], [[16, 28]], 0, 4)[0][0])

    def test_warning_crossing_and_mid_edge_activation(self):
        late = TimedLaser(20, 0, 20, 40, 1, 4, 100)
        early = TimedLaser(20, 0, 20, 40, 1, 2, 100)
        scene = Scene((16, 20), lasers=(late,), bounds=(0, 40, 0, 40))
        self.assertTrue(ForwardModel(scene).edges([[16, 20]], [[24, 20]], 0, 4)[0][0])
        self.assertFalse(ForwardModel(Scene((16, 20), lasers=(early,), bounds=scene.bounds)).edges(
            [[16, 20]], [[24, 20]], 0, 4)[0][0])

    def test_fade_and_capsule_endpoints(self):
        scene = Scene((20, 20), lasers=(TimedLaser(20, 16, 20, 24, 2, 0, 8),), bounds=(0, 40, 0, 40))
        model = ForwardModel(scene)
        self.assertFalse(model.edges([[20, 14]], [[20, 14]], 0, 0)[0][0])
        self.assertTrue(model.edges([[20, 13]], [[20, 13]], 0, 0)[0][0])
        self.assertTrue(model.edges([[20, 20]], [[20, 20]], 8, 4)[0][0])

    def test_bounds_and_degenerate_static_segment(self):
        model = ForwardModel(Scene((8, 16)))
        self.assertTrue(model.edges([[8, 16]], [[8, 16]], 0, 4)[0][0])
        self.assertFalse(model.edges([[8, 16]], [[7, 16]], 0, 4)[0][0])
        laser = TimedLaser(20, 20, 20, 20, 2, 0, 8)
        scene = Scene((16, 20), lasers=(laser,), bounds=(0, 40, 0, 40))
        valid, gaps = ForwardModel(scene).edges([[16, 20], [16, 24]], [[24, 20], [24, 24]], 0, 4)
        self.assertEqual(valid.tolist(), [False, True])
        self.assertAlmostEqual(gaps[1], 1.75)

    def test_observation_decoding_and_saturation_are_explicit(self):
        obs = {'player': [0., 0., 0., 0.], 'bullets': [[.25, -.25, 1., 0., .25, 0., 1.]],
               'lasers': np.zeros((64, 12))}
        obs['lasers'][0] = [0, 0, 100, 0, 20, 0, 55, 60, 90, 16, 0, 1]
        scene, metadata = scene_from_observation(obs)
        self.assertEqual(scene.player, (192., 224.))
        self.assertEqual(scene.circles[0], Circle(288, 112, 16, 0, 8))
        self.assertEqual((scene.lasers[0].on, scene.lasers[0].off), (5, 95))
        self.assertEqual(metadata['saturated_circle_rows'], 1)
        self.assertFalse(metadata['future_births_modeled'])
        obs['lasers'][0, 10] = 1
        with self.assertRaises(ValueError):
            scene_from_observation(obs)

    def test_float32_field_boundary_roundoff_is_not_out_of_bounds(self):
        obs = {'player': [(8 / 192) - 1, (16 / 224) - 1, 0, 0],
               'bullets': np.zeros((1, 7)), 'lasers': np.zeros((64, 12))}
        scene, _ = scene_from_observation(obs)
        self.assertTrue(ForwardModel(scene).edges([scene.player], [scene.player], 0, 4)[0][0])

    def test_invalid_geometry_and_time_fail_explicitly(self):
        with self.assertRaises(ValueError):
            Circle(0, 0, float('nan'), 0, 1)
        with self.assertRaises(ValueError):
            TimedLaser(0, 0, 0, 0, 2, 3, 2)
        with self.assertRaises(ValueError):
            Scene((500, 0))
        model = ForwardModel(Scene((8, 16)))
        for duration in (-1, float('inf')):
            with self.assertRaises(ValueError):
                model.edges([[8, 16]], [[8, 16]], 0, duration)
        with self.assertRaises(ValueError):
            model.edges([[8, 16]], [[9, 16]], 0, 0)
