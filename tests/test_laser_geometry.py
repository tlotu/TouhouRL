"""Static forecast hypothesis remains separate from native training."""
import unittest
from agent.laser_geometry import candidate_active, forecast_static_lasers, point_segment_distance


def laser(phase=1,timer=0):
    return {'phase_raw':phase,'timer_raw':timer,'warning_frames_raw':60,
        'firing_frames_raw':90,'fading_frames_raw':16,'width':20.,'speed':0.,
        'segment_candidate':[[0.,0.],[100.,0.]],'issues':[]}


class LaserGeometryTests(unittest.TestCase):
    def test_segment_projection_and_degenerate_endpoint(self):
        self.assertEqual(point_segment_distance(50,10,0,0,100,0),10)
        self.assertEqual(point_segment_distance(103,4,0,0,100,0),5)
        self.assertEqual(point_segment_distance(3,4,0,0,0,0),5)

    def test_warning_activation_and_firing_end(self):
        warning=laser(0,55)
        self.assertFalse(candidate_active(warning,4))
        self.assertTrue(candidate_active(warning,5))
        firing=laser(1,88)
        self.assertTrue(candidate_active(firing,1))
        self.assertFalse(candidate_active(firing,2))
        self.assertFalse(candidate_active(laser(2),0))

    def test_forecast_avoids_crossing_and_respects_activation(self):
        gaps=forecast_static_lasers((50.,20.),[laser()])
        self.assertLess(gaps[1],0)
        self.assertGreater(gaps[5],0)
        self.assertEqual(forecast_static_lasers((50.,20.),[laser(0,0)]),[24.]*9)
        self.assertEqual(forecast_static_lasers((50.,20.),[]),[24.]*9)

    def test_unknown_phase_and_motion_fail_closed(self):
        with self.assertRaises(ValueError):forecast_static_lasers((1,1),[laser(3)])
        moving=laser();moving['speed']=1.
        with self.assertRaises(ValueError):forecast_static_lasers((1,1),[moving])
