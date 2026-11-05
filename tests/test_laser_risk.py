"""Joint geometric gating remains explicitly offline."""
import unittest
from agent.laser_risk import combined_candidates
from agent.risk_control import canonical_risk_observation, risk_candidates
from agent.boundary_control import feasible_actions
from tests.test_laser_geometry import laser
from tests.test_risk_control import scene


class CombinedLaserRiskTests(unittest.TestCase):
    def test_no_lasers_matches_existing_risk_contract(self):
        obs=scene(); state=canonical_risk_observation(obs)
        old=risk_candidates(state,feasible_actions(state,4))
        new=combined_candidates(obs,[])
        self.assertEqual(new['candidates'],old['risk_allowed_actions'])
        self.assertEqual(new['combined_clearances'],old['risk_clearances'])

    def test_laser_crossing_excludes_dangerous_direction(self):
        obs=scene(); obs['bullets']=[]
        # Player at (192,384), laser horizontal at y=364.
        beam=laser();beam['segment_candidate']=[[100.,364.],[280.,364.]]
        result=combined_candidates(obs,[beam])
        self.assertNotIn(1,result['candidates'])
        self.assertIn(5,result['candidates'])
        self.assertGreaterEqual(result['combined_clearances'][5],4)

    def test_unknown_laser_fails_closed(self):
        with self.assertRaises(ValueError):combined_candidates(scene(),[laser(3)])
