"""Grid diagnostics are geometric, not survival or planning guarantees."""
import unittest
from agent.laser_room import partition_room
from tests.test_laser_geometry import laser


class LaserRoomTests(unittest.TestCase):
    def test_warning_wall_divides_components(self):
        beam=laser(0);beam['segment_candidate']=[[60.,0.],[60.,448.]];beam['width']=8.
        left=partition_room((30.,200.),[beam]);right=partition_room((200.,200.),[beam])
        self.assertGreater(right['area_pixels2'],left['area_pixels2'])
        self.assertGreater(left['area_pixels2'],0)
        self.assertLess(left['area_pixels2']+right['area_pixels2'],partition_room((30.,200.),[])['area_pixels2'])

    def test_faded_beam_does_not_split_and_blocked_seed_is_explicit(self):
        beam=laser(2);beam['segment_candidate']=[[60.,0.],[60.,448.]]
        self.assertEqual(partition_room((200.,200.),[beam])['area_pixels2'],partition_room((200.,200.),[])['area_pixels2'])
        beam['phase_raw']=1
        self.assertFalse(partition_room((60.,200.),[beam])['seed_free'])
        with self.assertRaises(ValueError):partition_room((0.,200.),[beam])
        beam['width']=float('nan')
        with self.assertRaises(ValueError):partition_room((60.,200.),[beam])
