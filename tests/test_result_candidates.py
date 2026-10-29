import struct
import unittest
from types import SimpleNamespace

from env.menu_probe import capture_regions, result_candidates


class ResultCandidateTests(unittest.TestCase):
    def test_build_specific_menu_offsets_decode_without_classifying(self):
        state = bytearray(0x260)
        struct.pack_into("<i", state, 4, 2)
        struct.pack_into("<i", state, 0x254, 1)
        state[0x258] = 1
        records = {"spell_result_active_candidate": {"hex": "01"},
                   "spell_result_state_candidate": {"hex": state.hex()}}
        result = result_candidates(records)
        self.assertEqual(result, {"available": True, "verified": False,
                                 "active": 1, "selection": 2, "phase": 1, "outcome_flag": 1})
        self.assertNotIn("success", result)

    def test_missing_or_short_data_is_unavailable(self):
        self.assertFalse(result_candidates({})["available"])
        regions = {"spell_result_active_candidate": {"hex": "01"},
                   "spell_result_state_candidate": {"hex": "00"}}
        self.assertFalse(result_candidates(regions)["available"])

    def test_probe_reads_exact_candidate_neighborhoods(self):
        calls = []
        def read(addr, size):
            calls.append((addr, size))
            return bytes(size)
        reader = SimpleNamespace(base_address=0x100000000, process=SimpleNamespace(read=read),
                                 addresses={"LIVES": 0x100, "PLAYER_BASE": 0x1000,
                                            "SPELL_RESULT_ACTIVE_CANDIDATE": 0x4F27B6,
                                            "SPELL_RESULT_MENU_CANDIDATE": 0x3E5430})
        regions = capture_regions(reader)
        self.assertIn((0x1004F27B6, 1), calls)
        self.assertIn((0x1003E5F70, 0x260), calls)
        self.assertTrue(result_candidates(regions)["available"])
