import struct
from types import SimpleNamespace
import unittest

from env.menu_probe import capture_regions, stable_differences


def row(data, label="left", build="test"):
    return {"label": label, "build_md5": build,
            "regions": {"status": {"hex": bytes(data).hex()}}}


class MenuProbeTests(unittest.TestCase):
    def test_identical_labeled_menu_bytes_produce_no_candidates(self):
        # Observed success_default versus success_retry captures gave no change.
        result = stable_differences([row([0, 1, 2])] * 20,
                                    [row([0, 1, 2], "right")] * 20)
        self.assertEqual(result[0]["candidate_count"], 0)
        self.assertEqual(result[0]["candidates"], [])

    def test_only_stable_differences_are_candidates(self):
        left = [row([1, 2, 0]), row([1, 3, 0])]
        right = [row([4, 5, 0], "right"), row([4, 6, 0], "right")]
        result = stable_differences(left, right)
        self.assertEqual(result[0]["candidate_count"], 1)
        self.assertEqual(result[0]["candidates"], [{"offset": "0x0", "left": 1, "right": 4}])

    def test_mixed_builds_or_labels_are_rejected(self):
        left = [row([1]), row([1])]
        for right in ([row([2], "right", "other")] * 2,
                      [row([2], "right"), row([2], "different")], left):
            with self.assertRaises(ValueError):
                stable_differences(left, right)

    def test_too_few_samples_are_rejected(self):
        with self.assertRaises(ValueError):
            stable_differences([row([1])], [row([2], "right")] * 2)

    def test_output_is_bounded_but_count_is_complete(self):
        result = stable_differences([row([1] * 100)] * 2,
                                    [row([2] * 100, "right")] * 2, limit=3)
        self.assertEqual(result[0]["candidate_count"], 100)
        self.assertEqual(len(result[0]["candidates"]), 3)

    def test_unreadable_regions_are_not_compared(self):
        left = [row([1])] * 2
        right = [{"label": "right", "build_md5": "test",
                  "regions": {"status": {"error": "short read"}}}] * 2
        self.assertEqual(stable_differences(left, right), [])

    def test_capture_reads_only_bounded_known_regions(self):
        calls = []
        def read(address, size):
            calls.append((address, size))
            return struct.pack("<Q", 0x20000) if address == 0x500 else bytes(size)
        reader = SimpleNamespace(base_address=0, addresses={"LIVES": 0x100,
                                 "PLAYER_BASE": 0x1000, "GUI": 0x500},
                                 process=SimpleNamespace(read=read))
        regions = capture_regions(reader)
        self.assertEqual(calls, [(0, 0x400), (0x8700, 0x300), (0x500, 8), (0x20000, 0x2000)])
        self.assertEqual(regions["gui_pointee"]["size"], 0x2000)

    def test_bad_gui_pointer_is_not_dereferenced(self):
        calls = []
        def read(address, size):
            calls.append(address)
            return bytes(size)
        reader = SimpleNamespace(base_address=0, addresses={"LIVES": 0x100,
                                 "PLAYER_BASE": 0x1000, "GUI": 0x500},
                                 process=SimpleNamespace(read=read))
        self.assertIn("error", capture_regions(reader)["gui_pointee"])
        self.assertEqual(len(calls), 3)

    def test_short_regions_are_recorded_without_fabricated_bytes(self):
        reader = SimpleNamespace(base_address=0, addresses={"LIVES": 0x100, "PLAYER_BASE": 0x1000},
                                 process=SimpleNamespace(read=lambda address, size: b""))
        regions = capture_regions(reader)
        self.assertTrue(all("error" in value and "hex" not in value for value in regions.values()))
