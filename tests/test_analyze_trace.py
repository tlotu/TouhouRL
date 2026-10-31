"""Offline motion diagnosis must not mix episodes or count result-menu jumps."""
import unittest

from scripts.analyze_trace import analyze


def transition(episode, step, action, x, next_x, terminated=False):
    def obs(px):
        return {"player": [px/192-1, 0.5, 0, 0], "bullets": [[0]*7], "enemies": []}
    return {"event": "transition", "episode": episode, "step": step,
            "action": action, "teacher_action": 7, "terminated": terminated,
            "truncated": False, "observation": obs(x), "next_observation": obs(next_x),
            "info": {"delta_frames": 2, "elapsed_frames": (step+1)*2, "dying": False}}


class AnalyzeTraceTests(unittest.TestCase):
    def test_delayed_motion_and_terminal_exclusion(self):
        rows = [transition(0, i, 3 if i%2 == 0 else 7, 192, 188 if i%2 == 0 else 196)
                for i in range(8)]
        rows.append(transition(0, 8, 3, 192, 0, terminated=True))
        result = analyze(rows)["episodes"][0]
        fits = result["motion_fits"]
        self.assertEqual([r["samples"] for r in fits], [6, 6, 6])
        self.assertLess(fits[1]["mean_squared_displacement_error"], 1e-10)
        self.assertGreater(fits[0]["mean_squared_displacement_error"], 60)
        self.assertIsNone(result["tail"][-1]["displacement"])

    def test_episode_boundaries_and_missing_steps(self):
        rows = [transition(e, i, 3, 192, 196) for e in range(2) for i in (0, 1)]
        rows.append(transition(1, 5, 3, 192, 196))
        reports = analyze(rows)["episodes"]
        self.assertEqual(len(reports), 2)
        self.assertTrue(all(r["motion_fits"][0]["samples"] == 0 for r in reports))
        with self.assertRaises(ValueError):
            analyze([])

    def test_one_frame_probe_and_timing_summary(self):
        rows = [transition(0, i, 3, 192, 194) for i in range(5)]
        for row in rows:
            row["info"]["delta_frames"] = 1
            row["info"]["timing"] = {"input_send_seconds": 0.002, "previous_action": 3}
            row["info"]["decision_seconds"] = 0.004
            row["student_inference_seconds"] = 0.003
        report = analyze(rows)["episodes"][0]
        self.assertLess(report["motion_fits"][0]["mean_squared_displacement_error"], 1e-10)
        self.assertEqual(report["timing_summary"]["input_send_seconds"]["samples"], 5)
        self.assertEqual(report["timing_summary"]["student_inference_seconds"]["median"], 0.003)
        self.assertNotIn("previous_action", report["timing_summary"])
