import contextlib
import io
import unittest
from pathlib import Path
import tempfile
import json
import sys

from env.action_emulator import ActionEmulator, InputError
from env.session import GameSession
from scripts.diagnostics import DiagnosticLog, run_episodes
from tests.fakes import FakeClock, FakeKeyboard, FakeReader, sample


class DiagnosticTests(unittest.TestCase):
    def test_full_records_survive_compact_console_and_transcript_captures_prompts(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            stdout, stderr = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                with DiagnosticLog(Path(directory)/"run.jsonl") as log:
                    print("Focus the game now", file=sys.stderr)
                    log.emit({"event": "step", "elapsed_frames": 60, "enemies": [{"hp": 900}]})
                    log.emit({"event": "episode_end", "episode": 0, "reason": "spell_success",
                              "enemies": [{"hp": 1}], "extra": "full evidence"})
            records = [json.loads(line) for line in log.path.read_text().splitlines()]
            self.assertEqual(records[0]["enemies"][0]["hp"], 900)
            self.assertEqual(records[1]["extra"], "full evidence")
            self.assertNotIn("enemies", stdout.getvalue())
            self.assertNotIn('"event": "step"', stdout.getvalue())
            transcript = log.console_path.read_text()
            self.assertIn("Focus the game now", transcript)
            self.assertIn("spell_success", transcript)
            self.assertEqual(transcript.count("Diagnostic log:"), 2)
            self.assertTrue(stderr.getvalue().rstrip().endswith(str(log.console_path.resolve())))

    def test_error_restores_streams_and_reprints_actual_numbered_paths(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory)/"run.jsonl"
            path.write_text("original\n")
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                before = sys.stdout, sys.stderr
                with self.assertRaises(InputError):
                    with DiagnosticLog(path) as log:
                        raise InputError("lost focus")
                self.assertEqual((sys.stdout, sys.stderr), before)
            self.assertEqual(path.read_text(), "original\n")
            self.assertEqual(log.path.name, "run-001.jsonl")
            self.assertIn("lost focus", log.console_path.read_text())
            self.assertIn(str(log.path.resolve()), log.console_path.read_text())

    def test_existing_transcripts_are_never_overwritten(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory)/"run.jsonl"
            transcript = Path(directory)/"run.console.log"
            transcript.write_text("original transcript\n")
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with DiagnosticLog(path) as log:
                    log.emit({"event": "summary", "completed_episodes": 1})
            self.assertEqual(transcript.read_text(), "original transcript\n")
            self.assertNotEqual(log.console_path, transcript)

    def test_trace_only_events_are_not_mirrored_to_console(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with DiagnosticLog(Path(directory)/"run.jsonl") as log:
                    log.emit({"event": "transition", "large_observation": [1, 2, 3]}, console=False)
            self.assertIn("large_observation", log.path.read_text())
            self.assertNotIn("large_observation", log.console_path.read_text())

    def test_existing_logs_are_preserved_and_numbered(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            requested = Path(directory) / "check.jsonl"
            requested.write_text("original\n", encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with DiagnosticLog(requested) as log:
                    first = log.path
                    log.emit({"event": "first"})
                with DiagnosticLog(requested) as log:
                    second = log.path
                    log.emit({"event": "second"})
            self.assertEqual(requested.read_text(encoding="utf-8"), "original\n")
            self.assertEqual(first.name, "check-001.jsonl")
            self.assertEqual(second.name, "check-002.jsonl")
            self.assertIn('"first"', first.read_text(encoding="utf-8"))
            self.assertIn('"second"', second.read_text(encoding="utf-8"))

    def test_ten_failures_with_nine_automatic_retries(self):
        samples = [sample(), sample(frame=11)]
        for episode in range(10):
            samples.append(sample(frame=13, deaths=1))
            if episode < 9:
                samples.extend([sample(frame=14, deaths=1, retry=True),
                                sample(frame=0, scene=12), sample(frame=11)])
        clock = FakeClock()
        controls = ActionEmulator(FakeKeyboard())
        reader = FakeReader(*samples)
        rows = []
        with GameSession(reader, controls, spell_retry=True,
                         clock=clock, sleep=clock.sleep) as session:
            session.resetter.menu_settle = 0
            run_episodes(session, 10, rows.append)
        self.assertEqual(rows[-1]["completed_episodes"], 10)
        self.assertEqual(rows[-1]["observed_retries"], 9)
        self.assertTrue(rows[-1]["all_requested_completed"])
        self.assertEqual(rows[-1]["death_or_retry_episodes"], 10)
        self.assertEqual(rows[-1]["unresolved_episodes"], 0)
        self.assertIsNone(rows[-1]["success_rate"])
        self.assertEqual(sum(r["event"] == "episode_end" for r in rows), 10)
        self.assertFalse(controls.held)
        self.assertTrue(reader.closed)

    def test_failure_keeps_partial_summary(self):
        clock = FakeClock()
        controls = ActionEmulator(FakeKeyboard())
        reader = FakeReader(sample(), sample(frame=11), InputError("focus lost"))
        rows = []
        with GameSession(reader, controls, clock=clock, sleep=clock.sleep) as session:
            with self.assertRaises(InputError):
                run_episodes(session, 3, rows.append)
        self.assertEqual(rows[-1]["completed_episodes"], 0)
        self.assertFalse(rows[-1]["all_requested_completed"])
        self.assertFalse(controls.held)

    def test_focus_wait_sends_no_keys(self):
        keyboard = FakeKeyboard()
        keyboard.focus = False
        clock = FakeClock()
        def sleep(seconds):
            clock.sleep(seconds)
            keyboard.focus = clock() >= 0.1
        ActionEmulator(keyboard).wait_for_focus(1, clock, sleep)
        self.assertFalse(keyboard.events)
        self.assertGreaterEqual(clock(), 0.1)

    def test_focus_wait_is_bounded(self):
        keyboard = FakeKeyboard()
        keyboard.focus = False
        clock = FakeClock()
        with self.assertRaises(InputError):
            ActionEmulator(keyboard).wait_for_focus(0.1, clock, clock.sleep)
        self.assertFalse(keyboard.events)

    def test_log_reports_errors_without_suppressing_them(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            with self.assertRaises(InputError):
                with DiagnosticLog() as log:
                    log.emit({"event": "start"})
                    raise InputError("lost focus")
        self.assertIn('"event": "error"', output.getvalue())
