"""Retry only frame races, bounded by existing deadlines and cleanup rules."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from env.action_emulator import ActionEmulator, InputError
from env.frame_stepper import FrameStepper, FrameTimeout, ACQUISITION_CONTRACT
from env.memory_reader import MemoryReadError, SnapshotRaceError, ResultMenu
from env.reset_controller import ResetController, ResetError
from env.session import GameSession
from tests.fakes import FakeClock, FakeKeyboard, FakeReader, sample
from tests.test_ppo_backend import AVAILABLE


class SnapshotRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.keyboard = FakeKeyboard()
        self.controls = ActionEmulator(self.keyboard)

    def stepper(self, reader):
        return FrameStepper(reader, self.controls, clock=self.clock, sleep=self.clock.sleep)

    def session(self, reader):
        return GameSession(reader, self.controls, clock=self.clock, sleep=self.clock.sleep)

    def test_step_recovers_without_fabricating_frames_or_resending_input(self):
        reader = FakeReader(SnapshotRaceError("race"), SnapshotRaceError("race"), sample(frame=14))
        self.controls.apply(3)
        stepper = self.stepper(reader)
        state, frames, _ = stepper.advance(sample(frame=10))
        self.assertEqual((state.frame, frames), (14, 4))
        self.assertEqual(stepper.last_timing["snapshot_calls"], 3)
        self.assertEqual(stepper.last_timing["snapshot_race_retries"], 2)
        self.assertEqual(self.keyboard.events.count(("down", "right")), 1)

    def test_continuous_or_slow_races_reach_existing_deadline(self):
        reader = FakeReader(SnapshotRaceError("race"))
        with self.assertRaisesRegex(FrameTimeout, "frame-race"):
            self.stepper(reader).advance(sample())
        self.assertGreaterEqual(self.clock(), .5)
        self.assertLess(self.clock(), .502)
        self.clock = FakeClock()
        def slow():
            self.clock.sleep(.6)
            raise SnapshotRaceError("slow race")
        reader.snapshot = slow
        with self.assertRaises(FrameTimeout):
            self.stepper(reader).advance(sample())
        self.assertEqual(self.clock(), .6)
        self.clock = FakeClock()
        reader = FakeReader(SnapshotRaceError("race"), sample())
        with self.assertRaisesRegex(FrameTimeout, "frames stalled"):
            self.stepper(reader).advance(sample())

    def test_other_read_errors_are_not_retried(self):
        reader = FakeReader(MemoryReadError("short read"), sample(frame=12))
        with self.assertRaisesRegex(MemoryReadError, "short read"):
            self.stepper(reader).advance(sample())
        self.assertEqual(self.clock(), 0)

    def test_focus_loss_is_checked_again_after_a_race(self):
        reader = FakeReader(sample())
        def lose_focus():
            self.keyboard.focus = False
            raise SnapshotRaceError("race")
        reader.snapshot = lose_focus
        self.controls.apply(3)
        with self.assertRaises(InputError):
            self.stepper(reader).advance(sample())
        self.assertFalse(self.controls.held)

    def test_reset_recovers_or_times_out_without_stale_state(self):
        reader = FakeReader(SnapshotRaceError("race"), sample(), sample(frame=11))
        reset = ResetController(reader, self.controls, timeout=.2,
                                clock=self.clock, sleep=self.clock.sleep)
        self.assertEqual(reset.reset(first=True).frame, 11)
        reset.reader = FakeReader(SnapshotRaceError("race"))
        with self.assertRaisesRegex(ResetError, "no sample"):
            reset.reset(first=True)
        self.assertFalse(self.controls.held)

    def test_update_boundary_recovers_but_cannot_bypass_timeout(self):
        menu = sample(result_menu=ResultMenu(True, 0, 1, 1))
        reader = FakeReader(SnapshotRaceError("race"), menu)
        session = self.session(reader)
        self.controls.apply(3)
        self.assertEqual(session.verify_training_boundary(), "success")
        self.assertFalse(self.controls.held)
        reader.snapshot = FakeReader(SnapshotRaceError("race")).snapshot
        with self.assertRaises(FrameTimeout):
            session.verify_training_boundary()

    def test_session_records_recovery_and_measured_frame_overshoot(self):
        reader = FakeReader(sample(), sample(frame=11), SnapshotRaceError("race"), sample(frame=14))
        with self.session(reader) as session:
            session.reset()
            _, _, done, truncated, info = session.step(3)
            self.assertFalse(done or truncated)
            self.assertEqual((info["delta_frames"], info["elapsed_frames"]), (3, 3))
            self.assertEqual(info["timing"]["snapshot_race_retries"], 1)
        self.assertFalse(self.controls.held)

    def test_failed_step_releases_keys_and_retains_only_last_verified_state(self):
        for error in (SnapshotRaceError("race"), MemoryReadError("short read")):
            with self.subTest(error=error):
                self.setUp()
                reader = FakeReader(sample(), sample(frame=11), error)
                with self.session(reader) as session:
                    session.reset()
                    with self.assertRaises(FrameTimeout if isinstance(error, SnapshotRaceError) else MemoryReadError):
                        session.step(3)
                    self.assertFalse(self.controls.held)
                    self.assertFalse(session.active)
                    self.assertEqual(session.sample.frame, 11)
                    self.assertEqual(session.elapsed_frames, 0)

    def test_result_settlement_can_retry_without_inventing_an_outcome(self):
        pending = sample(frame=13, result_menu=ResultMenu(True, 0, 0, 1))
        settled = sample(frame=14, result_menu=ResultMenu(True, 0, 1, 1))
        reader = FakeReader(sample(), sample(frame=11), pending, SnapshotRaceError("race"), settled)
        with self.session(reader) as session:
            session.reset()
            _, _, done, truncated, info = session.step(3)
            self.assertTrue(done)
            self.assertFalse(truncated)
            self.assertEqual(info["reason"], "spell_success")
            self.assertFalse(self.controls.held)

    @unittest.skipUnless(AVAILABLE, "Install training dependencies")
    def test_eval_preserves_the_pending_decision_when_acquisition_aborts(self):
        from scripts.evaluate_control import main
        from tests.test_boundary_control import observation, scores
        from tests.test_control_evaluation import metadata
        from tests.test_training import FakeTask
        class Task(FakeTask):
            capacity = 4
            elapsed_frames = 0
            def reset(self):
                _, info = super().reset()
                return observation(376), info
            def step(self, action):
                raise MemoryReadError("acquisition failed")
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def verify_training_boundary(self):
                self.boundary()
        class Backend:
            def __init__(self, *args, **kwargs):
                pass
            def action_logits(self, obs):
                return scores(3, 7)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            source = Path(directory) / "source"
            source.mkdir()
            (source / "model.zip").write_bytes(b"frozen")
            output = Path(directory) / "interrupted.jsonl"
            args = ["evaluate_control", "--resume", str(source), "--task", "source",
                    "--boundary-mask", "--output", str(output)]
            reader = SimpleNamespace(addresses={"SPELL_RESULT_ACTIVE": 1}, close=lambda: None)
            with patch("sys.argv", args), patch("scripts.evaluate_control.read_checkpoint", return_value=metadata()), \
                    patch("scripts.evaluate_control.connect", return_value=(reader, 1, "test-build")), \
                    patch("scripts.evaluate_control.WindowsKeyboard", return_value=FakeKeyboard()), \
                    patch("scripts.evaluate_control.GameSession", return_value=Task()), \
                    patch("agent.ppo_backend.PPOBackend", Backend), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(MemoryReadError):
                main()
            rows = [json.loads(s) for s in output.read_text().splitlines()]
            self.assertEqual(rows[0]["acquisition_contract"], ACQUISITION_CONTRACT)
            pending = next(r for r in rows if r["event"] == "interrupted_decision")
            self.assertEqual((pending["episode"], pending["step"]), (0, 0))
            self.assertEqual(pending["pending_decision"]["raw_student_action"], 3)
            self.assertEqual(pending["pending_decision"]["committed_action"], 7)
            self.assertFalse(any(r["event"] == "transition" for r in rows))
            summary = next(r for r in rows if r["event"] == "summary")
            self.assertEqual(summary["completed_episodes"], 0)
            self.assertEqual(summary["incomplete_episodes"], 1)
