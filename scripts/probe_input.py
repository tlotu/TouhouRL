"""Measure input responses, optionally comparing slow/fast movement modes."""
import argparse
import sys

from env.action_emulator import ActionEmulator
from env.session import GameSession, OBSERVATION_SCHEMA
from env.windows import WindowsKeyboard, connect
from env.movement_probe import MovementProbe, annotate_movement
from scripts.diagnostics import DiagnosticLog, run_episodes, timing_clock_info


# Long holds, releases and two-frame reversals; no model or teacher computation.
PHASES = ((0, 16), (3, 12), (7, 12), (3, 12), (7, 12), (0, 12),
          (3, 2), (7, 2), (3, 2), (7, 2), (3, 2), (7, 2), (0, 12))
PROBE_FRAMES = sum(frames for _, frames in PHASES)

# Mode changes are explicit experimental inputs, not a controller repair.
MOVEMENT_PHASES = ((0, 4, True), (3, 16, True), (7, 16, True),
                   (3, 16, False), (7, 16, False), (4, 8, True), (8, 8, True),
                   (4, 8, False), (8, 8, False), (0, 8, True))


class MovementModePolicy:
    def __init__(self, session):
        self.session = session

    def __call__(self, obs):
        end = 0
        for action, frames, slow in MOVEMENT_PHASES:
            end += frames
            if self.session.elapsed_frames < end:
                self.session.controls.slow = slow
                return action
        self.session.controls.slow = True
        return 0


class ProbePolicy:
    def __init__(self, session):
        self.session = session

    def __call__(self, obs):
        end = 0
        for action, frames in PHASES:
            end += frames
            if self.session.elapsed_frames < end:
                return action
        return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pid", type=int)
    parser.add_argument("--focus-timeout", type=float, default=30)
    parser.add_argument("--movement-modes", action="store_true",
                        help="Compare requested slow/fast cardinal/diagonal movement with candidate telemetry")
    args = parser.parse_args()
    if not 0 < args.focus_timeout <= 60:
        parser.error("focus-timeout must be in (0,60]")
    budget = sum(p[1] for p in MOVEMENT_PHASES) if args.movement_modes else PROBE_FRAMES
    with DiagnosticLog(args.output) as log:
        reader, pid, build = connect(args.pid)
        try:
            if "SPELL_RESULT_ACTIVE" not in reader.addresses:
                raise RuntimeError("Probe requires validated spell-result fields")
            controls = ActionEmulator(WindowsKeyboard(pid))
            movement = MovementProbe(reader, build) if args.movement_modes else None
            session = GameSession(reader, controls, repeat=1, capacity=128,
                                  max_frames=budget, spell_retry=True)
        except BaseException:
            reader.close()
            raise
        with session:
            log.emit({"event": "connected", "pid": pid, "build_md5": build,
                      "timing_clock": timing_clock_info(),
                      "schema": OBSERVATION_SCHEMA, "policy": "input_probe",
                      "collection": "input_probe", "repeat": 1, "capacity": 128,
                      "max_frames": budget, "phases": MOVEMENT_PHASES if args.movement_modes else PHASES,
                      "movement_modes": args.movement_modes,
                      "movement_candidates_verified": False})
            print("Start from the same spell RESULT menu and focus the game. "
                  "The probe releases keys after about two seconds; gameplay continues. "
                  "Return to the menu manually afterward.", file=sys.stderr)
            controls.wait_for_focus(args.focus_timeout)
            session.verify_training_boundary()
            def emit(row):
                if movement is not None:
                    row = annotate_movement(row, movement, controls)
                log.emit(row, console=row["event"] != "transition")
            run_episodes(session, 1, emit, policy="input_probe", trace=True,
                         action_fn=MovementModePolicy(session) if args.movement_modes else ProbePolicy(session))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped; keys released.", file=sys.stderr)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
