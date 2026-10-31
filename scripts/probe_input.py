"""Measure short horizontal input responses at one-frame observation intervals."""
import argparse
import sys

from env.action_emulator import ActionEmulator
from env.session import GameSession, OBSERVATION_SCHEMA
from env.windows import WindowsKeyboard, connect
from scripts.diagnostics import DiagnosticLog, run_episodes


# Long holds, releases and two-frame reversals; no model or teacher computation.
PHASES = ((0, 16), (3, 12), (7, 12), (3, 12), (7, 12), (0, 12),
          (3, 2), (7, 2), (3, 2), (7, 2), (3, 2), (7, 2), (0, 12))
PROBE_FRAMES = sum(frames for _, frames in PHASES)


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
    args = parser.parse_args()
    if not 0 < args.focus_timeout <= 60:
        parser.error("focus-timeout must be in (0,60]")
    with DiagnosticLog(args.output) as log:
        reader, pid, build = connect(args.pid)
        try:
            if "SPELL_RESULT_ACTIVE" not in reader.addresses:
                raise RuntimeError("Probe requires validated spell-result fields")
            controls = ActionEmulator(WindowsKeyboard(pid))
            session = GameSession(reader, controls, repeat=1, capacity=128,
                                  max_frames=PROBE_FRAMES, spell_retry=True)
        except BaseException:
            reader.close()
            raise
        with session:
            log.emit({"event": "connected", "pid": pid, "build_md5": build,
                      "schema": OBSERVATION_SCHEMA, "policy": "input_probe",
                      "collection": "input_probe", "repeat": 1, "capacity": 128,
                      "max_frames": PROBE_FRAMES, "phases": PHASES})
            print("Start from the same spell RESULT menu and focus the game. "
                  "The probe releases keys after about two seconds; gameplay continues. "
                  "Return to the menu manually afterward.", file=sys.stderr)
            controls.wait_for_focus(args.focus_timeout)
            session.verify_training_boundary()
            run_episodes(session, 1, lambda r: log.emit(r, console=r["event"] != "transition"),
                         policy="input_probe", trace=True, action_fn=ProbePolicy(session))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped; keys released.", file=sys.stderr)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
