"""Measure input responses, optionally comparing slow/fast movement modes."""
import argparse
import ctypes
import sys
import time

from env.action_emulator import ActionEmulator, InputError
from env.session import GameSession, OBSERVATION_SCHEMA
from env.windows import Input, WindowsKeyboard, connect
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

SWITCH_PHASES = tuple((action, 6) for action in (
    0, 3, 0, 7, 0, 2, 0, 6, 0, 4, 0, 8, 3, 7, 3, 7, 2, 8, 4, 6, 2, 4, 6, 8, 0))


class SwitchTimingKeyboard(WindowsKeyboard):
    """Time each existing key operation without reading memory between keys."""

    def __init__(self, pid):
        super().__init__(pid)
        self.events = []

    def _send(self, key, up):
        started = time.perf_counter()
        super()._send(key, up)
        self.events.append({'key': key, 'down': not up, 'started_at': started,
                            'returned_at': time.perf_counter()})

    def send_batch(self, changes):
        if not changes:
            return 0
        if not self.is_foreground():
            raise InputError('Game window is not in the foreground')
        events = (Input * len(changes))(*(self._input_event(key, not down) for key, down in changes))
        started = time.perf_counter()
        sent = self.user.SendInput(len(changes), events, ctypes.sizeof(Input))
        returned = time.perf_counter()
        self.events.extend({'key': key, 'down': down, 'started_at': started, 'returned_at': returned}
                           for key, down in changes)
        return sent


class SwitchTimingControls(ActionEmulator):
    def __init__(self, keyboard, read_frame, batch=False):
        super().__init__(keyboard)
        self.read_frame = read_frame
        self.batch = batch
        self.last_switch = None

    def apply(self, action):
        self.keyboard.events.clear()
        before = self.read_frame()
        if self.batch:
            self._apply_batch(action)
        else:
            super().apply(action)
        after = self.read_frame()
        self.last_switch = {'frame_before_send': before, 'frame_after_send': after,
                            'events': list(self.keyboard.events)}

    def _apply_batch(self, action):
        wanted = self._wanted_keys(action)
        self.check_focus()
        try:
            self._send_batch(wanted, sorted(self.held - wanted), sorted(wanted - self.held))
        except BaseException:
            self.release_all()
            raise


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
    def __init__(self, session, phases=PHASES):
        self.session, self.phases = session, phases

    def __call__(self, obs):
        end = 0
        for action, frames in self.phases:
            end += frames
            if self.session.elapsed_frames < end:
                return action
        return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pid", type=int)
    parser.add_argument("--focus-timeout", type=float, default=30)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--movement-modes", action="store_true",
                       help="Compare requested slow/fast cardinal/diagonal movement with candidate telemetry")
    modes.add_argument("--switch-timing", action="store_true",
                       help="Sample single-frame movement and bracket direction-key sends")
    modes.add_argument("--switch-batch", action="store_true",
                       help="Repeat the switch timing probe with one SendInput call per group")
    args = parser.parse_args()
    if not 0 < args.focus_timeout <= 60:
        parser.error("focus-timeout must be in (0,60]")
    switch_probe = args.switch_timing or args.switch_batch
    phases = MOVEMENT_PHASES if args.movement_modes else SWITCH_PHASES if switch_probe else PHASES
    budget = sum(p[1] for p in phases)
    with DiagnosticLog(args.output) as log:
        reader, pid, build = connect(args.pid)
        try:
            if "SPELL_RESULT_ACTIVE" not in reader.addresses:
                raise RuntimeError("Probe requires validated spell-result fields")
            movement = MovementProbe(reader, build) if args.movement_modes or switch_probe else None
            controls = (SwitchTimingControls(SwitchTimingKeyboard(pid), reader.frame, batch=args.switch_batch)
                        if switch_probe else ActionEmulator(WindowsKeyboard(pid)))
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
                      "max_frames": budget, "phases": phases,
                      "movement_modes": args.movement_modes,
                      "switch_timing": switch_probe,
                      "switch_send_mode": "batch" if args.switch_batch else "sequential" if switch_probe else None,
                      "movement_candidates_verified": False})
            print("Start from the same spell RESULT menu and focus the game. "
                  f"The probe releases keys after about {budget / 60:.1f} seconds; gameplay continues. "
                  "Return to the menu manually afterward.", file=sys.stderr)
            controls.wait_for_focus(args.focus_timeout)
            session.verify_training_boundary()
            def emit(row):
                if movement is not None:
                    row = annotate_movement(row, movement, controls)
                if switch_probe and row['event'] == 'transition':
                    row['switch_timing'] = controls.last_switch
                log.emit(row, console=row["event"] != "transition")
            run_episodes(session, 1, emit, policy="input_probe", trace=True,
                         action_fn=MovementModePolicy(session) if args.movement_modes else ProbePolicy(session, phases))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped; keys released.", file=sys.stderr)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
