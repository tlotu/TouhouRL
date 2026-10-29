"""Read game state, or run a bounded random-action spell-practice check."""
import argparse
from dataclasses import asdict
import json
import sys
import time

from env.action_emulator import ActionEmulator
from env.session import GameSession
from env.windows import WindowsKeyboard, connect
from scripts.diagnostics import DiagnosticLog, run_episodes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int)
    parser.add_argument("--profiles", help="Path to a build-profile JSON file")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--run", action="store_true", help="Send random movement; default is read-only")
    parser.add_argument("--spell-retry", action="store_true",
                        help="Confirm this is spell practice and the failure menu defaults to Retry")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=3600)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--focus-timeout", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=0, help="Random action seed; does not seed the game")
    parser.add_argument("--output", help="New JSONL log path; existing files are never overwritten")
    args = parser.parse_args()
    if min(args.samples, args.episodes, args.max_frames) <= 0:
        parser.error("samples, episodes, and max-frames must be positive")
    if not 1 <= args.repeat <= 16:
        parser.error("repeat must be from 1 to 16")
    if args.spell_retry and not args.run:
        parser.error("--spell-retry requires --run")
    if not 0 < args.focus_timeout <= 60:
        parser.error("focus-timeout must be in (0, 60]")
    with DiagnosticLog(args.output) as log:
        run(args, log.emit)


def run(args, emit):
    reader, pid, digest = connect(args.pid, args.profiles)
    emit({"event": "connected", "pid": pid, "build_md5": digest,
          "run": args.run, "spell_retry": args.spell_retry, "repeat": args.repeat,
          "seed": args.seed, "max_frames": args.max_frames})
    if not args.run:
        try:
            for _ in range(args.samples):
                sample = reader.snapshot()
                data = asdict(sample)
                data.pop("bullets")
                data["bullet_count"] = len(sample.bullets)
                data["radius_fallback_slots"] = [b.slot for b in sample.bullets if b.radius_estimated]
                emit({"event": "sample", **data})
                time.sleep(0.1)
        finally:
            reader.close()
        return
    try:
        controls = ActionEmulator(WindowsKeyboard(pid))
        session = GameSession(reader, controls, repeat=args.repeat,
                              max_frames=args.max_frames, spell_retry=args.spell_retry)
    except BaseException:
        reader.close()
        raise
    print(f"Focus the running spell-practice window within {args.focus_timeout:g} seconds. "
          "Do not switch windows during the test.", file=sys.stderr)
    with session:
        controls.wait_for_focus(args.focus_timeout)
        run_episodes(session, args.episodes, emit, args.seed)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped.", file=sys.stderr)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
