"""Read game state, or run a bounded random-action spell-practice check."""
import argparse
from dataclasses import asdict
import json
import random
import sys
import time

from env.action_emulator import ActionEmulator
from env.session import GameSession
from env.windows import WindowsKeyboard, connect


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
    args = parser.parse_args()
    if min(args.samples, args.episodes, args.max_frames) <= 0:
        parser.error("samples, episodes, and max-frames must be positive")
    if not 1 <= args.repeat <= 16:
        parser.error("repeat must be from 1 to 16")
    if args.spell_retry and not args.run:
        parser.error("--spell-retry requires --run")
    reader, pid, digest = connect(args.pid, args.profiles)
    print(json.dumps({"pid": pid, "build_md5": digest}), flush=True)
    if not args.run:
        try:
            for _ in range(args.samples):
                sample = reader.snapshot()
                data = asdict(sample)
                data.pop("bullets")
                data["bullet_count"] = len(sample.bullets)
                print(json.dumps(data), flush=True)
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
    print("Switch to the running spell-practice window within 5 seconds.", file=sys.stderr)
    with session:
        time.sleep(5)
        rng = random.Random(0)
        for episode in range(args.episodes):
            _, info = session.reset()
            print(json.dumps({"episode": episode, "event": "reset", **info}), flush=True)
            steps = 0
            while True:
                action = rng.randrange(9)
                _, reward, terminated, truncated, info = session.step(action)
                steps += 1
                if steps % 30 == 0 or terminated or truncated:
                    print(json.dumps({"episode": episode, "action": action, "reward": reward,
                                      "terminated": terminated, "truncated": truncated, **info}), flush=True)
                if terminated or truncated:
                    break
            if info["reason"] != "death":
                print("Task ended without a failure-menu retry; manual selection is required.", file=sys.stderr)
                break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped.", file=sys.stderr)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
