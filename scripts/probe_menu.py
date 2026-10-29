"""Capture manually labeled result screens or compare their stable bytes."""
import argparse
import json
from pathlib import Path
import sys
import time

from env.menu_probe import capture_regions, result_candidates, stable_differences
from env.windows import connect
from scripts.diagnostics import DiagnosticLog


def load_captures(path):
    return [row for line in Path(path).read_text(encoding="utf-8").splitlines()
            if (row := json.loads(line)).get("event") == "menu_capture"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    capture = modes.add_parser("capture")
    capture.add_argument("--label", required=True,
                         choices=("combat", "success_default", "success_retry", "death_default", "timeout_default"))
    capture.add_argument("--output", required=True)
    capture.add_argument("--samples", type=int, default=20)
    capture.add_argument("--delay", type=float, default=5)
    capture.add_argument("--pid", type=int)
    compare = modes.add_parser("compare")
    compare.add_argument("left")
    compare.add_argument("right")
    args = parser.parse_args()
    if args.mode == "compare":
        result = stable_differences(load_captures(args.left), load_captures(args.right))
        print(json.dumps({"event": "menu_candidates", "verified": False, "regions": result}, indent=2))
        return
    if not 2 <= args.samples <= 200 or not 0 <= args.delay <= 60:
        parser.error("samples must be 2-200 and delay must be 0-60 seconds")
    with DiagnosticLog(args.output) as log:
        reader, pid, digest = connect(args.pid)
        try:
            print(f"Show the labeled screen/selection within {args.delay:g} seconds. "
                  "Keep it unchanged during capture. No keys will be sent.", file=sys.stderr)
            time.sleep(args.delay)
            for index in range(args.samples):
                # A matching byte counter reduces, but cannot eliminate,
                # concurrent-update ambiguity in external memory samples.
                for _ in range(3):
                    before = reader.frame()
                    regions = capture_regions(reader)
                    after = reader.frame()
                    if before == after:
                        break
                else:
                    raise RuntimeError("Menu regions crossed frames on every attempt")
                log.emit({"event": "menu_capture", "label": args.label,
                          "label_source": "manual", "build_md5": digest, "pid": pid,
                          "sample": index, "frame": after, "regions": regions,
                          "result_candidates": result_candidates(regions)}, console=False)
                time.sleep(0.05)
            print(f"Captured {args.samples} labeled samples: {log.path}")
            print(json.dumps({"event": "unverified_result_candidates", **result_candidates(regions)}))
        finally:
            reader.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped.", file=sys.stderr)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
