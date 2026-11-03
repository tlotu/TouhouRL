"""Evaluate a rule-only attack-band experiment; no model or HP feedback."""
import argparse
from dataclasses import asdict
import sys

from agent.action_commitment import ActionCommitment
from agent.attack_zone import AttackZoneConfig, AttackZonePolicy
from env.action_emulator import ActionEmulator
from env.session import GameSession, OBSERVATION_SCHEMA
from env.windows import WindowsKeyboard, connect
from env.movement_probe import MovementProbe, annotate_movement
from env.frame_stepper import ACQUISITION_CONTRACT
from scripts.diagnostics import DiagnosticLog, run_episodes, timing_clock_info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--half-width", type=float, default=24)
    parser.add_argument("--min-clearance", type=float, default=4)
    parser.add_argument("--max-clearance-loss", type=float, default=1)
    parser.add_argument("--hold-frames", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pid", type=int)
    parser.add_argument("--focus-timeout", type=float, default=30)
    parser.add_argument("--movement-probe", action="store_true",
                        help="Log OS Shift and baseline-only candidate movement fields")
    args = parser.parse_args()
    try:
        config = AttackZoneConfig(args.half_width, args.min_clearance, args.max_clearance_loss)
        if (args.episodes < 1 or not 0 < args.focus_timeout <= 60 or
                not 2 <= args.hold_frames <= 16 or args.hold_frames % 2):
            raise ValueError("Invalid episodes, focus timeout or even hold duration (2-16)")
    except ValueError as exc:
        parser.error(str(exc))
    with DiagnosticLog(args.output) as log:
        reader, pid, build = connect(args.pid)
        try:
            if "SPELL_RESULT_ACTIVE" not in reader.addresses:
                raise RuntimeError("Attack-zone evaluation requires validated spell-result fields")
            controls = ActionEmulator(WindowsKeyboard(pid))
            movement = MovementProbe(reader, build) if args.movement_probe else None
            session = GameSession(reader, controls, repeat=2, capacity=128,
                                  max_frames=3600, spell_retry=True)
        except BaseException:
            reader.close()
            raise
        with session:
            rule = AttackZonePolicy(config)
            choose = ActionCommitment(rule, lambda: session.elapsed_frames, args.hold_frames)
            log.emit({"event": "connected", "pid": pid, "build_md5": build,
                      "schema": OBSERVATION_SCHEMA, "timing_clock": timing_clock_info(),
                      "acquisition_contract": ACQUISITION_CONTRACT,
                      "policy": "attack_zone", "collection": "attack_zone_evaluation",
                      "run": True, "trace": True, "repeat": 2, "task": args.task,
                      "capacity": 128, "max_frames": 3600, "hold_frames": args.hold_frames,
                      "attack_zone": asdict(config), "prediction_frames": 12,
                      "aim_projection_frames": 4})
            if movement is not None:
                log.emit({"event": "movement_probe_enabled", "verified": False,
                          "source": "reference tools/offsets.py and verify_focus.py"})
            print("Start from the same spell RESULT menu and keep focus. "
                  "Rule-only experiment: width is uncalibrated; no model, training or HP feedback.",
                  file=sys.stderr)
            controls.wait_for_focus(args.focus_timeout)
            session.verify_training_boundary()

            def emit(row):
                row = choose.annotate(row)
                if row["event"] == "transition":
                    row["raw_rule_action"] = row.pop("raw_student_action")
                    row["rule_decision_seconds"] = row.pop("student_inference_seconds")
                    row["rule_ranking"] = rule.last
                    executed = next((c for c in rule.last["candidates"]
                                     if c["action"] == row["action"]), None)
                    row["executed_candidate"] = executed
                    row["executed_meets_clearance_gate"] = bool(executed and executed["eligible"])
                    if movement is not None:
                        row = annotate_movement(row, movement, controls)
                log.emit(row, console=row["event"] != "transition")
            try:
                run_episodes(session, args.episodes, emit, policy="attack_zone", trace=True,
                             action_fn=choose)
            finally:
                log.emit(choose.summary())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped; keys released.", file=sys.stderr)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
