"""Evaluate an unchanged network with an explicit minimum action duration."""
import argparse
from pathlib import Path
import sys

from agent.action_commitment import ActionCommitment
from agent.checkpoints import read_checkpoint
from env.action_emulator import ActionEmulator
from env.session import GameSession, OBSERVATION_SCHEMA
from env.windows import WindowsKeyboard, connect
from scripts.diagnostics import DiagnosticLog, run_episodes, timing_clock_info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hold-frames", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--pid", type=int)
    parser.add_argument("--focus-timeout", type=float, default=30)
    args = parser.parse_args()
    if (args.episodes < 1 or not 0 < args.focus_timeout <= 60
            or not 2 <= args.hold_frames <= 16 or args.hold_frames % 2):
        parser.error("Invalid episode count, focus timeout or even hold duration (2-16)")
    saved = read_checkpoint(args.resume, evaluation=True)
    original_task = saved.get("initialization", {}).get("identity", {}).get("task")
    if original_task and original_task != args.task:
        parser.error("Task label differs from the checkpoint's demonstrations")
    from agent.ppo_backend import PPOBackend
    from env.touhou_env import TouhouEnv

    config = saved["config"]
    with DiagnosticLog(args.output) as log:
        reader, pid, build = connect(args.pid)
        try:
            if build != saved["build_md5"] or "SPELL_RESULT_ACTIVE" not in reader.addresses:
                raise RuntimeError("Control evaluation requires the validated checkpoint game build")
            controls = ActionEmulator(WindowsKeyboard(pid))
            session = GameSession(reader, controls, repeat=2, capacity=config["capacity"],
                                  max_frames=config["max_frames"], spell_retry=True)
        except BaseException:
            reader.close()
            raise
        with session:
            learner = PPOBackend(TouhouEnv(session), checkpoint=str(Path(args.resume) / "model.zip"))
            choose = ActionCommitment(learner.deterministic_action,
                                      lambda: session.elapsed_frames, args.hold_frames)
            log.emit({"event": "connected", "pid": pid, "build_md5": build,
                      "schema": OBSERVATION_SCHEMA, "timing_clock": timing_clock_info(),
                      "policy": "ppo_action_commitment", "collection": "control_evaluation",
                      "run": True, "trace": True, "repeat": 2, "task": args.task,
                      "capacity": config["capacity"], "max_frames": config["max_frames"],
                      "hold_frames": args.hold_frames,
                      "student_checkpoint": str(Path(args.resume).resolve())})
            print("Start from the same spell RESULT menu and keep game focus. "
                  "This evaluates a network plus action holding, without training or a teacher.",
                  file=sys.stderr)
            controls.wait_for_focus(args.focus_timeout)
            session.verify_training_boundary()

            def emit(row):
                log.emit(choose.annotate(row), console=row["event"] != "transition")
            try:
                run_episodes(session, args.episodes, emit, trace=True,
                             policy="ppo_action_commitment", action_fn=choose)
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
