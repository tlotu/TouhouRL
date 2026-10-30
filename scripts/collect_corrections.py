"""Record teacher labels on states visited by a teacher/student mixture."""
import argparse
from pathlib import Path
import sys

from agent.checkpoints import read_checkpoint
from agent.corrections import CorrectionPolicy
from env.action_emulator import ActionEmulator
from env.session import GameSession, OBSERVATION_SCHEMA
from env.windows import WindowsKeyboard, connect
from scripts.diagnostics import DiagnosticLog, run_episodes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", required=True, help="Student checkpoint directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--teacher-probability", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task", required=True, help="Same manual task label as the original demonstrations")
    parser.add_argument("--pid", type=int)
    parser.add_argument("--focus-timeout", type=float, default=30)
    args = parser.parse_args()
    if (args.episodes < 1 or not 0 <= args.teacher_probability <= 1
            or not 0 <= args.seed < 2**32 or not 0 < args.focus_timeout <= 60):
        parser.error("Invalid episodes, probability, seed, or focus timeout")
    saved = read_checkpoint(args.resume, evaluation=True)
    original_task = saved.get("initialization", {}).get("identity", {}).get("task")
    if original_task and original_task != args.task:
        parser.error("Task label differs from the student's demonstrations")
    from agent.ppo_backend import PPOBackend
    from env.touhou_env import TouhouEnv

    config = saved["config"]
    with DiagnosticLog(args.output) as log:
        reader, pid, build = connect(args.pid)
        try:
            if build != saved["build_md5"] or "SPELL_RESULT_ACTIVE" not in reader.addresses:
                raise RuntimeError("Correction collection requires a matching validated game build")
            controls = ActionEmulator(WindowsKeyboard(pid))
            session = GameSession(reader, controls, capacity=config["capacity"],
                                  max_frames=config["max_frames"], spell_retry=True)
        except BaseException:
            reader.close()
            raise
        with session:
            learner = PPOBackend(TouhouEnv(session), checkpoint=str(Path(args.resume) / "model.zip"))
            choose = CorrectionPolicy(learner.deterministic_action, args.teacher_probability, args.seed)
            log.emit({"event": "connected", "pid": pid, "build_md5": build,
                      "schema": OBSERVATION_SCHEMA, "policy": "mixed_teacher_student",
                      "teacher": "reactive_attack", "collection": "corrections",
                      "run": True, "trace": True, "repeat": 2, "task": args.task,
                      "capacity": config["capacity"], "max_frames": config["max_frames"],
                      "seed": args.seed, "teacher_probability": args.teacher_probability,
                      "student_checkpoint": str(Path(args.resume).resolve())})
            print("Start from the selected spell RESULT menu. Focus the game within "
                  f"{args.focus_timeout:g} seconds and keep it focused.", file=sys.stderr)
            controls.wait_for_focus(args.focus_timeout)
            session.verify_training_boundary()

            def emit(row):
                log.emit(choose.annotate(row), console=row.get("event") != "transition")
            try:
                run_episodes(session, args.episodes, emit, policy="mixed_teacher_student",
                             trace=True, action_fn=choose)
            finally:
                log.emit(choose.summary())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped; complete correction episodes remain usable.", file=sys.stderr)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
