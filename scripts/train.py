"""Small CPU PPO runs with updates held at verified spell-result menus."""
import argparse
from datetime import datetime
import importlib.metadata
from pathlib import Path
import sys

from agent.checkpoints import create_run_directory, read_checkpoint, save_checkpoint
from agent.training import train_episodic
from env.action_emulator import ActionEmulator
from env.session import GameSession
from env.windows import WindowsKeyboard, connect
from scripts.diagnostics import DiagnosticLog, run_episodes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=2048,
                        help="Additional transitions; finish the final episode before stopping")
    parser.add_argument("--rollout-steps", type=int)
    parser.add_argument("--capacity", type=int)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--resume", help="Completed checkpoint directory")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate --resume without updating")
    parser.add_argument("--episodes", type=int, default=3, help="Evaluation episodes")
    parser.add_argument("--pid", type=int)
    parser.add_argument("--focus-timeout", type=float, default=30)
    args = parser.parse_args()
    if args.timesteps < 2 or not 0 < args.focus_timeout <= 60:
        parser.error("timesteps must be >=2 and focus-timeout in (0,60]")
    if args.episodes <= 0 or (args.evaluate and not args.resume):
        parser.error("episodes must be positive; --evaluate requires --resume")
    saved = read_checkpoint(args.resume) if args.resume else None
    defaults = {"rollout_steps": 512, "capacity": 128, "max_frames": 3600, "seed": 0}
    config = dict(saved["config"] if saved else defaults)
    for name in defaults:
        requested = getattr(args, name)
        if requested is not None:
            if saved and requested != config[name]:
                parser.error(f"Resume must keep the checkpoint's {name}={config[name]}")
            config[name] = requested
    if not (2 <= config["rollout_steps"] <= 65536 and 1 <= config["capacity"] <= 640
            and 60 <= config["max_frames"] <= 36000 and 0 <= config["seed"] < 2**32):
        parser.error("Invalid rollout, capacity, frame limit, or seed")
    try:
        from agent.ppo_backend import PPOBackend
        from env.touhou_env import TouhouEnv
    except ImportError as exc:
        raise RuntimeError("Install training dependencies: python -m pip install -r requirements-training.txt") from exc
    run_dir = create_run_directory(args.run_dir or f"runs/ppo-{datetime.now():%Y%m%d-%H%M%S}")
    print(f"Training directory: {run_dir}", file=sys.stderr)
    with DiagnosticLog(run_dir / "training.jsonl") as log:
        reader, pid, build = connect(args.pid)
        try:
            if "SPELL_RESULT_ACTIVE" not in reader.addresses:
                raise RuntimeError("PPO requires a build with validated spell-result detection")
            if saved and saved["build_md5"] != build:
                raise RuntimeError("Checkpoint game build differs from the running game")
            controls = ActionEmulator(WindowsKeyboard(pid))
            session = GameSession(reader, controls, capacity=config["capacity"],
                                  max_frames=config["max_frames"], spell_retry=True)
        except BaseException:
            reader.close()
            raise
        with session:
            env = TouhouEnv(session)
            learner = PPOBackend(env, seed=config["seed"], rollout_steps=config["rollout_steps"],
                                 checkpoint=str(Path(args.resume) / "model.zip") if saved else None)
            if saved and learner.timesteps != saved["timesteps"]:
                raise RuntimeError("Checkpoint metadata/model counters disagree")
            log.emit({"event": "evaluation_start" if args.evaluate else "training_start", "pid": pid, "build_md5": build,
                      "config": config, "additional_steps": args.timesteps,
                      "resume": args.resume, "starting_steps": learner.timesteps,
                      "sb3_version": importlib.metadata.version("stable-baselines3"),
                      "torch_version": importlib.metadata.version("torch")})
            print("Start from a spell RESULT menu. Focus the game within "
                  f"{args.focus_timeout:g} seconds and keep it focused throughout training.", file=sys.stderr)
            controls.wait_for_focus(args.focus_timeout)
            session.verify_training_boundary()
            if args.evaluate:
                run_episodes(session, args.episodes, log.emit, policy="ppo_deterministic",
                             action_fn=learner.deterministic_action)
                return
            def checkpoint(state):
                return save_checkpoint(learner, run_dir, state, config, build)
            if not saved:
                initial = checkpoint({"updates": 0, "timesteps": 0})
                log.emit({"event": "initial_checkpoint", "checkpoint": str(initial)})
            train_episodic(env, learner, session.verify_training_boundary, checkpoint, log.emit,
                           args.timesteps, config["rollout_steps"], saved["updates"] if saved else 0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped; resume from the last completed checkpoint.", file=sys.stderr)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
