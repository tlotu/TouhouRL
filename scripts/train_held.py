"""Small PPO continuation with one sampled action per held decision."""
import argparse
from functools import partial
import hashlib
from pathlib import Path
import sys
import time

from agent.checkpoints import read_checkpoint, create_run_directory, save_checkpoint
from agent.held_training import (PPO_CONTRACT, HELD_REWARD_PROFILE, DEFAULT_GAMMA_FRAME,
                                 DEFAULT_LAMBDA_FRAME, collect_held_episodes)
from agent.training import train_episodic
from env.action_emulator import ActionEmulator
from env.session import GameSession
from env.windows import WindowsKeyboard, connect
from scripts.diagnostics import DiagnosticLog, timing_clock_info


def held_config(saved):
    config = dict(saved["config"])
    if config.get("decision_contract") != "attack-zone-held-v1" or config.get("features") != "hazard-zone-v1":
        raise ValueError("Requires a versioned held-zone checkpoint")
    hold = config.get("hold_frames")
    if type(hold) is not int or not 2 <= hold <= 16 or hold % 2:
        raise ValueError("Invalid checkpoint hold duration")
    expected = {"hold_frames": hold, "gamma_frame": DEFAULT_GAMMA_FRAME,
                "lambda_frame": DEFAULT_LAMBDA_FRAME, "learning_rate": 1e-5,
                "epochs": 2, "batch_size": 64, "entropy_coefficient": 0.,
                "clip_range": 0.1, "target_kl": 0.005}
    if "ppo_contract" in config:
        if config["ppo_contract"] != PPO_CONTRACT or config.get("held_ppo") != expected:
            raise ValueError("Held PPO settings or version differ")
        if saved["reward_profile"] != HELD_REWARD_PROFILE:
            raise ValueError("Held PPO reward version differs")
    elif saved["timesteps"] != 0 or saved["updates"] != 0 or saved["reward_profile"] != "spell-v2":
        raise ValueError("Only an unmodified clone can initialize held PPO")
    config.update(ppo_contract=PPO_CONTRACT, held_ppo=expected)
    config.pop("rollout_steps", None)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--decisions", type=int, default=256, help="Additional fresh decisions; finish the last episode")
    parser.add_argument("--rollout-decisions", type=int, default=256)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--pid", type=int)
    parser.add_argument("--focus-timeout", type=float, default=30)
    args = parser.parse_args()
    if (not 2 <= args.decisions <= 1000000 or not 2 <= args.rollout_decisions <= 65536
            or not 0 < args.focus_timeout <= 60):
        parser.error("Invalid decision budget or focus timeout")
    saved = read_checkpoint(args.resume)
    try:
        config = held_config(saved)
    except ValueError as exc:
        parser.error(str(exc))
    original_task = saved.get("initialization", {}).get("identity", {}).get("task")
    if not original_task or original_task != args.task:
        parser.error("Task must match the checkpoint's declared demonstrations")
    from agent.held_backend import HeldPPOBackend
    from env.touhou_env import TouhouEnv
    source = Path(args.resume)/"model.zip"
    provenance = saved.get("fine_tuning", {"source_checkpoint": str(Path(args.resume).resolve()),
                          "source_model_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                          "method": PPO_CONTRACT})
    run_dir = create_run_directory(args.run_dir)
    with DiagnosticLog(run_dir/"training.jsonl") as log:
        reader, pid, build = connect(args.pid)
        try:
            if build != saved["build_md5"] or "SPELL_RESULT_ACTIVE" not in reader.addresses:
                raise RuntimeError("Held PPO requires the matching validated game build")
            controls = ActionEmulator(WindowsKeyboard(pid))
            session = GameSession(reader, controls, repeat=2, capacity=config["capacity"],
                                  max_frames=config["max_frames"], spell_retry=True)
        except BaseException:
            reader.close()
            raise
        with session:
            learner = HeldPPOBackend(TouhouEnv(session), checkpoint=str(source))
            if learner.timesteps != saved["timesteps"]:
                raise RuntimeError("Checkpoint/model counters differ")
            if learner.model.policy.features_extractor.half_width != config["feature_options"]["half_width"]:
                raise RuntimeError("Checkpoint/model feature settings differ")
            learner.configure(config["held_ppo"])
            log.emit({"event": "training_start", "pid": pid, "build_md5": build,
                      "config": config, "task": args.task, "resume": str(source.parent),
                      "additional_decisions": args.decisions, "starting_decisions": learner.timesteps,
                      "rollout_decisions": args.rollout_decisions, "reward_profile": HELD_REWARD_PROFILE,
                      "timing_clock": timing_clock_info(), "fine_tuning": provenance})
            print("Start at the same spell RESULT menu and keep focus. "
                  "Stochastic held PPO; all optimizer updates occur in result menus.", file=sys.stderr)
            controls.wait_for_focus(args.focus_timeout)
            session.verify_training_boundary()

            def checkpoint(state):
                return save_checkpoint(learner, run_dir, {**state, "reward_profile": HELD_REWARD_PROFILE,
                    "initialization": saved["initialization"], "fine_tuning": provenance}, config, build)

            # Copy the starting state into the new run, never overwrite the source.
            initial = checkpoint({"updates": saved["updates"], "timesteps": learner.timesteps})
            log.emit({"event": "initial_checkpoint", "checkpoint": str(initial)})
            collector = partial(collect_held_episodes, hold_frames=config["hold_frames"],
                                gamma_frame=config["held_ppo"]["gamma_frame"])
            train_episodic(TouhouEnv(session), learner, session.verify_training_boundary,
                          checkpoint, log.emit, args.decisions, args.rollout_decisions,
                          saved["updates"], clock=time.perf_counter, collector=collector)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped; resume from the last complete held PPO checkpoint.", file=sys.stderr)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
