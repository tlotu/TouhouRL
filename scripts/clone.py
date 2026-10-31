"""Fit a fresh PPO actor from complete teacher traces without connecting to a game."""
import argparse
from pathlib import Path
import sys

from agent.checkpoints import create_run_directory, save_checkpoint
from agent.demonstrations import (load_demonstrations, split_episodes, stack_episodes,
                                  deduplicate_episodes, imitation_weights)
from scripts.diagnostics import DiagnosticLog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demos", nargs="+", required=True, help="Explicit teacher JSONL paths")
    parser.add_argument("--run-dir", default="runs/behavior-cloning")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-failures", action="store_true")
    parser.add_argument("--features", choices=("flat", "hazard-v1"), default="flat")
    parser.add_argument("--weighting", choices=("uniform", "correction-v1"), default="uniform")
    args = parser.parse_args()
    if (args.epochs < 1 or args.batch_size < 1 or not 0 < args.learning_rate <= 0.01
            or not 0 <= args.seed < 2**32):
        parser.error("Invalid epochs, batch size, learning rate, or seed")
    episodes, provenance = load_demonstrations(args.demos, args.include_failures)
    if args.weighting == "correction-v1":
        episodes, duplicates = deduplicate_episodes(episodes)
        provenance["duplicate_trajectories"] = duplicates
    train, validation = split_episodes(episodes, args.seed)
    identity = provenance["identity"]
    from agent.cloning import fit_actor
    from agent.ppo_backend import PPOBackend
    from env.touhou_env import state_space
    import gymnasium as gym
    import numpy as np

    class OfflineEnv(gym.Env):
        observation_space = state_space(identity["capacity"])
        action_space = gym.spaces.Discrete(9)

        def reset(self, **kwargs):
            raise RuntimeError("Cloning must not reset a game")

        def step(self, action):
            raise RuntimeError("Cloning must not step a game")

    train_data, validation_data = stack_episodes(train), stack_episodes(validation)
    train_weights = imitation_weights(train, args.weighting)
    validation_weights = imitation_weights(validation, args.weighting)
    run_dir = create_run_directory(args.run_dir)
    with DiagnosticLog(run_dir / "cloning.jsonl") as log:
        learner = PPOBackend(OfflineEnv(), seed=args.seed, features=args.features)
        config = {"rollout_steps": 512, "capacity": identity["capacity"],
                  "max_frames": identity["max_frames"], "seed": args.seed, "features": args.features}
        provenance.update({"method": "behavior_cloning", "epochs": args.epochs, "features": args.features,
                           "weighting": args.weighting, "selection_metric": "weighted_validation_loss",
                           "train_priority_samples": int((train_weights > 1).sum()),
                           "validation_priority_samples": int((validation_weights > 1).sum()),
                           "batch_size": args.batch_size, "learning_rate": args.learning_rate,
                           "seed": args.seed, "train_episode_ids": [ep["id"] for ep in train],
                           "validation_episode_ids": [ep["id"] for ep in validation],
                           "train_action_counts": np.bincount(train_data[1], minlength=9).tolist(),
                           "validation_action_counts": np.bincount(validation_data[1], minlength=9).tolist()})
        log.emit({"event": "cloning_start", **provenance})
        result = fit_actor(learner, train_data, validation_data, log.emit,
                           args.epochs, args.batch_size, args.learning_rate, args.seed,
                           train_weights=train_weights, validation_weights=validation_weights)
        # Demonstration frames and supervised epochs are not PPO interaction steps.
        provenance["result"] = result
        saved = save_checkpoint(learner, run_dir, {"updates": 0, "timesteps": 0,
                                "initialization": provenance}, config, identity["build_md5"])
        log.emit({"event": "cloning_complete", **result, "checkpoint": str(saved)})


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Cloning interrupted; no new checkpoint published.", file=sys.stderr)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
