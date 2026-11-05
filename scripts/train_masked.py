"""Prepare or train a masked held policy on one explicitly selected seen task."""
import argparse
from functools import partial
import hashlib
from pathlib import Path
import sys
import time

from agent.checkpoints import read_checkpoint, create_run_directory, save_checkpoint
from agent.masked_contract import MASKED_PPO_CONTRACT, MASKED_REWARD_PROFILE
from agent.risk_contract import RISK_PPO_CONTRACT, RISK_OBSERVATION_CONTRACT
from scripts.train_held import held_config
from scripts.diagnostics import DiagnosticLog, timing_clock_info


def masked_config(saved):
    config = dict(saved["config"])
    contract = config.get("ppo_contract")
    if contract is not None and contract != MASKED_PPO_CONTRACT:
        raise ValueError("Requires an untouched clone or matching masked PPO checkpoint")
    if contract is None and (saved["timesteps"] or saved["updates"] or saved["reward_profile"] != "spell-v2"):
        raise ValueError("Only untouched clones can initialize masked PPO")
    base = {**saved, "config": dict(config), "timesteps": 0, "updates": 0, "reward_profile": "spell-v2"}
    for key in ("ppo_contract", "held_ppo"):
        base["config"].pop(key, None)
    expected = held_config(base)
    expected["ppo_contract"] = MASKED_PPO_CONTRACT
    if contract is not None and (config != expected or saved["reward_profile"] != MASKED_REWARD_PROFILE):
        raise ValueError("Masked PPO settings or reward version differ")
    return expected


def risk_config(saved):
    from agent.risk_control import RISK_CONTRACT, RISK_OPTIONS
    source = {**saved, "config": dict(saved["config"])}
    resumed = source["config"].get("ppo_contract") == RISK_PPO_CONTRACT
    if resumed:
        source["config"].pop("risk_filter", None)
        source["config"]["ppo_contract"] = MASKED_PPO_CONTRACT
    elif source["config"].get("ppo_contract") != MASKED_PPO_CONTRACT:
        raise ValueError("Risk PPO requires a physical-mask PPO source or matching risk resume")
    expected = masked_config(source)
    expected["ppo_contract"] = RISK_PPO_CONTRACT
    expected["risk_filter"] = {"contract": RISK_CONTRACT,
                               "observation_contract": RISK_OBSERVATION_CONTRACT, **RISK_OPTIONS}
    if resumed and expected != saved["config"]:
        raise ValueError("Risk PPO settings differ")
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--risk-filter", action="store_true", help="Explicit risk-consistent PPO migration/resume")
    parser.add_argument("--decisions", type=int, default=256)
    parser.add_argument("--rollout-decisions", type=int, default=256)
    parser.add_argument("--pid", type=int)
    parser.add_argument("--focus-timeout", type=float, default=30)
    args = parser.parse_args()
    if not 2 <= args.decisions <= 1000000 or not 2 <= args.rollout_decisions <= 65536 or not 0 < args.focus_timeout <= 60:
        parser.error("Invalid budget or focus timeout")
    saved = read_checkpoint(args.resume)
    try:
        config = risk_config(saved) if args.risk_filter else masked_config(saved)
    except ValueError as exc:
        parser.error(str(exc))
    identity = saved.get("initialization", {}).get("identity", {})
    tasks = identity.get("tasks") or ([identity["task"]] if identity.get("task") else [])
    if args.task not in tasks:
        parser.error("This initial masked trainer requires a declared seen task")
    from agent.masked_backend import MaskedPPOBackend
    from agent.masked_policy import BoundaryActorCriticPolicy
    from agent.risk_policy import RiskActorCriticPolicy
    from agent.masked_training import collect_masked_episodes
    from agent.training import train_episodic
    from env.touhou_env import TouhouEnv, state_space
    from env.action_emulator import ActionEmulator
    from env.session import GameSession
    from env.windows import connect, WindowsKeyboard
    from env.frame_stepper import ACQUISITION_CONTRACT
    import gymnasium as gym

    class Offline(gym.Env):
        observation_space = state_space(config["capacity"])
        action_space = gym.spaces.Discrete(9)
        def reset(self, **kwargs):
            raise RuntimeError("Offline preparation must not reset a game")
        def step(self, action):
            raise RuntimeError("Offline preparation must not send input")

    source = Path(args.resume)/"model.zip"
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    backend = MaskedPPOBackend
    if args.risk_filter:
        from agent.risk_backend import RiskPPOBackend
        backend = RiskPPOBackend
    learner = backend(Offline(), checkpoint=str(source))
    actual_contract = (RISK_PPO_CONTRACT if isinstance(learner.model.policy, RiskActorCriticPolicy) else
                       MASKED_PPO_CONTRACT if isinstance(learner.model.policy, BoundaryActorCriticPolicy) else None)
    if actual_contract != saved["config"].get("ppo_contract"):
        raise RuntimeError("Source policy class and masked contract differ")
    if learner.timesteps != saved["timesteps"] or learner.model.policy.features_extractor.half_width != config["feature_options"]["half_width"]:
        raise RuntimeError("Restored model and metadata differ")
    learner.enable_mask(config["hold_frames"])
    learner.configure(config["held_ppo"])
    lineage = {"source_checkpoint": str(source.parent.resolve()), "source_model_sha256": source_hash,
               "method": config["ppo_contract"], "task": args.task,
               "prepare_only": args.prepare_only, "prior": saved.get("fine_tuning")}
    run = create_run_directory(args.run_dir)
    def publish(state):
        return save_checkpoint(learner, run, {**state, "reward_profile": MASKED_REWARD_PROFILE,
                               "initialization": saved["initialization"], "fine_tuning": lineage},
                               config, saved["build_md5"])
    with DiagnosticLog(run/"training.jsonl") as log:
        log.emit({"event": "masked_preparation", "task": args.task, "config": config,
                  "source_model_sha256": source_hash, "prepare_only": args.prepare_only,
                  "acquisition_contract": ACQUISITION_CONTRACT})
        if args.prepare_only:
            path = publish({"updates": saved["updates"], "timesteps": learner.timesteps})
            log.emit({"event": "prepared_checkpoint", "checkpoint": str(path)})
            return
        reader, pid, build = connect(args.pid)
        try:
            if build != saved["build_md5"] or "SPELL_RESULT_ACTIVE" not in reader.addresses:
                raise RuntimeError("Masked training requires the matching verified game build")
            controls = ActionEmulator(WindowsKeyboard(pid))
            session = GameSession(reader, controls, repeat=2, capacity=config["capacity"],
                                  max_frames=config["max_frames"], spell_retry=True)
        except BaseException:
            reader.close()
            raise
        with session:
            controls.wait_for_focus(args.focus_timeout)
            session.verify_training_boundary()
            initial = publish({"updates": saved["updates"], "timesteps": learner.timesteps})
            log.emit({"event": "training_start", "task": args.task, "checkpoint": str(initial),
                      "additional_decisions": args.decisions, "rollout_decisions": args.rollout_decisions,
                      "reward_profile": MASKED_REWARD_PROFILE, "fine_tuning": lineage,
                      "timing_clock": timing_clock_info()})
            collector = partial(collect_masked_episodes, hold_frames=config["hold_frames"],
                                gamma_frame=config["held_ppo"]["gamma_frame"])
            train_episodic(TouhouEnv(session), learner, session.verify_training_boundary,
                          publish, log.emit, args.decisions, args.rollout_decisions,
                          saved["updates"], clock=time.perf_counter, collector=collector)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped; keys released. Resume only from a complete checkpoint.", file=sys.stderr)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
