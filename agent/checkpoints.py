"""Publish complete checkpoint directories without overwriting models."""
import json
from pathlib import Path
import time
import uuid

from env.session import OBSERVATION_SCHEMA
from .training import REWARD_PROFILE


def _publish_directory(pending, target):
    # Windows can briefly deny directory renames after an archive is closed.
    for attempt in range(5):
        if target.exists():
            raise FileExistsError(target)
        try:
            pending.rename(target)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.05 * (2 ** attempt))


def create_run_directory(requested):
    requested = Path(requested)
    for index in range(10000):
        candidate = requested if not index else requested.with_name(f"{requested.name}-{index:03d}")
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise FileExistsError(f"No unused run directory for {requested}")


def save_checkpoint(learner, run_dir, state, config, build):
    root = Path(run_dir) / "checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"update-{state['updates']:06d}"
    if target.exists():
        raise FileExistsError(target)
    pending = root / f".pending-{uuid.uuid4().hex}"
    pending.mkdir()
    learner.save(pending / "model.zip")
    if not (pending / "model.zip").is_file() or not (pending / "model.zip").stat().st_size:
        raise RuntimeError("Backend did not produce a nonempty model archive")
    metadata = {"format": 1, "complete": True, "schema": OBSERVATION_SCHEMA,
                "reward_profile": REWARD_PROFILE, "build_md5": build,
                "config": config, **state}
    (pending / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    # An interrupted save leaves a pending directory, never a published model.
    _publish_directory(pending, target)
    pointer = root / f".latest-{uuid.uuid4().hex}.json"
    pointer.write_text(json.dumps({"checkpoint": target.name}), encoding="utf-8")
    pointer.replace(root / "latest.json")
    return target


def read_checkpoint(path, *, evaluation=False):
    path = Path(path)
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if (metadata.get("format") != 1 or metadata.get("complete") is not True
            or metadata.get("schema") != OBSERVATION_SCHEMA
            or metadata.get("reward_profile") not in (("spell-v1", REWARD_PROFILE) if evaluation else (REWARD_PROFILE,))
            or not (path / "model.zip").is_file()):
        raise ValueError("Incomplete or incompatible training checkpoint")
    if not isinstance(metadata.get("timesteps"), int) or not isinstance(metadata.get("updates"), int):
        raise ValueError("Checkpoint counters are invalid")
    if metadata["timesteps"] < 0 or metadata["updates"] < 0:
        raise ValueError("Checkpoint counters must be nonnegative")
    return metadata
