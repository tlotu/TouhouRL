# Learning algorithms

`baselines.py` provides seeded random, stationary, reactive bullet avoidance,
and `reactive_attack` policies. All use encoded observations and return action IDs
0-8. The reactive policy predicts linear bullet motion over 12 frames and assumes
fixed low-speed movement at 2 px/frame. It ignores enemies, lasers, acceleration,
and future spawns. This is a comparison baseline, not a trained model.
Holding Z does not ensure damage: this policy never aligns with enemies.
Surviving a spell timeout is not necessarily a successful spell clear.

`reactive_attack` prefers visible, collidable bosses above the player. It scores
the same bullet trajectories as `reactive`, then considers attack alignment only
among paths with at least 4 px clearance and within 1 px of the best clearance.
Tight situations use clearance alone. Attack scoring predicts horizontal alignment
four frames ahead with a 6 px dead zone and discourages approaching within 64 px
vertically. Without a suitable target it falls back to `reactive`.
This is predicted bullet safety, not a guarantee: lasers, enemy body hitboxes,
acceleration, and future spawns are still outside the model. HP is not an input.

`training.py` collects complete episodes and checks result-menu update boundaries.
`ppo_backend.py` uses SB3 PPO/GAE with variable-length batches, bypassing its
auto-reset learn() loop. `checkpoints.py` publishes complete policy/optimizer
checkpoints with build/schema/reward metadata. Optional dependencies are in
requirements-training.txt; see configs/TRAINING.md before the first live run.

`demonstrations.py` validates teacher JSONL and splits complete episodes into
training/validation sets. `cloning.py` fits the actor offline and selects weights
by held-out cross-entropy. Source hashes, episode splits, and per-action metrics
are stored with PPO-compatible checkpoints. See configs/IMITATION.md.

`corrections.py` mixes a frozen student and teacher for corrective capture. It
keeps executed actions distinct from supervised teacher labels and reports
disagreement. Complete correction episodes can be aggregated with original
teacher traces, including learner failures with valid teacher labels.

`hazard_features.py` optionally replaces flattened observations with 62 geometric
inputs for the actor/critic. It predicts bullet clearance for each action and
adds boundary and attack geometry. Bullet order does not affect the result.
It uses the baseline's approximate motion model, with no action override or
teacher action input. Enable with scripts.clone --features hazard-v1; checkpoint
loading restores the extractor for evaluation and PPO continuation.
