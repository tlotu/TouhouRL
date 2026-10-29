# Small complete-episode PPO

## Execution contract

Select a fixed spell/character/difficulty manually and start at a recognized
result menu on the validated baseline build. Use one CPU learner, fixed Z/Shift,
repeat=2, and state-v2 observations. Default bullet capacity is 128. Raw HP stays
diagnostic only. Keep the game focused and do not navigate during updates.

Collect with fixed policy weights until the batch reaches its minimum length,
then finish that episode. Several complete episodes may form one update. The
terminal result releases keys. Verify settled result state and focus immediately
before and after updating; reset only afterward. Menu animation still runs:
this is not engine pause or exact stepping.

SB3 supplies PPO losses, clipping, GAE, minibatches, optimizer, and serialization.
The custom collector bypasses SB3's auto-reset vector loop. Calling model.learn()
would violate this contract. The backend instead populates a complete-episode
DictRolloutBuffer and calls train(). Run the optional optimizer tests whenever
upgrading SB3 because this uses its lower-level buffer/train interfaces.

## Defaults

| Parameter | Value |
| --- | --- |
| Minimum rollout transitions | 512 |
| Actor/critic hidden layers | 128, 128 each |
| Learning rate | 0.0003, constant |
| Minibatch size / epochs | 64 / 4 |
| Gamma / GAE lambda | 0.999 / 0.95 |
| Clip / entropy coefficient | 0.2 / 0.01 |
| KL target | 0.03 |
| Compute | CPU, one Torch thread |
| Per-episode frame limit | 3600 |

Variable batches can exceed the minimum rollout length. SB3 handles partial
final minibatches. True terminal boundaries use zero bootstrap; GAE does not
cross episode starts. Artificial time-limit truncations are rejected rather than
misrepresented as terminal training examples.

## Reward profile spell-v2

Reward is +0.0005 per observed elapsed logic frame, +5 at recognized success, and
-1 at recognized failure. This differs from the diagnostic survival reward;
both returns are logged. No HP deltas, alignment reward, teacher actions, or win
inference from quiet screens are used. Outcomes are observed at result-menu
boundaries, not necessarily the final damaging frame. Gamma applies per decision,
not per native frame. Training starts from random weights, not the rule policy.
The default 3600-frame limit caps survival credit at 1.8. Longer configured
limits change this balance. No reward depends on staying near the bottom.

Historical spell-v1 used a time penalty that favored earlier failures when no
success was found. Its checkpoints remain available for evaluation only; start
a fresh run for spell-v2 rather than silently resuming with a different reward.

## Checkpoints and continuation

Each run writes training.jsonl and checkpoints/update-NNNNNN. Update records
include KL, clipping, entropy/value losses, sampling speed, update time, and the
checkpoint path. Interrupted saves leave unpublished .pending directories;
latest.json points to a completed checkpoint. Existing models are never replaced.
Update 0 saves the initialized model before collecting the first batch.

Resume restores policy, optimizer, cumulative timesteps, update number, and
configuration. Build, observation schema, reward profile, and spaces must match.
The --timesteps budget is additional work per invocation. Continuation uses a
new run directory. Game state, native RNG, rollouts, and exact Python/Torch random
streams are not restored; the game starts a fresh task.

Focus loss, invalid memory, unknown endings, or truncation abort collection
without training on the incomplete batch. Interrupted updates are not published.
Resume from the last completed checkpoint, with the game in a result menu again.

## Verification and scale

All 120 tests passed with training dependencies installed, including actual
optimizer updates, GAE boundaries, and checkpoint loading. Re-run the backend
tests when upgrading Torch/SB3.

Native smoke validation passed in runs/ppo-smoke/training.jsonl with SB3 2.9.2
and Torch 2.15.0: 514 transitions, two failed episodes, one PPO update, and a
completed update-000001 checkpoint. Sampling took 21.046 seconds (24.42
transitions/second); optimization took 0.125 seconds. KL was 0.00284 and clipping
fraction 0.00738. This validates one collection/update/save cycle, not learning
quality. Multi-update continuation passed in runs/ppo-continued/training.jsonl:
2170 additional transitions, eight failed episodes, and four further updates.
The final update-000005 checkpoint contains 2684 cumulative transitions. All
updates completed without logged errors; KL ranged from 0.00429 to 0.00646.
Sampling took 87.971 seconds and optimization took 0.515 seconds in total.
Deterministic evaluation in runs/ppo-evaluation/training.jsonl completed ten
failed episodes, each lasting 438 logic frames (7.3 seconds), with no dropped
bullets or radius fallbacks. Periodic action samples repeatedly selected down
and down-diagonal movements. Positions were not logged in this historical run,
so bottom-boundary contact is a user observation, not a measured conclusion.
This prompted spell-v2. New diagnostic logs include player coordinates during
combat and per-episode counts for all nine actions (stay, up, up-right, right,
down-right, down, down-left, left, up-left).

The spell-v2 native run in runs/ppo-v2/training.jsonl completed 4178 transitions,
15 failed episodes, and seven updates without logged errors. Checkpoint
update-000007 records this budget and reward profile. Mean episode duration was
556.67 logic frames (9.28 seconds), with a range of 348 to 1034 frames. KL ranged
from 0.00168 to 0.01118. Final policy entropy was approximately 2.176 nats, close
to the nine-action uniform entropy of 2.197. Training samples actions from the
policy, whereas the historical evaluation selected the highest-probability
action. Random-looking training motion is therefore not evidence of improvement
over that evaluation.

Deterministic evaluation of spell-v2 update-000007 in
runs/ppo-v2-evaluation/training.jsonl completed ten failures at exactly 380
frames each (6.33 seconds), with identical action counts: 112 down, 28 down-left,
and 50 up per episode. Sampled positions showed y=432 through frame 240,
followed by upward movement to y=237 at frame 360. All nine retries completed;
no bullets were dropped and no radius fallbacks occurred. This checkpoint has
not demonstrated effective avoidance. The identical episodes are repeated
trajectories under these conditions, not broad independent coverage. Keep
spell-v2 fixed for a longer continuation before changing rewards again; seven
updates and fifteen training episodes provide little learning evidence.

Continuation in runs/ppo-v2-20k/training.jsonl completed 20053 additional
transitions and 29 further updates, reaching 24231 transitions at update-000036.
All 76 episodes failed; mean duration was 527.14 frames (8.79 seconds), with a
range of 309 to 1680 frames. First-half and second-half means were 536.21 and
518.08 frames, so this run shows no sustained survival improvement. Final
entropy was about 1.971 nats; increasing action preference alone does not prove
better avoidance. No errors were logged and the final checkpoint was published.
Evaluate update-000036 before allocating another large training budget. Three
deterministic episodes suffice for an initial behavior check; identical retries
should not be interpreted as independent generalization evidence.

Evaluation of update-000036 in
runs/ppo-v2-20k-evaluation-001/training.jsonl failed all three episodes after
118 frames (1.97 seconds). Every episode selected up for all 59 decisions.
At frame 60 the player was at (192, 239), directly below a collidable boss at
(192, 96); no bullets were observed before termination. The user observed body
contact. This is consistent with the trajectory, but the result flag itself
does not identify the collision source. Enemy relative position and collidable
status are already in state-v2, so this is not a missing enemy-position input.
Stop scaling this checkpoint. A useful next experiment is behavior cloning
from the validated reactive_attack policy, followed by evaluation and PPO
fine-tuning. Keep that experiment separate from the pure-PPO baseline and
record demonstration coverage and held-out imitation accuracy. No cloning
model has been validated in the native game yet. The offline pipeline is now
implemented; see IMITATION.md for collection, held-out fitting, evaluation, and
PPO continuation. Its automated tests use synthetic teacher traces.

At nominal 60 logic FPS and repeat=2, one instance yields at most about 30
transitions/second before reset/update overhead. More rendering FPS, a different
machine, or a GPU alone does not establish faster correct sampling. Acceleration,
per-instance input, and parallel engines require separate development.
