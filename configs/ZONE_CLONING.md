# Held attack-zone teacher cloning

This is a separate versioned pipeline. Legacy demonstrations, models, features
and rewards retain their behavior. HP feedback is not used.

## Collection and explicit validation

Start at the same spell's result menu and keep focus. Avoid recorder shortcuts
containing Shift or direction keys while input is owned by the controller:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_attack_zone --half-width 24 --hold-frames 4 --episodes 10 --task spell01-character01-normal --movement-probe --output runs/demos/zone24-teacher.jsonl
```

This remains an attack_zone_evaluation trace. The legacy --demos loader rejects
it. --zone-demos explicitly validates its original schema, teacher parameters,
every recomputed raw ranking, actual held action, hold age, observation chain,
frame sequence and complete episode boundary. It also requires aligned playable
slow-mode telemetry consistent with the 2 px/frame prediction. Missing or
inconsistent telemetry fails validation rather than silently entering training.
Original source paths and SHA-256 hashes are retained. Do not rename collection
types to bypass checks or mix widths, safety parameters or hold durations.
The single-task --zone-demos path rejects mixed task labels. Explicit multi-task
cloning is described below.

Only fresh action selections are supervised, using the observation before that
selection and the raw teacher action. Intermediate two-frame holding observations
are validated but omitted from fitting. Executed held actions are not relabeled
as new teacher decisions. This produces the attack-zone-held-v1 contract.
Successful complete episodes are used by default; failed episodes require the
explicit include-failures option. Incomplete tails are discarded.

## Offline fit and evaluation

```powershell
.\.venv\Scripts\python.exe -m scripts.clone --zone-demos runs/demos/zone24-teacher.jsonl --epochs 30 --run-dir runs/bc-zone24
```

This defaults to hazard-zone-v1, with serialized attack half-width from the
teacher config. The extractor otherwise retains the existing geometry model.
Old hazard-v1 archives retain their six-pixel default. Exact duplicate decision
trajectories are removed before the whole-episode split. Cloning still supervises
only the actor and leaves PPO counters at zero.

Evaluate the printed checkpoint with the saved hold duration:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_control --resume runs/bc-zone24/checkpoints/update-000000 --hold-frames 4 --episodes 3 --task spell01-character01-normal --output runs/bc-zone24-eval.jsonl
```

No teacher or clearance gate overrides the neural output in this evaluation;
only explicit action holding remains. Teacher performance and network imitation
performance must remain separate. The control entrypoint checks the declared
hold duration. The legacy PPO train/evaluate entry and legacy correction collector
reject this contract: their two-frame sampling/teacher assumptions do not match.
Zone-specific corrective rollouts remain future work. Held PPO now has a separate
train_held entrypoint; see HELD_PPO.md rather than the legacy collector.

## Three-episode pilot

The validated source runs/attack-zone-w24-movement.jsonl yields 262/251/255 fresh
decisions (768 total) from 1534 original transitions. No exact decision trajectory
duplicates were removed. Episodes 2 and 1 form 506 training samples; episode 0
provides 262 validation samples. This is a small, model-selected validation set.

runs/bc-zone24-pilot/cloning.jsonl ran 30 epochs and selected epoch 30. Validation
accuracy is 76.72%, movement accuracy 43.81%, and NLL 0.79955. Up and down-right
validation accuracy are zero. The checkpoint is
runs/bc-zone24-pilot/checkpoints/update-000000; keep it as an offline pipeline
pilot, not a proven avoidance model. Collect more successful episodes before
another fit, and do not compare percentages directly with older data splits.

## Ten-episode dataset and fit

runs/demos/zone24-teacher.jsonl contains ten successes, zero deaths, no dropped
bullets or radius fallbacks, 5059 original transitions and 2532 validated fresh
decisions. No exact decision trajectories were duplicated. All nine actions
occur, but up has only 17 examples. The split uses eight episodes (2020 samples)
for training and episodes 7/8 (512 samples) for validation.

runs/bc-zone24/cloning.jsonl selected epoch 30 after 30 epochs. Train accuracy
is 96.24%; validation accuracy is 92.19%, movement accuracy 77.56%, and NLL
0.24469. Down/right-down validation accuracy remains 44.44%/50%; up has only two
validation examples. This is model-selected held-out imitation, not native
success. The published checkpoint is runs/bc-zone24/checkpoints/update-000000.
Reloaded predictions match the logged validation accuracy; width=24 and zero
PPO steps are restored. See runs/bc-zone24/offline-review.json for the model hash.

The evaluation command above now targets an existing fitted model. Evaluate
three episodes before any further training. DiagnosticLog keeps full JSONL data
and a sibling .console.log transcript, displays concise progress, and repeats
actual paths at exit. Training output is in runs/bc-zone24/cloning.console.log.

## First cloned-controller native result

runs/bc-zone24-eval-001.jsonl completed three successes at 992/996/964 frames,
zero deaths, no dropped bullets or radius fallbacks, and two successful retries.
Mean time to result is 16.40 seconds. This uses the learned actor plus declared
four-frame holding, without teacher execution or candidate-gate overrides.
Holding overrides 137 of 1476 raw proposals. All three trajectories have zero
direct left/right reversals. Stable-direction movement samples are about
2 px/frame; this entrypoint does not verify focus telemetry on each frame.

This supports fixed-task feasibility, not generalization or superiority over
the teacher. Freeze this archive for ten repeated evaluation episodes before
allocating further training. The offline report is
runs/bc-zone24-eval-001-analysis.json. Held continuation now uses train_held;
the legacy PPO entry remains blocked. See HELD_PPO.md.

## Repeated cloned-controller validation

runs/bc-zone24-eval-repeat.jsonl completed ten successes with zero deaths,
dropped bullets, radius fallbacks or unresolved endings. Mean time to result
is 16.88 seconds (16.27-17.35), with nine successful retries. There are no sampled
positions at x<=16 or x>=368. Eight episodes have zero direct left/right
reversals and two have one. All 580 steady-direction motion samples are about
2 px/frame; engine focus telemetry is not sampled by this entrypoint.

The unchanged archive has now completed three plus ten fixed-task evaluations
successfully, not a cross-task or population success claim. Its hash matches
offline-review.json. runs/bc-zone24/baseline.json records checkpoint identity,
control contract and evaluation sources; the original checkpoint is unchanged.
Detailed results are in runs/bc-zone24-eval-repeat-analysis.json.

Stop repeated acceptance runs of this unchanged baseline. Held PPO collection
with matched likelihoods and measured frame returns is implemented in train_held
(HELD_PPO.md); its native smoke result is pending. Keep the legacy entry blocked
and preserve this baseline for regression checks.

## Explicit joint-task cloning

The stage-6 teacher completed seven additional successes at
704/705/705/684/757/716/738 frames, with zero deaths. Together with the initial
three episodes this gives ten unique successful trajectories, 1803 fresh labels,
and mean time to result 12.00 seconds. Strict telemetry and label validation
passed for both files. None of these teacher trajectories reached x>=320.
See runs/demos/stage06-zone24-teacher-analysis.json for source hashes and splits.

Use a separate opt-in to combine the original stage-1 and stage-6 sources:

```powershell
.\.venv\Scripts\python.exe -m scripts.clone --multitask-zone-demos runs/demos/zone24-teacher.jsonl runs/stage06-zone24-teacher-check.jsonl runs/demos/stage06-zone24-teacher.jsonl --epochs 30 --run-dir runs/bc-zone24-joint
```

Each file passes the unchanged single-task validator before aggregation. The
build, schema, capacity, frame limit, teacher parameters and holding contract
must match. Source paths/hashes and original task labels are retained per file
and episode under dataset_contract=attack-zone-multitask-v1. Action execution
remains attack-zone-held-v1. The network receives the existing observation,
without a task identifier; training remains uniform per sample.

Exact full decision trajectories are deduplicated globally before splitting.
Every declared task must retain at least two unique eligible episodes. Each
task has its own deterministic whole-episode 80/20 split with seed 0. Identity
contains the original tasks list, with no synthetic single task label;
task_splits records per-task membership and counts. Global lists are also saved.

This run has 20 unique episodes and 4335 total labels. Stage 1 keeps its original
eight training/two validation episodes (2020/512 labels); stage 6 uses eight/two
(1434/369 labels). Total training/validation counts are 3454/881. Epoch selection
uses pooled, sample-weighted validation loss; per-task offline results must be
checked separately. This is a fresh actor fit, not PPO continuation or a change
to either frozen reference. PPO counters stay zero.

Evaluate the separate candidate on the current stage-6 spell first:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_control --resume runs/bc-zone24-joint/checkpoints/update-000000 --hold-frames 4 --episodes 3 --task stage06-straight-dense-character01-normal --output runs/bc-zone24-joint-stage06-eval.jsonl
```

Then manually select the original stage-1 spell with the same character,
equipment and difficulty, reach its result menu, and check regression:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_control --resume runs/bc-zone24-joint/checkpoints/update-000000 --hold-frames 4 --episodes 3 --task spell01-character01-normal --output runs/bc-zone24-joint-stage01-eval.jsonl
```

Both tasks are now seen during training, so no task-transfer flag is needed.
Evaluation logs retain training_tasks and the actual evaluation_task. An unseen
third task still requires explicit transfer and a reason. Do not call the new
stage-6 result zero-shot generalization. Task selection remains manual, and each
run saves a sibling console log. The single-task train_held entry still rejects
the joint identity; multi-task PPO scheduling is not implemented. Edge-trap
correction collection remains pending. Initial native results are recorded below.

## Joint candidate offline result

runs/bc-zone24-joint selected epoch 30 by pooled validation loss. Training
agreement is 96.35%; pooled validation is 821/881 (93.19%), NLL 0.21878.
Reloaded per-task results on the same recorded validation episodes are:

| Model | Stage-1 agreement | Stage-1 movement | Stage-6 agreement | Stage-6 movement |
| --- | --- | --- | --- | --- |
| Frozen clone | 92.19% | 77.56% | 84.28% | 64.79% |
| Frozen PPO update 4 | 92.58% | 78.85% | 84.28% | 64.79% |
| Joint candidate | 94.92% | 84.62% | 90.79% | 76.76% |

Stage-1/stage-6 denominators are 512/369 observations and 156/142 movement
labels. These validation episodes selected the epoch; they are not independent
native tests. The candidate restored zero PPO counters and width 24, with finite
parameters. Model SHA256 is
a7a0123cb7b5800121a3e5bd0f4f15c4d851dbf23540bfc5e4f78c17fd181964.
Both frozen reference hashes are unchanged. See offline-review.json in the new
run for source hashes, task splits and reloaded-model checks. The full offline
suite passed 209 tests. Initial native joint-model results follow.

## Initial joint-model native acceptance

Both evaluations used the same unchanged archive, matching the offline SHA256.
Each recorded both original training tasks, the actual seen evaluation task,
hold=4, repeat=2, width=24 and capacity=128. No teacher override was used.

| Task | Outcomes | Frames to result | Mean seconds | Direct left/right reversals |
| --- | --- | --- | --- | --- |
| Stage 6 | 3 successes, zero deaths | 731/720/705 | 11.98 | 4/1/0 |
| Stage 1 | 3 successes, zero deaths | 924/954/1003 | 16.01 | 0/1/0 |

Both runs completed two retries with no unresolved endings or radius fallbacks.
All frame deltas were 1 or 2. Steady interior motion samples were 2 px/frame
(104 stage-6 and 138 stage-1 samples); internal slow flags were not collected.
No active sample reached x>=320 or x<=16. Stage-6 peak bullet count was 240,
with 112 omitted from observations; stage-1 peak was 69, without truncation.

This supports initial seen-task adaptation and retention of the old task.
The unpaired three-episode results do not establish a success-rate or speed
improvement over the original models. No right-edge trap occurred, so recovery
from that state is still untested. Keep task outcomes separate from the teacher
and other model results. The frozen comparison reference is recorded in
runs/bc-zone24-joint/baseline.json; details and source hashes are in
runs/bc-zone24-joint-native-analysis.json. The earlier offline review remains an
unchanged record of the pre-native checks.

Before more fitting or PPO, repeat ten episodes of the denser stage-6 task with
the same frozen archive, character, equipment, difficulty and result-menu start:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_control --resume runs/bc-zone24-joint/checkpoints/update-000000 --hold-frames 4 --episodes 10 --task stage06-straight-dense-character01-normal --output runs/bc-zone24-joint-stage06-repeat.jsonl
```

This checks repeated closed-loop behavior; it does not deliberately introduce
edge-trap states. The repeat result is below. Explicit edge corrections and
multi-task PPO remain pending.

## Stage-6 repeated acceptance

runs/bc-zone24-joint-stage06-repeat.jsonl completed ten successes, zero deaths,
nine retries and no unresolved endings. Frames to result were
698/710/696/690/685/716/711/780/720/717; mean time was 11.87 seconds, median
11.84, range 11.42-13.00. This is 3563 transitions and 7123 logical frames.

Direct left/right reversals were 1/1/0/2/0/0/2/4/1/0. No active sample reached
x>=320 or x<=16; active position ranges across episodes were x=107.17-258.83,
y=384.00-422.04. All 374 steady interior motion samples were about 2 px/frame.
Frame intervals were 1 or 2. Peak bullet count was 238, with 110 omitted from
observations; there were no radius fallbacks. Internal focus telemetry was not
collected, and the omitted bullets cannot be reconstructed from these traces.

The archive SHA256 still matches the frozen joint reference. That exact model
has now completed 13 stage-6 successes and three stage-1 regression successes.
Keep the task counts separate from other models and the teacher. Reports and
source hashes are in runs/bc-zone24-joint-stage06-repeat-analysis.json;
baseline.json retains initial evaluations and adds the repeat evidence/totals.

This completes the planned repeated acceptance on the seen stage-6 spell.
Do not keep extending identical acceptance runs or fit again just for these
successes. Preserve this reference for the next experiment on a manually chosen
unseen spell with different bullet motion. The selected task and first trial
are recorded below. Explicit edge-recovery testing remains a separate coverage item:
avoiding entry into the old trap does not establish recovery from it. Multi-task
PPO also remains unimplemented. No code, weights or training parameters changed;
the most recent full offline test result remains 209 passing tests.

## Unseen stage-6 spiral and delayed-bullet trial

The user selected another stage-6 spell: a spiral emission pattern whose paths
spawn additional bullets that begin moving after a delay. This is a user
description, not a measured trajectory or a verified spell identifier. Spiral
emission geometry does not by itself establish that individual bullets curve.

Use the manual alias stage06-spiral-delayed-character01-normal. Keep the same
character, equipment and difficulty. The frozen joint model has not trained on
this task. From this new spell's result menu, run three episodes:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_control --resume runs/bc-zone24-joint/checkpoints/update-000000 --hold-frames 4 --episodes 3 --task stage06-spiral-delayed-character01-normal --allow-task-transfer --transfer-reason "Unseen stage 6 spiral pattern with trail-spawned delayed-moving bullets" --output runs/bc-zone24-joint-stage06-spiral-eval.jsonl
```

The task-transfer identity was checked offline against the unchanged reference
archive. The first trial is recorded below; weights remained frozen and no
new-task fitting preceded it.
The command retains hold=4, repeat=2, width=24, capacity=128 and the existing frame
limit. JSONL and a sibling console transcript are saved automatically.

The reader retains states 1-4 and does not filter solely on zero velocity.
Observation rows include state/4, but the current hazard-zone-v1 extractor does
not use that field. Its forecast is constant velocity over 12 frames; birth,
arming/launch timers, acceleration and observation history are absent. Whether
every delayed-bullet phase is readable under the existing state filter still
requires native evidence. A nominal zero velocity is not an activation timer.

Inspect success/death/timeout separately, critical observations around delayed
launches, bullet counts/truncation, radius fallbacks and frame intervals. Rows
retain only nearest-128 encoded bullets with no stable slot IDs, so strict
birth/activation tracking may require a later read-only diagnostic extension.
Do not label an inferred row match as verified bullet identity, or infer the
exact launch delay from these fields alone. If failures occur, separate state
coverage from the teacher/feature forecast assumptions before fitting more
straight-line demonstrations. Source weights and training identities stay fixed.

## Spiral trial result and corner failure

runs/bc-zone24-joint-stage06-spiral-eval.jsonl completed two successes at
1372/1187 frames and one failure at 881 frames with one death. This is the
first zero-shot trial of the frozen joint model on that task. Model hash matches
the reference; frame intervals are 1 or 2, with no radius fallbacks. Peak bullet
count was 278 and up to 150 were omitted from observations. Stationary encoded
bullets and states 1/4 are present; this does not identify activation semantics.

The failed third episode spent 294 active frames at x>=320. Of 74 fresh
right-side decisions, 59 had a leftward candidate passing the original teacher
clearance gate, and the network selected one only 14 times. At logged end frames
590/602, leftward forecast clearance was 24 px while the actor chose right.
Those are local forecasts from retained observations, not proof of a successful
counterfactual trajectory. The first successful episode also reached the right
edge, so edge occupancy alone does not classify failure.

Of 220 fresh decisions in the failure, 35 chose an action outside the teacher's
12-frame wall forecast. At x=376, right commands produced zero displacement;
up-right commands produced vertical-only motion. Hazard features expose wall
validity but the actor has no hard action mask. Forecast geometry assumes the
nominal direction, so selecting a wall-clamped direction can invalidate its
predicted clearance. Around logged end frame 862, left was predicted dangerous
and down safer; the actor assigned right about 90.44% probability and chose it.
Actor probability is not calibrated survival probability.

The nearest encoded threat observations before collision include a bullet from
above-left with velocity about (0.691,0.981) px/frame. This is consistent with
the user's observation, but no stable slot ID identifies the exact fatal bullet.
The final death counter is delayed relative to the first frozen player-position
samples. The trace does not prove delayed activation caused this death: missed
earlier retreat and repeated wall-directed fresh choices are already observable.

Full source hashes, local teacher audits, critical actor probabilities and
threat observations are in
runs/bc-zone24-joint-stage06-spiral-eval-analysis.json. No weights or feature
settings changed. Before fitting more data, compare the unchanged rule teacher
on this same spell from its result menu:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_attack_zone --half-width 24 --hold-frames 4 --episodes 3 --movement-probe --task stage06-spiral-delayed-character01-normal --output runs/stage06-spiral-zone24-teacher-check.jsonl
```

This is a distinct teacher-control diagnostic, not the network's performance.
If it succeeds, prioritize explicit edge-state correction and feasible action
selection rather than assuming more centered success trajectories cover escape.
If it fails, inspect the prediction/ranking and lifecycle observations before
using new labels. The teacher comparison completed three successes. A separate
optional boundary-mask experiment is now implemented in BOUNDARY_CONTROL.md;
the default actor and teacher modes remain unchanged. Masked native results,
held edge-state corrections and lifecycle tracking remain pending.
