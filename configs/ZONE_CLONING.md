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
types to bypass checks or mix widths, safety parameters, tasks or hold durations.

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
