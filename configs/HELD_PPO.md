# Held-decision PPO continuation

Use scripts.train_held for attack-zone-held-v1 clones or its own held-smdp-v1
checkpoints. The legacy scripts.train remains blocked for these models. Preserve
the successful clone and runs/bc-zone24/baseline.json for independent comparison.

## Sampling and time contract

Sample the network once at a fresh decision and retain that observation, action,
value and log probability. Execute exactly that sampled action on successive
two-frame session steps until at least the saved hold duration has elapsed or
the episode terminates. Intermediate observations are not separate PPO actions;
there is no ActionCommitment override, teacher or safety filter in collection.

The game is still real-time, so actual duration can exceed four frames. An early
terminal can end sooner. Every transition stores measured duration_frames.
Collect complete episodes and update only at verified result menus. Artificial
truncations, unknown endings, focus/read errors abort the batch. The last batch
row must be truly terminal; no artificial bootstrap or cross-episode GAE occurs.

The decision rate matches held evaluation logically, but intermediate neural
inference is omitted in training. Changed host timing can affect real-time native
trajectories; identical trajectories are not guaranteed.

## Frame returns

Define gamma_frame=sqrt(0.999) and lambda_frame=sqrt(0.95), preserving the old
two-frame reference decay. For duration d, bootstrap uses gamma_frame**d and
the GAE recursion uses (gamma_frame*lambda_frame)**d, with episode-boundary masks.
Advantages/returns are populated explicitly; SB3 still supplies PPO optimization.

Per observed frame, survival reward is 0.0005. Success adds 5 and failure -1 at
the final observed frame. Microstep rewards are discounted within the held
interval; GAE applies the remaining duration discount between intervals. The
version is spell-v2-frame-discounted. It is not an unmarked continuation of the
old reward stream. A clone has zero PPO history, so migration is explicit at
initialization; further resumes require matching contract, rewards and settings.
No HP, alignment or imitation regularization term is added.

## Conservative first update

| Setting | Value |
| --- | --- |
| Learning rate | 1e-5 |
| Optimization epochs | 2 |
| Minibatch | 64 |
| Entropy coefficient | 0 |
| Clip range | 0.1 |
| Target approximate KL | 0.005 |

The cloned critic was not supervised. These settings reduce update pressure but
do not prevent forgetting or guarantee improved play. Sampling is stochastic,
unlike deterministic baseline evaluation. Do not compare raw training success
fractions to the frozen baseline as a controlled improvement estimate.

Start from the same manually selected spell's result menu and keep focus:

```powershell
.\.venv\Scripts\python.exe -m scripts.train_held --resume runs/bc-zone24/checkpoints/update-000000 --task spell01-character01-normal --decisions 256 --rollout-decisions 256 --run-dir runs/ppo-zone24-smoke
```

Budget units are fresh held decisions, not two-frame substeps. The minimum batch
and final whole episode may exceed 256. This invocation produces one optimizer
update after its complete batch. It writes a starting checkpoint to a new run,
then update-000001, without modifying the source archive. Actual paths are printed
at start/end, with full training.jsonl and compact training.console.log.

Inspect metrics, successes, actual frame counts and checkpoint integrity before
another training run. Then evaluate the new archive through the existing held
entrypoint; keep the printed directory if a numeric suffix was added:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_control --resume runs/ppo-zone24-smoke/checkpoints/update-000001 --hold-frames 4 --episodes 3 --task spell01-character01-normal --output runs/ppo-zone24-smoke-eval.jsonl
```

Resume from the last completed held checkpoint using train_held and a new run
directory. --decisions is additional work. Policy, optimizer, settings, counters,
original cloning provenance and source model hash persist; game state and exact
native/Python random streams do not. A source clone may only initialize this
path with zero PPO counters. The first native update and its separate post-update
evaluation are recorded below.

## Offline verification

Tests cover action/likelihood association, no intermediate resampling, short
terminal holds, frame overshoot, intra-hold rewards, duration GAE and episode
boundaries, aborted batches, menu updates, additional budgets, configuration
validation, real gradients, saved optimizer restoration, second updates, and CLI
checkpoint publication without touching the source. The real clone also passed
offline load/configuration with unchanged archive hash; see
runs/bc-zone24/held-ppo-preflight.json. This did not connect to the game or update
the baseline weights.

## First native smoke update

runs/ppo-zone24-smoke/training.jsonl completed two successful pre-update sampled
episodes (932/1035 frames), 492 fresh decisions, 984 substeps and 1967 frames.
491 decisions lasted four frames; one terminal lasted three. Sampling took
36.75 seconds and the single update 0.137 seconds. Approximate KL was 1.1793e-5,
clip fraction zero, and no errors/nonfinite values were logged. Explained
variance was -0.21978; the cloned critic was not pretrained and two episodes
do not establish reliable value prediction.

The published update-000001 reloads 492 decisions, width=24 and optimizer state.
Its initial policy equals the frozen clone and the original archive hash is
unchanged. On the original 512 teacher validation states, action/movement
accuracy changes from 92.19%/77.56% to 92.58%/78.85%, with only two argmax
changes. This is a drift check on an existing validation set, not evidence of
native improvement. Details and hashes are in offline-review.json in the run.

The two sampled successes preceded optimization; the later deterministic
post-update evaluation is recorded separately below.

## First post-update evaluation

runs/ppo-zone24-smoke-eval.jsonl evaluates update-000001 deterministically with
four-frame holding. All three episodes succeed without deaths at 926/919/1021
frames (mean 15.92 seconds), with no direct left/right reversals, sampled edge
positions, dropped bullets, radius fallbacks or errors. The model hash is
unchanged since the update. Three unpaired trials do not establish superiority
over the clone's 13 successful baseline evaluations.

The evaluation report is runs/ppo-zone24-smoke-eval-analysis.json. Next validate
multiple updates and restored optimizer state with a modest additional budget:

```powershell
.\.venv\Scripts\python.exe -m scripts.train_held --resume runs/ppo-zone24-smoke/checkpoints/update-000001 --task spell01-character01-normal --decisions 2048 --rollout-decisions 512 --run-dir runs/ppo-zone24-2k
```

The changed rollout minimum is recorded in the run header; rewards, control and
optimizer settings stay fixed. Complete episodes can exceed budget, and final
update numbers must come from actual logs. Inspect the run before further
continuation, then evaluate its last complete checkpoint. The completed run is
recorded below.

## Multi-update continuation result

runs/ppo-zone24-2k/training.jsonl adds 2271 fresh decisions across nine successful
sampled episodes and updates 2-4, reaching 2763 cumulative decisions. Batches have
777/739/755 decisions; whole episodes explain budget overshoot. Approximate KL
is 4.7711e-5/1.7275e-5/1.3599e-5 and clipping remains zero. All recorded decision
values are finite with no logged errors. Final explained variance is -0.14688;
the critic is not yet established as reliable.

The new run's initial policy and Adam states match the source update 1 exactly.
All checkpoint metadata/contract checks pass, latest.json points to update 4,
and the final archive restores 2763 decisions and eight cumulative optimizer
epochs with finite parameters. The source archive hash is unchanged. On the
old 512 teacher validation states, update 4 has no argmax changes relative to
update 1 (92.58% overall, 78.85% movement accuracy). This does not establish
closed-loop equivalence or improvement. See offline-review.json in the run.

The nine episodes use mixed pre-update stochastic policies. Evaluate the final
archive separately before more training:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_control --resume runs/ppo-zone24-2k/checkpoints/update-000004 --hold-frames 4 --episodes 3 --task spell01-character01-normal --output runs/ppo-zone24-2k-eval.jsonl
```

The final update-4 native evaluation is recorded below. The old legacy PPO entry
remains incompatible with this contract.

## Final update-4 evaluation and milestone

runs/ppo-zone24-2k-eval.jsonl completed three successes at 982/985/1010 frames,
zero deaths and no dropped bullets, radius fallbacks, unresolved endings or
errors. Mean time to result is 16.54 seconds. Direct left/right reversals are
0/1/0, with no sampled positions at x<=16 or x>=368. The archive hash matches
the training review. Reports are in runs/ppo-zone24-2k-eval-analysis.json and
runs/ppo-zone24-2k/evaluated-model.json; the source checkpoint is unchanged.

This completes the current fixed-task cloning/held-PPO/update/resume/evaluation
milestone. Clone, update-1 and update-4 outcomes are separate model results, not
one pooled success streak. Small unpaired trials do not establish improvement
over the clone or reliable generalization. Stop growing the same-task budget.

The next comparison uses the user's stage-6 spell with faster, denser, straight
round bullets. Keep character/equipment/difficulty fixed and both networks frozen.
This preserves a similar bullet family, but does not isolate speed or density
from all other spell differences. No new-spell results are available yet.

## Frozen cross-spell comparison

Control evaluation now accepts a different manual task label only with
`--allow-task-transfer` and a nonempty `--transfer-reason`. It records
`training_task`, `evaluation_task`, `task_transfer`, `transfer_reason`, manual
task verification, and the source model SHA256. Existing `task` names the actual
evaluation task. Source metadata stays unchanged; training task guards and the
saved hold duration still apply. The game does not verify spell names, character,
equipment or difficulty, so select them manually before starting each run.

Use `stage06-straight-dense-character01-normal` as a descriptive manual alias for
the proposed stage-6 spell, not a claim that its spell-card number is 6. Record
its actual displayed name/number with the results. If this alias describes the
wrong character or difficulty, correct it and keep the two runs identical.

From the selected new spell's result menu, run the clone first:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_control --resume runs/bc-zone24/checkpoints/update-000000 --hold-frames 4 --episodes 3 --task stage06-straight-dense-character01-normal --allow-task-transfer --transfer-reason "Stage 6 straight round bullets; faster and denser; frozen comparison" --output runs/bc-zone24-stage06-eval.jsonl
```

After it finishes, keep the same spell, character, equipment and difficulty,
return to its result menu, then run PPO update 4 separately:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_control --resume runs/ppo-zone24-2k/checkpoints/update-000004 --hold-frames 4 --episodes 3 --task stage06-straight-dense-character01-normal --allow-task-transfer --transfer-reason "Stage 6 straight round bullets; faster and denser; frozen comparison" --output runs/ppo-zone24-stage06-eval.jsonl
```

Each command also saves a sibling `.console.log`; actual numbered paths are
printed at entry and exit. These runs use deterministic inference with shooting,
slow movement, repeat=2, capacity=128, and the checkpoint's frame limit. They do
not train, collect teacher labels, or adjust the hold duration/attack width.
Compare success/failure, survival frames, reversals/edges and death trajectories.
Check `max_bullet_count`, `max_dropped_bullets`, radius fallbacks and actual frame
deltas for the denser spell. Capacity retains the nearest 128 bullets; a positive
dropped count is a representation limit, not proof that omitted bullets caused
a death. Do not increase capacity during this initial comparison. Completion
times across different spells are not directly comparable.

Three episodes per model are exploratory and are not paired by random seed.
Before any new demonstrations or adaptation, save both frozen results. If both
fail, examine coverage, control timing and the teacher separately; keep the old
spell available for later forgetting checks. The transfer flag only applies to
evaluation, never to training or checkpoint relabeling.
