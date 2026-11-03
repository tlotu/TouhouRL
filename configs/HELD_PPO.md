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
from all other spell differences. The first frozen results are recorded below.

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

## First stage-6 frozen results

Both runs completed two successes and one native failure. The clone failed at
329 frames with one death, then succeeded at 704/699 frames. PPO update 4 failed
at 1080 frames with zero deaths, then succeeded at 735/711 frames. The user
observed a timeout in the PPO failure; the final collidable boss sample had
217 HP. This is a native failure, not the Python 3600-frame limit. These small,
unpaired trials establish some transfer but no ranking between the two models.

The clone made a fresh rightward choice shortly before collision despite a
safer leftward forecast. PPO spent 632 active frames at x>=320, with 23 direct
left/right reversals. At 83 fresh right-side decisions the existing teacher's
clearance gate allowed a leftward action; PPO selected one at 48 of these.
Those offline forecasts are local and cannot prove an alternative trajectory
would have succeeded. Diagnostic HP decreased in the right-side phase but much
more slowly, so nonzero damage alone would not be a sufficient alignment signal.

Maximum bullet counts were 242/239, dropping 114/111 from observations at peak.
All four successful episodes also exceeded capacity. No radius fallbacks or
frame deltas above 2 were logged; current model hashes match the frozen inputs.
See runs/stage06-frozen-comparison-analysis.json for source hashes, episode
metrics and teacher forecast examples. Full omitted-bullet counterfactuals
cannot be reconstructed from these traces.

Before adaptation, evaluate the unchanged rule teacher on the same stage-6
spell, character, equipment and difficulty. Start from its result menu:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_attack_zone --half-width 24 --hold-frames 4 --episodes 3 --movement-probe --task stage06-straight-dense-character01-normal --output runs/stage06-zone24-teacher-check.jsonl
```

Keep capacity and other settings fixed. This is a diagnostic teacher run, not
an automatic cloning/training step. If the teacher succeeds, inspect new-task
state coverage and corrective labels. If it also gets trapped or collides,
inspect its local forecast and ranking before treating its labels as targets.
The first teacher run is recorded below; adaptation remains pending.

## Stage-6 teacher check

runs/stage06-zone24-teacher-check.jsonl completed three successes at
735/760/695 frames with zero deaths, mean 12.17 seconds to result. Direct
left/right reversals were 0/1/0; active x remained within 124.69-251.07 across
the runs. All 1093 playable telemetry samples passed the existing slow-mode
consistency checks; 126 steady interior motion samples were 2 px/frame.
Peak bullet count was 240, with 112 omitted from observations. No radius
fallbacks or frame intervals above 2 were recorded. Three wins are preliminary,
not a guarantee of long-run teacher reliability or a paired policy ranking.

The strict zone loader accepted all three complete, unique episodes and retained
548 fresh decision labels (184/190/174), including 178 movement labels. Both
frozen models produced exactly the same argmax actions on these observations:
473/548 teacher agreement (86.31%), or 123/178 (69.10%) when the teacher moved.
This is an offline diagnostic, not closed-loop evaluation or a held-out score
for a newly selected model. It shows no deterministic action change from the
old-task PPO update on this particular new-task sample. See
runs/stage06-zone24-teacher-check-analysis.json for source hashes and details.

Collect seven more episodes with unchanged settings to reach ten total:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_attack_zone --half-width 24 --hold-frames 4 --episodes 7 --movement-probe --task stage06-straight-dense-character01-normal --output runs/demos/stage06-zone24-teacher.jsonl
```

Keep the initial three-episode file as part of the prospective dataset. Validate
and deduplicate the two files before fitting. The current loader can combine
them because their tasks/settings match; it deliberately rejects a direct mix
with the old stage-1 task. Explicit multiple-task provenance and per-task whole
episode splits are needed before joint training. Do not relabel raw logs or
replace the frozen reference models. This teacher trace contains no x>=320
states, so later student-state corrections may still be needed for edge escape;
ordinary successful demonstrations do not establish that coverage. The seven
additional episodes have now completed successfully. Explicit joint cloning is
implemented in ZONE_CLONING.md, with a separate candidate model. Its stage-6
evaluation is now an adapted, seen-task test. Native joint-model evaluations
remain pending; the single-task held-PPO entry still rejects this multi-task
checkpoint.
