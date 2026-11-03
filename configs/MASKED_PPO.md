# Masked held PPO

`scripts.train_masked` converts an untouched attack-zone clone or resumes a
`masked-held-smdp-v1` checkpoint. The first run uses the frozen joint clone on
one explicitly selected seen task. Original demonstration identities remain
intact; `fine_tuning.task` records the active task and source hashes preserve
the checkpoint chain. Automatic task scheduling and unseen-task adaptation
are not implemented. `train_held` remains a separate, incompatible entry.

## Policy and timing

The policy removes directions whose nominal four-frame endpoint falls outside
x=8..376, y=16..432 at 2 px/frame, then normalizes the remaining probabilities.
Sampling, stored log probabilities, action reevaluation, entropy and PPO ratios
use that same distribution. Forbidden directions have zero probability. This
is a physical feasibility mask, not a bullet-collision guarantee.

Copying the actor/critic and optimizer preserves all learned tensors. Switching
policy class changes archive bytes without training. Raw diagnostic logits
remain available separately. Before each update, stored behavior likelihoods
must match the current masked policy; externally replaced actions are rejected.

A fresh sampled action executes on successive requested two-frame steps, up to
four measured frames or termination. A remaining-hold boundary check can end a
macro early; the next macro samples a new action. It never substitutes an action
under the previous action's likelihood. Actual overshoot and shorter intervals
enter the existing duration-aware returns/GAE. Complete episodes are collected;
updates occur only at verified result menus. Focus/read errors and artificial
truncations abort incomplete batches.

## Reward and update settings

The explicit new reward profile is `spell-v3-masked-survival`:

| Event | Reward |
| --- | --- |
| Each observed logical frame | +0.0005 |
| Native spell success | +5 |
| Native spell failure with a recorded death | -5 |
| Native spell failure without a recorded death | -1 |

Terminal bonuses occur at the final observed frame. Intra-macro rewards and
cross-macro returns use frame discounts: gamma=sqrt(0.999), lambda=sqrt(0.95).
There is no absolute-height, HP, alignment or clear-time reward. The stronger
death penalty expresses a preference, not a strict survival-first guarantee.
Discounting still favors earlier success. Ordinary-stage progression and
timeout success are not implemented or inferred from practice-mode failures.
The diagnostic environment return retains its original meaning; optimization
uses the separately logged training return.

Learning rate=1e-5, optimization epochs=2, minibatch=64, entropy coefficient=0,
clip range=0.1 and target approximate KL=0.005. These conservative settings do
not guarantee improvement or prevent forgetting. Collection is stochastic;
compare final checkpoints with separate deterministic evaluations.

## Prepared source and first native run

Offline preparation has completed at
`runs/masked-zone24-preflight/checkpoints/update-000000`, derived from
`runs/bc-zone24-joint/checkpoints/update-000000`. Both have zero PPO updates.
`runs/masked-zone24-preflight/offline-review.json` records exact tensor equality,
the unchanged source hash, and 2341 historical observations with matching masked
argmax actions. Maximum conditional-probability error was 2.98e-7. This review
used no native input and made no gradient update on the prepared real model.

Select the previously tested dense straight stage-6 spell, with the same
character, equipment and difficulty, and start from its result menu:

```powershell
.\.venv\Scripts\python.exe -m scripts.train_masked --resume runs/masked-zone24-preflight/checkpoints/update-000000 --task stage06-straight-dense-character01-normal --decisions 256 --rollout-decisions 256 --run-dir runs/ppo-masked-zone24-smoke
```

Budget units are fresh decisions. Collection finishes whole episodes, so the
256-decision minimum can be exceeded. This invocation publishes update 0 and
one optimized update 1 into a new run without modifying the source. Inspect
`training.jsonl` and `training.console.log`; numbered directories may be chosen
if the requested directory exists. The first completed native run is recorded
below.

After reviewing that run, evaluate its actual final checkpoint using
`scripts.evaluate_control --boundary-mask --hold-frames 4`, first on the trained
task and then on stage 1 and the unseen spiral task with explicit task-transfer
metadata where required. Masked checkpoints require `--boundary-mask` so that
evaluation preserves remaining-hold checks and action provenance. Retain the
frozen joint reference and its historical code hashes unchanged.

Resume only through `train_masked` with a matching contract and settings.
Task switching, if later requested, must be explicit and separately evaluated.
Do not load training/evaluation traces as pure teacher demonstrations.

## Offline checks

The full suite passes 234 tests. New coverage includes scalar/tensor mask
agreement, sampled and reevaluated likelihoods, actual actor gradients,
optimizer tensor restoration and a second update, rejection of mismatched
likelihoods, remaining-hold interruption, source-preserving preparation,
evaluation guards, and checkpoint publication using a fake native session.
These are implementation checks, not native performance evidence.

## First native update

The actual run is `runs/ppo-masked-zone24-smoke-001`. Its two pre-update sampled
episodes succeeded with zero deaths at 689/724 logical frames. Collection
retained 354 fresh decisions, 707 substeps and 1413 frames; completing whole
episodes explains the budget overshoot. There were 353 four-frame decisions
and one one-frame terminal decision, with no interrupted holds. Collection
took 27.709 seconds; the single batch update took 0.144 seconds.

Approximate KL=3.6496e-5, clip fraction=0 and explained variance=-0.14764. The
critic is not yet reliable. `train/n_updates=2` counts optimization epochs;
there is one published batch update, `checkpoints/update-000001`.

`offline-review.json` verifies initial tensors against the prepared source and
original clone, unchanged source hashes, finite final parameters and restored
Adam state, matching resume settings, and the latest pointer. Actor and critic
parameters changed. The archive restores 354 decisions and twelve optimizer
parameter states, each with step=12. Original demonstration provenance remains
intact. On the original per-task held-out states, masked argmax changes are
0/512 for stage 1 and 0/369 for dense stage 6. Teacher agreement remains
94.92%/90.79%; this is an offline drift check, not a forgetting guarantee.

The two native successes preceded optimization. Evaluate the final policy
separately from the same dense-spell result menu before adding training:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_control --resume runs/ppo-masked-zone24-smoke-001/checkpoints/update-000001 --boundary-mask --hold-frames 4 --episodes 3 --task stage06-straight-dense-character01-normal --output runs/ppo-masked-zone24-smoke-stage06-eval.jsonl
```

Stage-1 regression and unseen spiral evaluation follow review of this run.
Training traces contain no per-substep observations/masks, so they cannot
support a full geometry, speed, omitted-bullet or acquisition-retry replay.
Zero terminal retry counters are not whole-episode retry totals. No additional
native input or gradient update was performed during the offline review.

## First post-update evaluation: two successes, one death

`runs/ppo-masked-zone24-smoke-stage06-eval.jsonl` completed successes at 726/723
frames and a native failure with one death at 481 frames. All three episodes
completed, with two successful retries and no unresolved endings. The evaluated
archive hash matches update 1. This is the first independent final-policy result;
the two training successes remain pre-update sampling results.

All 966 transitions replayed the boundary/holding controller correctly. There
were zero boundary replacements or hold interruptions, no modeled execution
violations, and zero recorded frame-race retry calls. Frame deltas were 1/2.
The failure stayed at x=151.03..201.17, away from either side wall. Peak bullets
were 234, omitting up to 106; both successful episodes also omitted bullets.
Radius fallbacks were zero. Capacity limits remain, without a proven causal
link to this death.

Reloaded logits match recorded logits within 4.77e-6. The starting masked BC and
updated policy differ on only three recorded observations, one fresh decision
per episode. In the failure, step 122 (244 elapsed frames) changes left to
down-left; all later recorded proposals agree. These are open-loop comparisons
on the updated trajectory, not evidence that the source would survive it.
Near elapsed frame 468, the local teacher also forecasts low clearance for all
candidates (best about 0.23 px); collision timing and a safe alternative cannot
be established from that forecast or the later menu transition.

The analysis is `runs/ppo-masked-zone24-smoke-stage06-eval-analysis.json`;
`runs/ppo-masked-zone24-smoke-001/evaluated-model.json` registers the candidate
result. Three unpaired episodes do not establish either improvement or PPO
regression. Keep the frozen BC reference and pause further training. First run
the original joint clone on the current spell with identical settings:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_control --resume runs/bc-zone24-joint/checkpoints/update-000000 --boundary-mask --hold-frames 4 --episodes 3 --task stage06-straight-dense-character01-normal --output runs/bc-zone24-joint-stage06-masked-ppo-control.jsonl
```

This is a contemporaneous, unpaired source comparison. Review it before further
PPO updates or broader regressions; no reward, duration, capacity or speed
setting has been changed in response to this single failed episode.

## Contemporaneous frozen BC comparison

`runs/bc-zone24-joint-stage06-masked-ppo-control.jsonl` completed three successes
with zero deaths at 701/706/726 frames (mean 11.85 seconds to result). Both
models retain their prior hashes and declared control settings match the first
PPO evaluation. These are unpaired native runs, not matching RNG seeds.

All 1067 transitions replayed correctly, with zero boundary replacements, hold
interruptions, modeled execution violations or recorded frame-race retries.
Frame deltas were 1/2; peak bullets=236, maximum omitted=108, radius fallbacks=0.
The 83 held-action overrides are normal holding behavior, not boundary changes.

Applying frozen PPO update 1 to these BC observations gives the same proposal
at all 535 fresh decisions. Two intermediate proposals differ, both during
active holds; replaying those proposals with the recorded timing reproduces
all 1067 BC commands. This is fixed-observation replay and does not guarantee
identical live trajectories under asynchronous native execution.

The report is
`runs/bc-zone24-joint-stage06-masked-ppo-control-analysis.json`. The new BC
three-of-three and earlier PPO two-of-three results show no demonstrated PPO
improvement; they do not establish regression from six unpaired episodes.
Retain BC as the working reference and keep additional training on hold. Run
one five-episode repeat of the unchanged PPO update 1 on the same spell:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_control --resume runs/ppo-masked-zone24-smoke-001/checkpoints/update-000001 --boundary-mask --hold-frames 4 --episodes 5 --task stage06-straight-dense-character01-normal --output runs/ppo-masked-zone24-smoke-stage06-repeat.jsonl
```

Review that complete batch, including any failure, before another native run.
Do not repeat acceptance indefinitely or change rewards/control to fit this
single initial failure. Stage-1 and spiral checks remain pending.

## Bounded five-episode repeat

`runs/ppo-masked-zone24-smoke-stage06-repeat.jsonl` completed all five episodes
successfully with zero deaths at 710/703/765/685/711 frames (mean 11.91 seconds
to result). The model hash is unchanged. All 1789 transitions replayed the
controller correctly; zero boundary replacements, hold interruptions, modeled
execution violations, frame-race retries or radius fallbacks were recorded.
Peak bullets=239, maximum omitted=111; capacity limits remain.

The same frozen update 1 now has seven successes and one death failure across
eight independent dense-spell evaluations. Keep the first failure, both batch
records, the BC source comparison and the two pre-update training successes
separate. The repeat does not establish improvement or equal reliability to BC.
The report is `runs/ppo-masked-zone24-smoke-stage06-repeat-analysis.json`.

Stop repeating this same-task check. Keep training on hold and evaluate the
unchanged archive on stage 1, then the unseen spiral spell. Manually select each
spell with matching character, equipment and difficulty, starting each command
from that spell's result menu:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_control --resume runs/ppo-masked-zone24-smoke-001/checkpoints/update-000001 --boundary-mask --hold-frames 4 --episodes 3 --task spell01-character01-normal --output runs/ppo-masked-zone24-smoke-stage01-eval.jsonl
```

After switching to the previously tested spiral/delayed-spawn spell:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_control --resume runs/ppo-masked-zone24-smoke-001/checkpoints/update-000001 --boundary-mask --hold-frames 4 --episodes 3 --task stage06-spiral-delayed-character01-normal --allow-task-transfer --transfer-reason "Frozen masked PPO update 1; unseen spiral transfer after dense-task training" --output runs/ppo-masked-zone24-smoke-spiral-eval.jsonl
```

Review these per-task results before any further optimization. Neither command
trains or incorporates spiral demonstrations. Native RNG pairing and automated
spell/character verification remain unavailable.

## Stage-1 passed; declared spiral task requires verification

`runs/ppo-masked-zone24-smoke-stage01-eval.jsonl` completed three successes with
zero deaths at 1005/982/1026 frames. All 1507 transitions replayed correctly;
zero boundary replacements, hold interruptions, frame-race retries or radius
fallbacks were recorded. Peak bullets=73, omitted=0. The reset boss signature
(900 HP at x=192, y=96) matches the historical stage-1 reference. This supports
old-task compatibility in three trials, not a general no-forgetting guarantee.

`runs/ppo-masked-zone24-smoke-spiral-eval.jsonl` also contains three genuine
native successes, at 693/698/721 frames with zero deaths. Its 1057 transitions
replay correctly and the archive hash matches the same frozen update 1.
However, its declared task conflicts with historical spiral telemetry:

| Diagnostic | New declared spiral, all three episodes | Historical spiral reference |
| --- | --- | --- |
| Reset boss HP | 750 | 1600 |
| Reset boss position | (192,128) | (192,96) |
| Bullet count near elapsed frame 200 | 96 | 39 |

The new values match the dense straight reference. This strongly suggests a
selection/label mismatch, without identifying a spell conclusively. CLI task
labels and transfer flags are declarations; the game does not verify them.
Do not count these successes as spiral transfer, merge them into dense-task
totals, or rewrite the raw logs. The new declared-spiral batch remains unverified.
In particular, its shorter completion time is not evidence of spiral improvement.

Per-file analysis reports and
`runs/ppo-masked-zone24-smoke-cross-task-analysis.json` retain outcome evidence,
reference hashes, diagnostic signatures and the unresolved identity. The
candidate registration records stage 1 as checked and spiral as pending.
Training remains on hold; BC remains the working reference.

Manually select the previously tested spiral spell that leaves delayed-moving
bullets along its trajectories, keeping character, equipment and difficulty
fixed. From that spell's result menu, collect a new, separate three-episode run:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_control --resume runs/ppo-masked-zone24-smoke-001/checkpoints/update-000001 --boundary-mask --hold-frames 4 --episodes 3 --task stage06-spiral-delayed-character01-normal --allow-task-transfer --transfer-reason "Manual spiral spell recheck after task-signature mismatch" --output runs/ppo-masked-zone24-smoke-spiral-recheck.jsonl
```

Only the pending spiral task needs another run. Diagnostic signature checks here
are offline evidence, not a newly implemented automatic spell detector.

## Corrected task identity and completed spiral check

The user confirmed the preceding spell-selection mistake. Derivative records
now resolve `ppo-masked-zone24-smoke-spiral-eval.jsonl` as dense straight, with
the original declaration and source hash retained. No raw log was rewritten.
That adds three successes to the dense-task record, not to spiral transfer.

`runs/ppo-masked-zone24-smoke-spiral-recheck.jsonl` matches the historical spiral
signature: all resets have 1600 HP at (192,96), with 39 bullets near frame 200.
The three episodes succeed with zero deaths at 1326/1158/1083 frames, mean
19.82 seconds to result. All 1784 controller transitions replay correctly.
The first episode reaches x=375.24: 39 intermediate mask-proposal changes
include 19 applied choices at fresh decisions. The other two episodes need no
mask replacements. Hold interruptions, frame-race retries and radius fallbacks
remain zero; peak bullets=268, maximum omitted=140. Direct left/right reversals
are 24/1/0. Success does not eliminate corner oscillation or capacity limits.

The same frozen update 1 now has these separate task records:

| Task | Independent native results |
| --- | --- |
| Stage 1 | 3 successes, zero deaths |
| Dense straight stage 6 | 10 successes, 1 death failure across three batches |
| Unseen spiral | 3 successes, zero deaths |

Two pre-update sampled training successes and BC comparisons remain separate.
These checks complete the initial masked-PPO update/evaluation milestone;
they do not establish improvement over BC. Keep BC as the working reference
and preserve update 1 as an evaluated candidate. No extra acceptance repeats
are scheduled.

## Bounded multi-update continuation

`runs/ppo-masked-zone24-smoke-001/resume-preflight.json` records successful
offline loading/configuration of update 1: 354 decisions, one published batch
update, two optimization epochs, twelve finite Adam parameter states at step
12. No gradient update or native input occurred in this check.

Next add 1024 fresh decisions on the previously trained dense straight spell,
with a 512-decision rollout minimum. Reward, optimizer, attack width, capacity
and action-hold settings stay fixed. Manually switch back from spiral to the
dense straight spell, keep character/equipment/difficulty fixed, and start at
its result menu:

```powershell
.\.venv\Scripts\python.exe -m scripts.train_masked --resume runs/ppo-masked-zone24-smoke-001/checkpoints/update-000001 --task stage06-straight-dense-character01-normal --decisions 1024 --rollout-decisions 512 --run-dir runs/ppo-masked-zone24-1k
```

This is additional work, not a cumulative target of 1024. Complete episodes can
overshoot either minimum; use actual logs and `latest.json` for final counters
and checkpoint numbers. A new run preserves all sources. The purpose is to
verify restored-optimizer continuation and multiple updates, not to assume
performance gains. Review `training.jsonl` and `training.console.log` before
evaluating the actual final checkpoint on the dense task, stage 1 and spiral.
This continuation is now complete; results follow.

## Completed 1k continuation

`runs/ppo-masked-zone24-1k` adds 1035 fresh decisions in six sampled episodes
(five successes, one death failure), reaching 1389 cumulative decisions.
Updates 2 and 3 contain 675/360 decisions, with approximate KL
1.8271e-5/5.1755e-6 and zero clipping. The final batch requests the remaining
349 decisions instead of the full rollout target of 512, then finishes its
second episode at 360. Total collection is 4131 frames and 2067 substeps.

The first four episodes use update 1; the final two use update 2. Final update 3
has no native evaluation yet. Neither the mixed training record nor its last
two successes is a final-policy performance estimate. Explained variance stays
negative (-0.11969/-0.11402), so critic quality remains unestablished.

`offline-review.json` verifies exact source/starting policy and Adam equality,
all checkpoint counters/configurations, finite tensors, unchanged source hashes
and the latest pointer. Restored Adam steps progress 12 -> 34 -> 46; final
update 3 has six cumulative optimization epochs. On the original held-out
teacher states, one argmax changes per task relative to update 1: stage-1
agreement 94.92% -> 95.12%, dense 90.79% -> 90.51%. These are drift checks.

Hold further training and evaluate the actual final archive from the same
dense straight spell's result menu, with character/equipment/difficulty fixed:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_control --resume runs/ppo-masked-zone24-1k/checkpoints/update-000003 --boundary-mask --hold-frames 4 --episodes 3 --task stage06-straight-dense-character01-normal --output runs/ppo-masked-zone24-1k-stage06-eval.jsonl
```

Review this result before stage-1/spiral regressions or another training budget.
Keep the evaluated update-1 candidate and frozen BC reference separately.

## Update-3 dense-task evaluation

`runs/ppo-masked-zone24-1k-stage06-eval.jsonl` completes three successes with
zero deaths at 713/711/876 frames (mean 12.78 seconds to result). The archive
hash matches the reviewed final update 3. Reset bosses (750 HP at (192,128))
and 96 bullets near frame 200 match the dense task reference.

All 1151 transitions replay the controller correctly. There are zero boundary
replacements, hold interruptions, modeled execution violations, recorded
frame-race retries or radius fallbacks. One step advances three actual frames;
the remaining steps advance one or two. The third episode reaches x=362.38
and succeeds without a mask replacement. Peak bullets=241, maximum omitted=113.
These three unpaired successes do not establish superiority over BC or update 1.
The report is `runs/ppo-masked-zone24-1k-stage06-eval-analysis.json`; the run's
`evaluated-model.json` registers this model separately from its sources.

Keep training on hold. Evaluate the same frozen update 3 for three stage-1
episodes, then manually switch to the previously verified spiral/delayed-spawn
spell for three transfer episodes. Keep character/equipment/difficulty fixed
and start each command at the selected spell's result menu:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_control --resume runs/ppo-masked-zone24-1k/checkpoints/update-000003 --boundary-mask --hold-frames 4 --episodes 3 --task spell01-character01-normal --output runs/ppo-masked-zone24-1k-stage01-eval.jsonl
```

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_control --resume runs/ppo-masked-zone24-1k/checkpoints/update-000003 --boundary-mask --hold-frames 4 --episodes 3 --task stage06-spiral-delayed-character01-normal --allow-task-transfer --transfer-reason "Frozen masked PPO update 3; spiral transfer after 1k continuation" --output runs/ppo-masked-zone24-1k-spiral-eval.jsonl
```

CLI task labels do not select or identify native spells. Retain all per-model,
per-task results and review these regressions before further optimization.
