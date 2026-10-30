# Teacher demonstrations and behavior cloning

## Collect a fixed task

Select the same spell, character, equipment, and difficulty used for the
reactive_attack acceptance run. Start from its result menu and keep game focus.
The task label is a manual declaration, not an automatically verified spell ID.
Use a descriptive label instead of reusing it for a different task.

```powershell
.\.venv\Scripts\python.exe -m scripts.sample_game --run --spell-retry --policy reactive_attack --episodes 10 --capacity 128 --repeat 2 --trace --task spell01-character01-normal --output runs/demos/reactive-attack.jsonl
```

Each transition records the pre-action observation and teacher action. Transition
JSON goes only to the file; reset, periodic steps, episode results, and summary
remain visible. Files can be large. Existing filenames receive numeric suffixes;
use the actual printed path below. A focus/memory error stops capture, but earlier
complete episodes remain usable. Truncated or unfinished episodes are excluded.

## Fit offline

No game process, focus, keyboard input, or GPU is required. Training dependencies
are the same as PPO. At least two successful episodes are required; aim for ten
or more, including varied boss positions and active bullet patterns.

```powershell
.\.venv\Scripts\python.exe -m scripts.clone --demos runs/demos/reactive-attack.jsonl --epochs 30 --run-dir runs/bc-spell01
```

Multiple explicit paths may follow --demos. Schema, build, capacity, frame limit,
and task must match. The loader validates finite observation shapes, contiguous
steps, current teacher labels, and terminal episode boundaries. Duplicate files
are rejected. Successful episodes are used by default; --include-failures
explicitly includes complete classified failures for a separate experiment.
Old traces without a task label or complete observations must be recollected.

The split holds out 20% of entire episodes, with at least one on each side.
Repeated identical spell trajectories can still make this validation optimistic.
This measures imitation on teacher-visited states, not autonomous game success.
Inspect per-action counts and accuracy as well as overall accuracy; absent
actions have null accuracy. Action order is stay, up, up-right, right,
down-right, down, down-left, left, up-left.

The actor minimizes negative teacher log probability with a separate Adam
optimizer. The critic is not supervised. Epoch zero is also eligible; retain
the weights with the lowest held-out cross-entropy. The final checkpoint is
checkpoints/update-000000. Its PPO counters remain zero; supervised epochs and
teacher transitions are not native PPO interactions. Metadata stores source
SHA-256 hashes, episode split, action coverage, hyperparameters, and metrics.
PPO's optimizer starts fresh. There is no forced teacher action or safety filter
when evaluating the cloned policy.

## Evaluate, then fine-tune

First run autonomous evaluation on the manually selected task:

```powershell
.\.venv\Scripts\python.exe -m scripts.train --evaluate --resume runs/bc-spell01/checkpoints/update-000000 --episodes 3 --run-dir runs/bc-spell01-eval
```

Review the actual game behavior before PPO fine-tuning. If the clone fails,
inspect coverage and disagreement before adding PPO budget. Errors can compound
when the clone visits states absent from the demonstrations. Extra teacher
episodes may be needed; high held-out accuracy does not ensure a clear.

Once evaluation is satisfactory, start a short PPO continuation:

```powershell
.\.venv\Scripts\python.exe -m scripts.train --resume runs/bc-spell01/checkpoints/update-000000 --timesteps 2048 --run-dir runs/bc-spell01-ppo
```

The existing result-menu update boundaries and spell-v2 reward still apply.
Continuation preserves cloning provenance in every checkpoint. Keep pure PPO,
teacher performance, cloned evaluation, and PPO fine-tuning results separate.
Initial tests cover synthetic demonstrations and a real supervised/PPO optimizer
round trip. Native acceptance results are recorded below; successful teacher
capture does not establish that the cloned policy can clear the spell.

## First native dataset and offline fit

runs/demos/reactive-attack.jsonl contains ten verified successes and 4411
transitions, with no incomplete episodes, dropped bullets, or radius fallbacks.
All nine actions occur, but up has only 13 examples across the whole dataset.
The episode split uses eight training episodes (3523 samples) and two validation
episodes (888 samples). All teacher labels passed recomputation checks.

The first offline fit in runs/bc-spell01/cloning.jsonl ran 30 epochs and selected
epoch 29 by validation cross-entropy. Training accuracy was 93.05%; validation
accuracy was 87.50%, compared with 66.67% for always staying still on validation
states. Accuracy on validation samples requiring movement was 71.28%. Up-action
accuracy remained zero (only five validation examples). These metrics measure
imitation on teacher states and cannot establish autonomous avoidance.

The completed checkpoint is runs/bc-spell01/checkpoints/update-000000, with zero
PPO interaction steps and full dataset provenance.

## First autonomous clone evaluation

runs/bc-spell01-eval/training.jsonl completed three classified failures after
383, 1156, and 739 logic frames (6.38, 19.27, and 12.32 seconds). Mean duration
was 12.66 seconds. All retries worked, with no dropped bullets or radius
fallbacks. The actor chose eight different actions across these episodes;
pure up was never chosen. Stay accounted for 733 of 1140 decisions (64.30%).
Sampled positions and user observations show lower-screen, mostly horizontal
movement with some vertical adjustment. This differs from the pure-PPO
checkpoint's repeated upward charge, but three failures do not establish
reliable avoidance or a statistically supported improvement.

Next, run a controlled 2048-step PPO continuation from this unchanged clone and
evaluate the resulting checkpoint. Preserve the clone as the comparison model.
The critic was not pretrained, and PPO samples actions while this evaluation
used deterministic actions; training motion can therefore look different.
If continuation degrades behavior or remains ineffective, inspect teacher-state
disagreement and collect corrective teacher labels on learner-visited states
before allocating another large training budget. That corrective collection
workflow is described below.

## First PPO fine-tuning run

runs/bc-spell01-ppo/training.jsonl completed 2139 interaction steps, four
updates, and eight failed episodes without logged errors. Mean duration was
534.25 frames (8.90 seconds), with a maximum of 866 frames (14.43 seconds).
This stochastic training sample is not directly comparable with the clone's
deterministic evaluation. The first update logged approximate KL 0.05729;
later values were 0.02828, 0.03304, and 0.01119. The final checkpoint is
runs/bc-spell01-ppo/checkpoints/update-000004.

An offline comparison on the same 888 held-out teacher states reloaded all five
archives without native reset or input. Teacher-action accuracy declined from
87.50% (clone) to 86.37%, 84.68%, 83.22%, and 81.08% after updates 1-4. Accuracy
on movement labels declined from 71.28% to 65.54% at update 4; teacher NLL rose
from 0.51745 to 0.72167. This demonstrates drift from the teacher on that fixed
state set, not necessarily worse game performance. Preserve all checkpoints,
evaluate update-000004, and do not increase the training budget until its
autonomous behavior has been checked.

runs/bc-spell01-ppo-eval/training.jsonl subsequently recorded three failures at
414, 578, and 753 frames (mean 9.69 seconds), versus the clone's 12.66-second
three-episode mean. This small sample does not establish statistical regression,
but it provides no reason to expand the current PPO budget. Both evaluations
completed all retries without dropped bullets or radius fallbacks. Return to
the unchanged original clone for corrective collection.

## Correct labels on learner-visited states

The teacher-only dataset samples states reached by the teacher. Small clone
errors can lead to different states. `collect_corrections.py` queries both the
original clone and reactive_attack at each observed state, then samples which
action to execute. This is a DAgger-style mixed rollout, not a safety guarantee.
No online fitting occurs during capture. Teacher labels come from the baseline
heuristic, not a verified optimal controller.

Keep the original manual task unchanged and start from its result menu:

```powershell
.\.venv\Scripts\python.exe -m scripts.collect_corrections --resume runs/bc-spell01/checkpoints/update-000000 --task spell01-character01-normal --teacher-probability 0.5 --episodes 10 --output runs/demos/corrections-01.jsonl
```

Each transition distinguishes `action` (executed), `student_action`,
`teacher_action` (supervised label), and `executed_by`. Probability 0 executes
only the student; probability 1 executes only the teacher. Neither setting
changes which teacher label is recorded. The mixture seed is logged, but native
game randomness is uncontrolled. Its summary reports disagreement and teacher
execution counts; success under assistance is not autonomous clone success.

The loader accepts both complete successes and complete failures from correction
traces by default: labels are teacher suggestions even when the student fails.
This exception does not change success-only filtering for teacher-only traces.
Truncations and incomplete tails are still excluded. Validate shapes, labels,
action provenance, and task/build/config consistency before fitting. Retain the
original demonstrations and aggregate both sources:

```powershell
.\.venv\Scripts\python.exe -m scripts.clone --demos runs/demos/reactive-attack.jsonl runs/demos/corrections-01.jsonl --epochs 30 --run-dir runs/bc-spell01-corrected
```

Use the actual printed filenames if a suffix was added. This fits a fresh actor
on the combined dataset and preserves each source's collection type, student
checkpoint path, teacher probability, and SHA-256 hash. Teacher behavior cloning
remains distinct from PPO fine-tuning. Evaluate the new model autonomously before
any further PPO work:

```powershell
.\.venv\Scripts\python.exe -m scripts.train --evaluate --resume runs/bc-spell01-corrected/checkpoints/update-000000 --episodes 3 --run-dir runs/bc-spell01-corrected-eval
```

Tests cover probability extremes, seeded mixing, executed/label separation,
trace alignment across retries, failed correction inclusion, source aggregation,
and rejection of corrupt labels or mismatched tasks. Native correction capture
and the corrected model's game performance have not yet been validated.

## Corrected data and geometry feature experiment

runs/demos/corrections-01.jsonl completed ten assisted episodes: four successes
and six failures, 3720 transitions, no dropped bullets, no radius fallbacks, and
no incomplete episodes. Student and teacher disagreed on 1388 states (37.31%).
The teacher executed 1829 actions. All labels passed recomputation checks; all
ten episodes were retained because supervision uses the teacher's suggestion.
Combined with the original ten successes, the dataset has 8131 samples across
20 episodes. The split holds out four episodes (1510 samples), leaving 16
training episodes (6621 samples).

The default flattened-MLP run in runs/bc-spell01-corrected/cloning.jsonl selected
epoch 9 after 30 epochs. Validation accuracy was 64.17%, movement accuracy
26.78%, and NLL 1.05423. Always staying still scores 60.93% on this validation
set. Keep this model as a baseline rather than assuming more data solved the
problem. Later training epochs lowered training loss but did not improve the
selected validation loss.

An explicit second experiment uses `--features hazard-v1`:

```powershell
.\.venv\Scripts\python.exe -m scripts.clone --demos runs/demos/reactive-attack.jsonl runs/demos/corrections-01.jsonl --epochs 30 --features hazard-v1 --run-dir runs/bc-spell01-corrected-hazard
```

The 62 inputs contain player state, selected visible/collidable enemy offset,
bullet density, and six geometric features for each of nine candidate actions:
boundary validity, predicted bullet clearance, clearance relative to the best
valid candidate, edge distance, horizontal aiming error, and proximity below
the target. Predictions use the baseline's 12-frame linear-motion approximation
and assumed 2 px/frame player speed. No action is masked or overridden and no
teacher action ID is an input. The actor learns the decision using these
handcrafted geometric inputs. This is a different representation experiment,
not pure learning from flattened object records. Exact enemy body shapes,
lasers, acceleration, and future spawns remain unmodeled.

On the same four held-out episodes, this run selected epoch 30: validation
accuracy 92.38%, movement accuracy 84.58%, and NLL 0.22649; training accuracy
was 97.04%. Up accuracy was still only 25% (12 validation examples). Validation
was used for model selection, not as an independent final test. The dataset and
split differ from the first clone's 888-state validation; those older percentages
must not be compared directly. Both corrected-data experiments share a split.

The completed model is
runs/bc-spell01-corrected-hazard/checkpoints/update-000000. Loading the archive
restores the custom extractor automatically; retain agent/hazard_features.py.
Use the original task's result menu for the next native check:

```powershell
.\.venv\Scripts\python.exe -m scripts.train --evaluate --resume runs/bc-spell01-corrected-hazard/checkpoints/update-000000 --episodes 3 --run-dir runs/bc-spell01-corrected-hazard-eval
```

This checkpoint has not yet been evaluated in the native game. Do not start PPO
fine-tuning until its autonomous behavior is checked. Geometry tests compare
clearances and wall eligibility with the baseline, check bullet permutation and
padding invariance, verify enemy filtering, and exercise checkpoint reload plus
a real PPO update without a game.

Checkpoint publication retries transient directory rename denials up to eight
attempts with short backoff; permanent failure still leaves an unpublished
pending directory. Existing checkpoint directories are never overwritten.
