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

The native evaluation in runs/bc-spell01-corrected-hazard-eval/training.jsonl
failed all three episodes at 348, 477, and 348 frames (mean 6.52 seconds).
Stay accounted for 487 of 587 actions (82.96%). All resets completed without
dropped bullets or radius fallbacks. The first episode remained at (192,384)
through the frame-300 periodic sample and failed at frame 348; its 18 movement
actions are not fully resolved by periodic logging. The 92.38% validation
accuracy did not translate to successful autonomous play. These three trials do
not establish general performance, but they do not justify PPO fine-tuning.

Before fitting another variant, capture complete student-only trajectories with
teacher annotations. Existing periodic evaluation logs omit observations needed
to locate the first disagreement and measure predicted clearance at that point.
Probability zero guarantees student execution; the teacher only supplies labels:

```powershell
.\.venv\Scripts\python.exe -m scripts.collect_corrections --resume runs/bc-spell01-corrected-hazard/checkpoints/update-000000 --task spell01-character01-normal --teacher-probability 0 --episodes 3 --output runs/demos/hazard-student-diagnostic.jsonl
```

Use these traces first for per-step disagreement and near-collision diagnosis,
not automatic retraining or claims of improved performance. Check whether
teacher disagreement is concentrated at necessary evasive decisions and whether
the learner reaches states absent from the mixed demonstrations. These are
hypotheses until the student-only observations are captured and inspected.

Geometry tests compare
clearances and wall eligibility with the baseline, check bullet permutation and
padding invariance, verify enemy filtering, and exercise checkpoint reload plus
a real PPO update without a game.

Checkpoint publication now reserves an exclusive target directory and writes
metadata.json last as an atomic completion marker, after closing the archive.
It no longer renames directories. Failed writes leave an incomplete directory;
read_checkpoint rejects it and latest.json does not advance before completion.
Existing checkpoint names are never overwritten. The earlier bounded directory
rename retries proved insufficient in this Windows execution environment.

## Student-only diagnosis and correction weighting

runs/demos/hazard-student-diagnostic.jsonl recorded three identical failures at
348 frames, with no teacher execution. Each episode had 174 decisions and only
four teacher/student disagreements (97.70% agreement). At decision 156, the
student continued down while the teacher chose down-left. At decisions 162,
164, and 166, the student chose left while the teacher chose right. Intervening
decisions selected right; the player alternated between x=191.41 and x=193.41
at y=415.90. Predicted clearance for the chosen direction fell from about 21.37
to 1.20 px over decisions 162-169. The 12-frame constant-action prediction is
not a guarantee when the controller repeatedly reverses after two frames.
Teacher suggestions on an already deviated trajectory are not proven recovery
actions. Counterfactual teacher execution has not been measured here.

The correction-v1 experiment removes exact duplicate observation/label
trajectories before splitting, retaining the first copy. Each ordinary sample
has weight 1, teacher/student disagreements have weight 3, and disagreements
within the last 64 observed frames of a failed episode have weight 20. Weights
are capped and affect both supervised fitting and validation-loss selection.
Failure proximity is a heuristic priority signal, not a causal collision label.
Uniform weighting remains the default for reproducibility.

```powershell
.\.venv\Scripts\python.exe -m scripts.clone --demos runs/demos/reactive-attack.jsonl runs/demos/corrections-01.jsonl runs/demos/hazard-student-diagnostic.jsonl --features hazard-v1 --weighting correction-v1 --epochs 30 --run-dir runs/bc-spell01-critical
```

Two duplicate student trajectories were removed. The remaining 21 episodes
contain 8305 samples: 7265 training samples and 1040 validation samples. The
diagnostic trajectory is in training. Epoch 25 was selected after 30 epochs;
validation accuracy is 94.23% and prioritized-sample accuracy is 83.33% (288
validation samples). This split differs from prior runs, so its accuracy is not
a direct improvement estimate over their percentages.

On replay of the recorded diagnostic training trajectory, disagreements dropped
from four to two. Decisions 156, 164, and 166 now match the teacher; decision 162
still differs, and a new disagreement appears at 155. This is an in-sample
regression check, not a simulation of the new closed-loop trajectory or proof
of recovery. It supports one new native check, not a PPO budget increase.

The trained weights survived a directory-publication PermissionError. They were
reloaded from the complete historical pending archive and republished using the
completion-marker implementation. cloning.jsonl retains the original error,
then records cloning_complete with recovered_from_pending and diagnostic_replay.
The completed checkpoint is runs/bc-spell01-critical/checkpoints/update-000000.

```powershell
.\.venv\Scripts\python.exe -m scripts.train --evaluate --resume runs/bc-spell01-critical/checkpoints/update-000000 --episodes 3 --run-dir runs/bc-spell01-critical-eval
```

Tests cover trajectory
deduplication, bounded tail weighting, actual weighted optimization, metadata
commit failures, rejection of incomplete checkpoints, and publication without
directory rename. Weights do not override actions during autonomous evaluation.

## Critical-weighted model native evaluation

runs/bc-spell01-critical-eval-001/training.jsonl recorded three failures, each
after 379 logic frames (6.32 seconds), with identical action counts:
[157, 0, 0, 9, 6, 2, 4, 10, 2]. Stay accounted for 82.63% of decisions. At
elapsed frame 360, all three periodic samples show (192.59, 415.56), with a
left action. No dropped bullets or radius fallbacks were logged. The user
observed avoidance of the first downward bullet followed by collision with the
second. The periodic log cannot identify the individual collision bullet or
resolve the intervening action sequence. The prior model's 348-frame diagnostic
is a different closed-loop trajectory and cannot substitute for this one.

Do not expand PPO training. Capture one complete student-only episode from the
unchanged critical model before another fit:

```powershell
.\.venv\Scripts\python.exe -m scripts.collect_corrections --resume runs/bc-spell01-critical/checkpoints/update-000000 --task spell01-character01-normal --teacher-probability 0 --episodes 1 --output runs/demos/critical-student-diagnostic.jsonl
```

The teacher only annotates states. Compare the final action sequence and teacher
labels, whether the actor returns to the bullet lane, and clearances for
successive visible bullets. hazard-v1 pools only the minimum clearance per
candidate over a 12-frame constant-motion prediction; it does not retain the
identity or arrival times of successive bullets. This is a possible
representation limitation, not an established cause of this failure.

The subsequent runs/demos/critical-student-diagnostic.jsonl reproduces the
379-frame failure, with 190 decisions, eight disagreements, no teacher execution,
and no dropped bullets or radius fallbacks. Decisions 163-168 move left from
roughly x=190.59 to x=171.76, then the actor returns toward the center. At
decisions 178, 180, and 182 it selects right while the teacher selects left.
During decisions 177-185, x alternates between 190.59 and 192.59 at y=415.56.
These observations support repeated reversals rather than a missing second
bullet observation; they do not identify the exact colliding bullet.

A second issue is visible in measured motion: right decisions 178/180/182/184
are followed by a 2 px left displacement, and the intervening left decisions
by a 2 px right displacement. Predictions assume immediate 2 px/frame motion.
An offline comparison on 36 common interior, nonterminal moving-context samples
gives squared displacement errors of 11.44, 6.07, and 18.96 px squared for
zero-, one-, and two-decision lag hypotheses. This suggests a control/observation
delay, but does not isolate SendInput latency, input polling, snapshot timing,
or time spent computing actions. Do not reverse the action mapping or claim
an exact fixed delay from this trace. Sustained left motion is correctly left.

The report is runs/demos/critical-student-analysis.json, generated with:

```powershell
.\.venv\Scripts\python.exe -m scripts.analyze_trace runs/demos/critical-student-diagnostic.jsonl --output runs/demos/critical-student-analysis.json
```

The analyzer refuses to overwrite reports. Prioritize measurement of observation
age, inference/input timing, and response to direction changes before another
weighted clone fit. Neither an arbitrary longer prediction horizon nor stronger
weights resolves the mismatch between predicted and observed switched motion.

## Measure input response before changing the controller

Start from the same spell's result menu, then focus the game:

```powershell
.\.venv\Scripts\python.exe -m scripts.probe_input --output runs/demos/input-response.jsonl
```

The probe runs 100 requested logic frames (about 1.7 seconds at 60 FPS), with
long horizontal holds, releases, and two-frame reversals. It targets one-frame
sampling and records actual frame deltas. No student or teacher is evaluated.
It stops and releases keys at the budget or on death/focus failure. The game
continues after truncation; return to the menu manually. Do not count survival
or truncation here as a model result, and do not use this trace for cloning.

After inspecting this isolated response, capture the unchanged critical model
again if needed with the existing student-only correction command. New captures
include student/teacher inference durations, total decision duration, age of the
previous returned snapshot when input starts, input-send duration, frame-wait
duration and snapshot read timings. Existing archives remain compatible. The
teacher is computed even at probability zero; its extra decision time is now
visible. Observation age also includes encoding and logging, so it must not be
added to decision duration as though they were independent stages.

Host timings and frame-response traces cannot establish the exact engine input
poll instant. Compare long holds with reversals and inspect skipped frames
before selecting any delay-compensation model. No action mapping, hazard horizon,
reward, or saved model was changed for this measurement.

## First isolated input-response result

runs/demos/input-response.jsonl completed 100 one-frame transitions, without
bullets or deaths, and stopped at its time limit. Long holds move exactly
2 px/frame horizontally in the requested direction. All eight left/right
reversals have zero displacement on their first interval and 2 px displacement
in the new direction on the following interval. Both releases stop within the
first sampled interval. The actor is not running during this test. This does
not support a universal one-decision action delay or an inverted mapping.
It also cannot establish why reversals in the model-driven trace sometimes
move opposite to the new request; policy/teacher/logging overhead differs.

The common-sample motion-fit errors are 0.50, 0.75, and 3.00 px squared for
zero-, one-, and two-decision hypotheses. Switch-specific observations matter
more than choosing a fixed delay solely from this aggregate fit. See the saved
report runs/demos/input-response-analysis.json.

The first timing implementation used time.monotonic, which on this installed
Python uses GetTickCount64 with 15.625 ms resolution. Many sub-frame durations
were consequently recorded as zero. They cannot establish negligible compute
or read cost. Session/frame-stepper, decision, and correction inference timings
now use time.perf_counter (QueryPerformanceCounter on this host, reported
resolution 0.1 microseconds). New capture headers include clock metadata. Fake
clock injection remains supported, and no motion/controller setting changes.

The isolated probe need not be repeated. Capture one model-driven episode with
the corrected timings to compare observation age with the frame interval and
separate inference, teacher, input-send, and read costs:

```powershell
.\.venv\Scripts\python.exe -m scripts.collect_corrections --resume runs/bc-spell01-critical/checkpoints/update-000000 --task spell01-character01-normal --teacher-probability 0 --episodes 1 --output runs/demos/critical-timing.jsonl
```

This still executes only student actions. The teacher only labels states.
Keep the checkpoint unchanged until native timing data supports a specific
control or prediction adjustment; a constant one-step delay is not justified.

## High-resolution model timing and control ablation

runs/demos/critical-timing.jsonl reproduces the 379-frame failure and eight
teacher disagreements with the unchanged actor. It contains 189 two-frame
transitions and one terminal one-frame transition; there are no frame overshoots,
dropped bullets, radius fallbacks, or teacher-executed actions. Median student
inference is 1.494 ms, teacher inference 0.068 ms, decision time 1.619 ms, and
snapshot-return-to-input-start age 1.960 ms (maximum 3.874 ms). Input-send median
is 0.010 ms. These data do not support decision computation consuming a whole
16.67 ms game frame. They do not rule out missing an input-poll boundary within
that frame. During decisions 177-185, the opposing movement pattern persists.

Median final-snapshot read time is 0.969 ms. Total polling-read time is 13.738 ms
within a 31.323 ms median post-input wait. Those reads overlap the wait; do not
add the read total as an independent extra delay. Observation age includes
decision computation and logging; those also must not be double-counted.
The saved offline report is runs/demos/critical-timing-analysis.json.

Test one explicit action-duration ablation before changing the network or
allocating further PPO interactions:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_control --resume runs/bc-spell01-critical/checkpoints/update-000000 --task spell01-character01-normal --hold-frames 4 --episodes 1 --output runs/control-hold4.jsonl
```

Start at the same spell-result menu. Inference and observation still run every
two requested frames, but a newly selected network action is held for at least
four elapsed frames before the next selection. Neutral actions are held too;
urgent changes can be delayed. Raw predictions and executed actions are logged
separately. There is no teacher, safety filter, training, or checkpoint write.
This changes the deployed controller and must be reported as network plus action
holding, even if it clears. The cloning loader rejects these traces. Hold=2
provides a control with no imposed change at the original two-frame cadence.
The fixed four-frame duration is an experiment, not a fitted latency estimate
or evidence that input timing is the sole cause. Check closed-loop movement and
failure before adopting any action-duration change; offline replay is insufficient.

## First four-frame commitment native clear

runs/control-hold4.jsonl records one verified spell_success after 952 logic
frames (15.87 seconds), 476 transitions, and zero deaths. The settled result
menu reports outcome_flag=1. No teacher ran; no model weights were updated.
No dropped bullets or radius fallbacks were logged. The controller retained
four-frame action commitment throughout this run, overriding 54 raw predictions
(11.34% of decisions). This is the first recorded clear by the critical network
plus explicit action holding, not a clear by the unmodified two-frame policy.

In the common first 378 logic frames, the earlier critical-timing trace has
23 executed action changes and nine direct left/right reversals. This run has
14 changes and zero direct left/right reversals. Over its full, longer episode
it has 83 executed changes and two direct left/right reversals, versus 101
changes and eight direct reversals in its raw network proposals. These counts
support reduced switching on the observed trajectory. Different visited states
and the absence of teacher-label computation prevent treating the comparison as
an isolated causal estimate. Holding can still delay urgent avoidance elsewhere.
The offline report is runs/control-hold4-analysis.json.

Keep the checkpoint and controller fixed and repeat five episodes before
adopting it or attempting further PPO training:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_control --resume runs/bc-spell01-critical/checkpoints/update-000000 --task spell01-character01-normal --hold-frames 4 --episodes 5 --output runs/control-hold4-repeat.jsonl
```

Repeated identical spell trajectories check reproducibility, not generalization
or independent success probability. A subsequent hold=2 run through this same
entrypoint can control for the removal of teacher computation. Neither this
single clear nor any follow-up controller-assisted clears justify loading these
logs as teacher demonstrations or resuming the old PPO collector unchanged.

## Five-episode commitment repeat

runs/control-hold4-repeat.jsonl completed all five episodes: four successes and
one classified failure, with no observed deaths or dying-state samples in any
episode. There were no dropped bullets or radius fallbacks. Episodes 0, 1, 2,
and 4 cleared at 1402, 984, 1071, and 1024 frames respectively (23.37, 16.40,
17.85, and 17.07 seconds). The empirical clear fraction is 4/5 for this fixed
task and network-plus-holding controller; it is not a general success estimate.

Episode 3 failed at 1680 frames (28 seconds), with terminal deaths=0 and a
settled failure result menu. The user observed a spell timeout. The last sampled
visible/collidable Boss had 309 HP at frame 1558, and the median sampled absolute
horizontal player/Boss separation during that episode was 149.67 px. These
observations support insufficient attack alignment, but the reader does not
provide a validated remaining spell timer or automatic timeout-cause label.
The summary's legacy death_or_retry_episodes=1 must not be read as one death:
that aggregate also counts a classified failure without death.

Offline teacher queries on the failed controller trajectory show a discrepancy
that holding alone does not address. In frames 902-1200 (150 decisions), the
teacher recommends right in 96 states and down-right in 43; executed actions
include left in 96 states, and raw network proposals include left in 92.
At frame 1400's pre-action state, the player is at x=8 and the teacher selects
right while both raw and committed actions select left. Holding did not invent
that left prediction. This suggests a network limitation in edge/alignment
states; teacher suggestions are heuristic labels, not counterfactual proof of
recovery. Preserve the current checkpoint as the successful avoidance baseline.
The offline report is runs/control-hold4-repeat-analysis.json.

Before changing labels, network, rewards or training cadence, run the same
evaluation entrypoint with hold=2 for three episodes, removing teacher-compute
and entrypoint differences from the control-duration comparison:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_control --resume runs/bc-spell01-critical/checkpoints/update-000000 --task spell01-character01-normal --hold-frames 2 --episodes 3 --output runs/control-hold2-baseline.jsonl
```

Subsequent improvement work should target edge recovery and attack alignment on
the visited states. Existing full traces can support offline teacher queries;
raw control-evaluation logs remain excluded from cloning to preserve provenance.
Do not increase the PPO budget or mix a held executed action with the raw
two-frame action log probability in the unchanged PPO collector.

## Two-frame same-entrypoint control

runs/control-hold2-baseline.jsonl completed three failures at 379, 747, and
379 logic frames (6.32, 12.45, 6.32 seconds), each with one observed death.
All 754 decisions executed the raw network action: zero commitment overrides,
no teacher computation, no dropped bullets or radius fallbacks. The first and
third episodes repeat the known trajectory with nine direct left/right
reversals each. The middle episode has eight such reversals and ends oscillating
near x=187.21-189.21, y=416.04. Compared with the four-frame controller's four
clears in five trials, this supports retaining four-frame holding as the current
working baseline, without inferring a population success rate or sole cause.
The offline report is runs/control-hold2-baseline-analysis.json.

The user reports repeated downward volleys from the Boss: prioritizing exact
horizontal alignment can place the player in the firing lane. Do not respond
to the earlier no-death failure by unconditionally increasing alignment reward.
On middle-episode decisions 361-369, offline reactive_attack queries match every
network action, yet alternate right and left. Best predicted clearance drops
from 24 to about 0.21 px while net motion remains small. Improving teacher-label
agreement alone cannot address all such failures. Those actions often maximize
the existing short-horizon clearance, so these observations do not isolate an
excessive alignment preference as the sole cause either.

Future work should consider safe firing positions, successive projectiles,
movement continuity and teacher limitations. These are design directions, not
implemented mechanisms. Preserve the successful held-controller checkpoint;
no weights, rewards, or runtime controller settings changed for this analysis.
