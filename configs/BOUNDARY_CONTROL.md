# Frozen actor boundary-mask experiment

The spiral rule teacher completed three successes at 1589/1253/1142 frames,
zero deaths, mean 22.13 seconds to result. One episode reached x=359.25 and
finished successfully. All three passed strict zone validation, giving 998 fresh
labels, but none has been fitted into the frozen joint actor. Peak bullet count
was 297, with up to 169 omitted; radius fallbacks were zero. Source hashes and
telemetry details are in runs/stage06-spiral-zone24-teacher-check-analysis.json.
Three unpaired successes are preliminary teacher evidence.

The original joint actor's third spiral episode repeatedly selected outward
commands at x=376. Its geometry predicts nominal motion while the game clamps
the player at the wall. A validity input is not an enforced constraint. Changing
the reward cannot change a frozen actor during evaluation, and arbitrary large
negative feature values would change its input distribution without guaranteeing
rejection. The explicit experiment excludes infeasible choices at selection.

## Run

Keep the same spiral spell, character, equipment and difficulty. Start at its
result menu and retain the same checkpoint, half-width, capacity and frame limit:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_control --resume runs/bc-zone24-joint/checkpoints/update-000000 --boundary-mask --hold-frames 4 --episodes 3 --task stage06-spiral-delayed-character01-normal --allow-task-transfer --transfer-reason "Spiral task; frozen actor with explicit hold-endpoint boundary mask" --output runs/bc-zone24-joint-stage06-spiral-boundary-eval.jsonl
```

This is a frozen actor plus a declared controller constraint, not the original
unmodified network controller or a new trained policy. Default evaluations remain
unchanged without --boundary-mask. Do not pool this experiment with the original
two-of-three zero-shot result. The first native masked result is recorded below.

## Selection contract

hold-endpoint-mask-v1 uses the existing geometry bounds x=[8,376], y=[16,432],
nominal slow speed 2 px/frame and diagonal components sqrt(2) px/frame. At a fresh
selection, reject a direction whose nominal endpoint crosses these bounds within
the saved hold duration (four frames here). Choose the highest original actor
logit among the allowed directions, equivalent to setting excluded selection
scores to negative infinity. Logged logits remain finite original actor scores.
Ties use the lowest action ID. Staying and tangential motion remain available
at a wall. Unsupported player positions and invalid scores fail explicitly.

The horizon is the execution hold, not the teacher's 12-frame forecast. The
mask does not reject every direction near a wall. At every substep it rechecks
the held direction over the remaining hold, with a minimum of two frames for
the next requested substep. If it would cross, interrupt the hold, select again
with a full hold horizon, and record the interruption. Otherwise retain the
existing commitment. Measured frame overshoot or unexpected motion can still
depart from this model; actual frame deltas and displacement remain diagnostic.

No teacher, bullet-clearance gate, attack-band preference, centering force or
reward shaping is added. Existing bullet/attack inputs and the actor choose
between feasible directions. This does not repair the 12-frame linear forecast,
predict delayed launches, or guarantee earlier retreat or survival. A direction
that stays in bounds for four frames can still cross the boundary later in the
12-frame forecast; testing a different feature model would be another experiment.

## Evidence and action provenance

The header uses policy=ppo_boundary_mask_action_commitment and
collection=boundary_control_evaluation. It records the boundary contract, bounds,
speed, horizon and permission to interrupt holding. Source model hash and original
training/evaluation task identities are retained. A sibling console log is saved.

Every transition records:

- raw_student_action: original actor argmax before any controller choice.
- masked_student_action: actor argmax among full-hold feasible directions.
- committed_action/action: executed choice after remaining-hold checks.
- boundary_allowed_actions and actor_logits: full-hold selection evidence.
- boundary_execution_horizon_frames and boundary_execution_allowed_actions:
  actual remaining-hold check.
- boundary_mask_overridden: raw and masked proposals differ.
- commitment_overridden: masked proposal differs from the held action.
- action_overridden: original actor argmax differs from the executed action.
- boundary_hold_interrupted: boundary recheck forced a new selection.

The summary separates all three override counts and interruption count. A held
action can remain feasible for two frames even if it is absent from the new
four-frame proposal mask; the execution fields distinguish that case. These
logs are rejected as ordinary teacher demonstrations.

Snapshot auditing on the original trajectories changes 81/0/56 raw argmax
proposals, including 40/0/30 original fresh selections. At logged end frames
802/842/854/858/862 the snapshot masked actions become left/stay/stay/down-left/
down-left. At 590/602 right remains executable, so it is unchanged. This is not
an alternate rollout and does not establish that the masked policy survives.
See runs/bc-zone24-joint/boundary-mask-preflight.json. Source model SHA256 remains
a7a0123cb7b5800121a3e5bd0f4f15c4d851dbf23540bfc5e4f78c17fd181964.

Eight new tests cover walls/corners, tangential movement, four-versus-twelve-frame
horizons, raw/masked/executed provenance, safe remaining holds, interruption on
overshoot, resets, unchanged interior behavior, invalid input, and the fake CLI
with source preservation and teacher-log rejection. The full suite passed 217
tests. No native input was sent during implementation or offline verification.

Training with a mask would require applying the same mask in the sampled PPO
distribution and its log probabilities. External action replacement with an
unmasked likelihood is not supported; this option only exists in evaluation.

## Initial native masked result

runs/bc-zone24-joint-stage06-spiral-boundary-eval.jsonl completed three successes,
zero deaths and two retries, at 1137/1232/1129 frames. Mean time to result was
19.43 seconds. Source archive hash still matches the original joint reference.

| Episode | Direct left/right reversals | Changed mask proposals | Applied fresh mask choices | Hold interruptions |
| --- | --- | --- | --- | --- |
| 0 | 3 | 5 | 3 | 0 |
| 1 | 2 | 0 | 0 | 0 |
| 2 | 3 | 0 | 0 | 0 |

All three applied changes were near the lower boundary at y=429.37, logged end
frames 614/618/622: down-left was excluded and stay selected. The other two
changed proposals occurred inside an existing hold. Summary totals distinguish
five mask proposals, 173 commitment overrides and 176 final/raw differences;
these are different, potentially overlapping counts. No right-wall intervention
occurred. Episode 2 briefly reached x=328.85, but no active sample reached x>=368.

Replaying all 1750 logged transitions with their saved actor scores exactly
reproduced selected actions, mask sets, hold ages, interruption flags and summary
counts. There were no executed-horizon constraint violations. All observed frame
intervals were 1 or 2; all 231 steady interior motion samples were about
2 px/frame. Peak bullet count was 278, omitting up to 150 from observations;
radius fallbacks were zero. This entry does not collect internal slow telemetry.

The batch supports a successful initial composite-controller run. Two episodes
never required masking, and the old right-corner failure was not reproduced, so
the three unpaired trials do not establish that masking caused the full apparent
improvement or fixed right-edge escape. See
runs/bc-zone24-joint-stage06-spiral-boundary-eval-analysis.json and the separate
boundary_evaluations entry in the joint baseline manifest. Original actor and
teacher results remain separate; no source weights or defaults changed.

Repeat ten episodes with the same spell, character, equipment and difficulty,
starting at the result menu:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_control --resume runs/bc-zone24-joint/checkpoints/update-000000 --boundary-mask --hold-frames 4 --episodes 10 --task stage06-spiral-delayed-character01-normal --allow-task-transfer --transfer-reason "Spiral task; frozen actor with explicit hold-endpoint boundary mask" --output runs/bc-zone24-joint-stage06-spiral-boundary-repeat.jsonl
```

Inspect applied interventions and right-edge coverage, not just the success
count. Natural repeated runs may still miss the old trap; deliberate edge-state
testing remains a separate possible experiment. Ten-episode results, hold
interruption under native overshoot and masked regressions on the two trained
spells remain pending. No fitting or PPO continuation is started by this run.

The repeated run later completed seven successes and aborted in episode 7 on
frame-crossing acquisition, with no verified eighth outcome. Episodes 5/6 had
28/59 applied right-boundary replacements and succeeded. Preserve that partial
evidence and finish three more episodes after the bounded retry repair; see
[STATE_ACQUISITION.md](STATE_ACQUISITION.md). The earlier ten-episode command is
the historical requested run, not a statement that ten episodes completed.

## Completion episodes and survival objective

The actual completion log is
runs/bc-zone24-joint-stage06-spiral-boundary-repeat-remaining-001.jsonl. It used
same-frame-bounded-retry-v1 and completed three successes at 2440/1125/1115
frames, zero deaths, without another acquisition error. All frame-race retry
counts were zero, so the native recovery branch itself was not exercised.

In episode 0, the player reached y=74.24, spent 856 active frames in the upper
half, and spent 152 frames within 16 px of the boss's height. Boss HP continued
decreasing; the last collidable sample had 1 HP at frame 2318, followed by native
success. The user observed homing shots completing the fight; projectile type
is not directly recorded. There were 30 applied right-boundary mask choices,
with no execution-horizon violation. Episodes 1/2 used lower routes without
mask interventions. All 2341 transitions replayed with matching provenance.

The planned repeat now has seven plus three complete successes, preserving the
earlier interrupted episode separately. Including initial acceptance, the masked
controller has 13 complete spiral successes and one unknown interrupted outcome.
No weights changed. Source hashes and details are in
runs/bc-zone24-joint-stage06-spiral-boundary-repeat-remaining-001-analysis.json.

Survival remains the primary objective. Do not add an absolute upper-screen
penalty or forbid a successful route merely because it differs from the teacher
or takes longer. Physical boundary feasibility is a hard constraint; tactical
height, alignment and damage efficiency remain task-dependent. Being at the
boss's height is not itself a collision measurement. Some tasks may require
upper routes or circling, and homing equipment changes the alignment tradeoff.

Keep normal-stage progression, boss defeat/card capture and spell-practice
result semantics distinct. The user's description of timeout allowing ordinary
stage progression is not a validated full-stage outcome detector in this repo.
Existing native practice failures and time-limit truncations are not relabeled
globally. No reward profile, upper-area restriction or terminal classifier was
changed in this analysis.

Spiral repeated acceptance can now stop. Before making this optional controller
a general default, run three regressions on each trained spell. Select each
spell manually with the same character/equipment/difficulty and start from its
result menu. Both tasks are seen, so no transfer flag is required:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_control --resume runs/bc-zone24-joint/checkpoints/update-000000 --boundary-mask --hold-frames 4 --episodes 3 --task spell01-character01-normal --output runs/bc-zone24-joint-stage01-boundary-eval.jsonl
```

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_control --resume runs/bc-zone24-joint/checkpoints/update-000000 --boundary-mask --hold-frames 4 --episodes 3 --task stage06-straight-dense-character01-normal --output runs/bc-zone24-joint-stage06-straight-boundary-eval.jsonl
```

Results of these regressions follow. Deliberate edge recovery, lifecycle
tracking and PPO mask integration remain pending. The latest offline suite
remains 227 passing tests; this stage changed analysis and documentation only.

## Regression-checked reference configuration

Both trained-task regressions completed successfully with the same frozen actor,
hold-endpoint-mask-v1 and same-frame-bounded-retry-v1:

| Task | Native result | Frames to result | Mean seconds |
| --- | --- | --- | --- |
| Stage 1 | 3 successes, zero deaths | 946/1007/1011 | 16.47 |
| Dense straight stage 6 | 3 successes, zero deaths | 723/700/766 | 12.16 |

The actual straight-spell log ends in boundary-eval-001.jsonl. Both runs had
zero mask proposals changed, zero hold interruptions, zero frame-race retries,
zero incomplete episodes and no errors. All 2578 transitions replayed correctly,
with no modeled execution-horizon violations. These runs check compatibility
on normal routes; they do not add rare edge-intervention coverage.

Frame intervals were 1 or 2. Stage 1 had 172 steady interior speed samples at
2 px/frame. The straight spell had 164 at 2 and one each at 1/3 in consecutive
two-frame leftward intervals (steps 180/181, episode 0). Their combined speed
was 2 px/frame over four frames; this is not evidence of sustained high speed.
No internal slow flags were collected. Peak bullet counts were 69/240, with
0/112 omitted and no radius fallbacks. The race-recovery branch remains without
a native triggered example despite clean runs of the updated acquisition path.

Reports and source hashes are in
runs/bc-zone24-joint-boundary-regression-analysis.json. The explicit recommended
reference configuration is runs/bc-zone24-joint/recommended-control.json; it
records source hashes, model hash, both control contracts, geometry and task
counts. baseline.json points to it and retains every raw, teacher and masked
batch separately. This recommends an experimental reference, not a universal
success guarantee. CLI masking remains opt-in, with no changed weights/defaults.

Current masked complete results are 3 successes on stage 1, 3 on the dense
straight spell and 13 on the spiral spell, with the older spiral infrastructure
interruption kept separately. Stop repeating these same acceptance runs.
Preserve survival-first behavior and freedom to use the upper screen.

The next implementation stage is mask-consistent PPO: sample from feasible
actions and use that same normalized distribution for stored log probabilities
and optimizer updates. Preserve the four-frame/duration-aware contract and
source checkpoints. The existing joint actor is not supported by train_held.
The separate implementation is now documented in [MASKED_PPO.md](MASKED_PPO.md):
234 offline tests pass and a source-preserving policy conversion is prepared.
Native masked training remains pending. The initial entry selects one seen
task per run; automatic multi-task scheduling remains pending. Per-task
outcomes and ordinary-stage progression semantics remain explicit. Do not
append an external mask to old sampled actions while reusing their likelihoods.

## First native remaining-hold interruption

`runs/ppo-masked-zone24-1k-spiral-eval.jsonl`, episode 2 step 517, provides
the first verified native interruption example. At elapsed frame 1034,
x=373.598/y=414, the held up-right action is two frames old. Its remaining
two-frame endpoint would cross x=376, so the controller starts a fresh
four-frame choice and selects left. All action/mask/age fields replay exactly;
no modeled execution violation occurs. No three-frame step or acquisition race
is needed to trigger this check: nominal motion and sampled native displacement
can differ. This branch is now covered by native evidence, while actual
frame-race recovery remains unobserved in these runs.

The episode later fails with a recorded death at frame 1046. Interruption
correctness does not establish collision avoidance or that the interruption
caused the death. See `runs/ppo-masked-zone24-1k-spiral-eval-analysis.json` for
the full transition evidence and fixed-observation policy comparisons.
