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
two-of-three zero-shot result. Native masked results are still pending.

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
