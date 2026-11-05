# Frozen actor with a clearance gate

`scripts.evaluate_control --boundary-mask --risk-filter` enables an opt-in,
evaluation-only controller, `linear-clearance-gate-v1`. No weights or rewards
change. Ordinary boundary evaluation and PPO training keep their existing
behavior. The new collection is `risk_control_evaluation`, excluded from
teacher demonstration loaders.

## Selection

At each observation, forecast minimum bullet clearance for all nine directions
over 12 frames, using the existing 2 px/frame linear-motion geometry and retained
bullets. Keep physical feasibility separate: directions must fit the actual
four-frame hold endpoint, not a twelve-frame boundary endpoint.

Among physically feasible directions, let best be the largest predicted gap.
If best >= 4 px, retain gaps >= max(4, best - 1); choose the highest actor score
within that set. Otherwise retain the maximum-clearance directions and choose
by actor score. Ties use the smaller action index. Clearance is a geometric
estimate, not a survival probability.

Only fresh choices apply the gate. Existing holds retain their action unless
the physical remaining-hold check interrupts them. Bullet-risk changes alone
do not interrupt a hold. The predictor still ignores lifecycle state/timers,
future spawns, acceleration, omitted bullets and exact input latency.

## Provenance

- `raw_student_action`: unfiltered actor argmax.
- `boundary_student_action`: argmax after the physical endpoint mask.
- `masked_student_action` / `risk_filtered_action`: combined-filter proposal.
- `risk_allowed_actions`, `risk_clearances`, `risk_threshold`,
  `risk_best_clearance`, `risk_candidate_mode`: geometric selection evidence.
- `risk_mask_overridden`: physical proposal differs from the risk proposal.
- Existing commitment fields identify actual held execution.

Summary `risk_changed_proposals` counts observations; `risk_applied_fresh_choices`
counts changed fresh choices. Boundary counts retain their physical meaning.
Replay these files with `RiskBoundaryCommitment`, not the boundary-only class.
An executed held action may differ from the current risk proposal. This is
explicit controller assistance, not newly learned actor behavior.

Do not pass filtered evaluation actions into existing PPO updates under old
sampling likelihoods. Training with a risk mask would need a separate consistent
policy distribution and contract; the explicit implementation is now documented
in [RISK_PPO.md](RISK_PPO.md).

## Evidence and first native experiment

Pre-continuation update 1 also fails one of three current spiral trials, so the
later 1k continuation alone cannot explain the observed risk. In its failed
episode, at elapsed frame 932 the actor selects down with a predicted gap of
0.6005 px while stay has 19.8219 px. The gate selects stay on that recorded state.
At frames 936/940 it also excludes negative-gap downward proposals.

`runs/spiral-risk-filter-preflight.json` checks 847 fresh source observations
and 982 fresh update-3 observations, changing 103/183 proposals respectively.
These are instantaneous fixed-observation checks, not alternate native runs.
Median filter-only time was about 0.52 ms on this host; added runtime work may
affect real-time trajectories. Model hashes remain unchanged.

The full offline suite passes 241 tests, including geometry/feature agreement,
emergency fallback, four-frame physical bounds, held-action preservation,
provenance, CLI guards, fake-native execution, archive preservation and teacher
loader exclusion. The first native result is recorded below.

On the current verified spiral spell, keep character/equipment/difficulty fixed
and start from its result menu. Freeze the same update 3 that previously had
two successes and one death; add only the explicit risk controller:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_control --resume runs/ppo-masked-zone24-1k/checkpoints/update-000003 --boundary-mask --risk-filter --hold-frames 4 --episodes 3 --task stage06-spiral-delayed-character01-normal --allow-task-transfer --transfer-reason "Frozen update 3 with explicit geometric risk candidate filter" --output runs/ppo-masked-zone24-1k-spiral-risk-eval.jsonl
```

Review all outcomes, interventions and actual timing before any training or
additional acceptance runs. Keep the original candidate failures and BC reference.

## First native result and trained-task regressions

`runs/ppo-masked-zone24-1k-spiral-risk-eval.jsonl` completes three successes,
zero deaths at 1114/1948/1498 frames (mean 25.33 seconds to result). The model
hash and spiral reset signature match the reference. All 2280 transitions replay
with the risk controller, including every clearance/candidate/holding field.
Fresh risk-set violations and modeled execution violations are zero.

There are 1141 fresh choices and 123 applied risk replacements (25/49/49).
Boundary replacements occur at 89 fresh choices; these counts may overlap.
Only two fresh choices use emergency mode. During retained holds, 98 executed
actions fall outside the newly computed risk set, as explicitly allowed by
this version. Risk changes alone still do not interrupt holding.

All steps advance two frames, with zero race retries, radius fallbacks or hold
interruptions. Peak bullets=323, maximum omitted=195. Episode 1 reaches x=375.72
and has 62 direct left/right reversals; episode 2 reaches x=376. Survival here
does not establish efficient edge escape. Three unpaired successes versus the
earlier same-model boundary-only two-of-three result are preliminary evidence
for the combined controller, not a PPO learning gain or reliability proof.

Evidence is in `runs/ppo-masked-zone24-1k-spiral-risk-eval-analysis.json` and
`runs/ppo-masked-zone24-1k/risk-control.json`. Keep original-controller results
and this separate collection distinct. Continue holding PPO training. Next
check both demonstration tasks, keeping the same archive, flags, character,
equipment and difficulty. Manually select stage 1 and start at its result menu:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_control --resume runs/ppo-masked-zone24-1k/checkpoints/update-000003 --boundary-mask --risk-filter --hold-frames 4 --episodes 3 --task spell01-character01-normal --output runs/ppo-masked-zone24-1k-stage01-risk-eval.jsonl
```

After switching to the dense straight stage-6 spell and its result menu:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_control --resume runs/ppo-masked-zone24-1k/checkpoints/update-000003 --boundary-mask --risk-filter --hold-frames 4 --episodes 3 --task stage06-straight-dense-character01-normal --output runs/ppo-masked-zone24-1k-stage06-risk-eval.jsonl
```

Review those per-task outcomes before further experiments or implementing a
risk-masked training distribution. External filtered traces are not PPO rollout data; use the explicit risk-policy entry for training.

## Initial compatibility milestone completed

Both trained-task regressions succeed three of three with zero deaths:

| Task | Frames to result | Mean seconds | Applied fresh risk choices |
| --- | --- | --- | --- |
| Stage 1 | 939/980/1056 | 16.53 | 69 |
| Dense straight stage 6 | 700/717/737 | 11.97 | 38 |
| Previously checked spiral | 1114/1948/1498 | 25.33 | 123 |

All 2566 new regression transitions replay correctly, including risk fields.
There are no modeled execution or fresh risk-set violations, acquisition race
retries, radius fallbacks or incomplete episodes. The dense spell has one
physical hold interruption: episode 2 step 231, elapsed frame 462, y=429.556;
continuing down-left for two frames would cross y=432, so a fresh choice selects
up-right. This is a boundary interruption, not a risk-triggered interruption.

Per-file analyses, `runs/ppo-masked-zone24-1k-risk-regression-analysis.json` and
`runs/ppo-masked-zone24-1k/recommended-risk-control.json` preserve the separate
task results, hashes, contracts and pending limitations. CLI defaults and model
weights are unchanged. All three tasks have only three unpaired trials, and
the spiral edge oscillation remains; no universal reliability claim is made.

This completes the initial combined-controller compatibility milestone. Stop
repeating the same acceptance setup. The next implementation stage is PPO with
the same risk-filtered sampling, stored likelihoods and action reevaluation,
plus explicit checkpoint migration/version guards and duration-aware collection.
That training path is now implemented in [RISK_PPO.md](RISK_PPO.md), with 247
passing tests and an offline-prepared source. Native risk training is pending.
Plain `train_masked` still uses only the physical mask; risk training explicitly
requires `--risk-filter` and its separate contract.
