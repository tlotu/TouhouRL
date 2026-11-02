# Attack-band rule experiment

This is a separate rule controller, not a model update. Original reactive_attack,
hazard features, rewards and checkpoints remain unchanged. No HP feedback is
used. The existing cloning loader rejects these evaluation traces.

## Ranking

Reuse wall-valid, 12-frame linear clearance candidates. Require clearance >=
max(min_clearance, best_clearance - max_clearance_loss) before considering attack
position. If best_clearance < min_clearance, choose maximum clearance instead.
Clearance is a heuristic in pixels, not a calibrated survival probability.

Within the eligible set, minimize max(0, abs(target_dx - vx*4) - half_width),
plus the original close-target penalty. Break ties with the original reactive
score, then action ID. There is no extra center preference inside the band.
For qualified-target states, half_width=6 reproduces the old teacher's ranking.
Without a target, the new rule still applies its clearance gates.

The 24 px half-width is experimental, not measured shot coverage. It does not
infer character, equipment, homing capability or distance-dependent shot spread.

## Native comparison

Start each command at the same spell's result menu and keep focus. Both runs use
repeat=2, hold=4, capacity=128, min_clearance=4, max_clearance_loss=1. Run the
narrow-band control first, then the wider band:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_attack_zone --half-width 6 --hold-frames 4 --episodes 3 --task spell01-character01-normal --output runs/attack-zone-w6.jsonl
.\.venv\Scripts\python.exe -m scripts.evaluate_attack_zone --half-width 24 --hold-frames 4 --episodes 3 --task spell01-character01-normal --output runs/attack-zone-w24.jsonl
```

Launch each separately and focus the game again. Safety thresholds are explicit
CLI options; keep them fixed during the width comparison. Later tolerance sweeps
need separate logs. Neither run establishes a neural-policy improvement.

Trace records contain raw_rule_action, committed_action, rule_ranking (candidate
clearance, eligibility and zone excess), executed_candidate, and
executed_meets_clearance_gate. Ranking describes the current raw recommendation;
holding can execute an older direction that no longer passes the current gate
or wall prediction. The flag reports this without a new safety override.
Even an eligible path is not guaranteed collision-free.

Compare recognized outcomes, deaths, clear time, HP trajectory, offsets, edge
dwell and reversals. Lower alignment error alone is not success; survival without
adequate damage can still fail. HP remains diagnostic only.

## Offline evidence

Nine new tests cover the deadband, safe second-best selection, absolute and
relative gates, emergency fallback, walls, target absence, configuration, legacy
targeted ranking and dataset provenance. Recorded-state replay is saved in
runs/attack-zone-replay.json. On 3271 states, widths 6 and 24 disagree on 378
actions; the wider rule stays where the narrow rule moves on 181 states. Every
non-emergency selected candidate passes its gates. Width 6 matches the original
teacher on all 2970 replay states with a qualified target. These are offline
ranking checks, not simulation, held-action validation or native success.

## First native comparison

runs/attack-zone-w6.jsonl and runs/attack-zone-w24.jsonl each completed three
successes without deaths, dropped bullets or radius fallbacks. Frame counts were
1024/974/953 (width 6) and 1132/1040/1005 (width 24); mean time to result was
16.39 and 17.65 seconds. This supports feasibility, not superior success rate
or faster completion. These are rule-controller results, not neural learning.

In width-24 episode 0, the player moves from x=15.06 to x=279.55 over 72 frames,
while Boss x changes from 264.59 to 297.46: offset falls from 249.52 to 17.91 px.
However, that episode has 124 unchanged-action moving samples at about 4 px/frame,
whereas the predictor assumes 2. Other five episodes are mostly 2 px/frame.
The old held-model timeout episode also has such 4 px/frame samples. Fixed Shift
ownership does not verify the engine's actual movement mode. Do not attribute
the wider range/recovery entirely to attack width or infer a particular cause.

The comparison report, source hashes and per-episode speed histograms are saved
in runs/attack-zone-comparison.json. Verify movement-mode consistency before
adopting a new teacher or expanding tuning. HP remains diagnostic only.
