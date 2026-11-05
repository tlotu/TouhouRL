# Static-laser PPO smoke

The user reports warning lines do not hurt, bright lasers do, and the reconstructed
width approximately matches gameplay. This supports the interpretation for the
manually selected stage-6 young demon lord spell; exact native collision width,
endcaps and all activation boundaries are not independently decoded.

The 0.02-second capture contains 2209 samples, 888 combat pools, no exhausted
races, peak 48 candidate lasers and 160 round bullets. All 888 raw pools redecode
to their recorded rows. Three death-counter increases are captured; in two, the
last playable sample overlaps a candidate bright laser under the hypothesis.
The collision instant is unavailable after the pool stops being sampled; bullets
coexist, so the cause cannot be established from those samples alone.

## Isolated training contract

`scripts.train_laser` only accepts `stage06-young-demon-lord-character01-lunatic`.
The spell/difficulty are manually selected, not decoded. It accepts the intrinsic
risk PPO checkpoint or its own laser checkpoints. Laser checkpoints use
`laser-static-risk-held-smdp-v1`, `touhourl-static-laser-state-v1` and
`static-laser-rows-f32-v1`; old training/evaluation paths stay separate.

`LaserReader` brackets the ordinary snapshot and candidate pool with the same
byte-frame check. `LaserSession` appends 64 laser rows and releases keys on
encoding failure. Unknown phases, nonfinite data, overflow and reported nonzero
axial speed are rejected. Observed Young Demon Lord waves were static; speed=0
alone cannot detect rotating/translating axes. Do not generalize this contract
to other laser spells without verification.

The policy receives existing 62 features plus nine direction-dependent 12-frame
laser clearance features. Combined candidates use the minimum of bullet
clearance and static-laser capsule clearance, subject to physical bounds. Warning
timers may activate inside the horizon; phase 1 is considered dangerous and phase
2 inactive under the reference interpretation. Exact fade damage, capsule
endcaps, width, latency, future spawns and enemy-body collisions remain limits.
The policy samples from this masked distribution and reevaluates that same
distribution for PPO likelihoods. Four-frame holding and physical interruption
remain; risk changes alone do not interrupt a hold.

The external `--attack-return` evaluation ranking is not used during PPO
training. Each `held_decision` records the sampled action, probability,
combined/bullet/laser gaps, candidate set, float32 player and laser rows; full
bullet observations are not recorded in training JSONL. The reward remains
`spell-v3-masked-survival`; updates occur after complete episodes at a verified
result menu. Source identity and old checkpoints are preserved.

## Actual source prepared offline

`runs/ppo-laser-young-preflight/checkpoints/update-000003` is prepared from the
frozen risk update 3, not trained on this spell. Source model SHA256:
`d57643c4d2a1e1c88693d6a058ffe8d9cf3930a26a3157d0113c0dcef5415a4e`.
Prepared archive SHA256:
`a40c92d6bf01c2b1a22a9997af368e96c1067140f374c0588b1937ce6bd44f2c`.
Source weights and all 12 Adam states copy exactly; nine additional feature
input weights and their Adam moments start at zero. Prior counters remain 1389
decisions / three published updates; no laser gradient update has occurred. A
separate resume-check archive has identical model and Adam tensors.

`offline-review.json` records 36 calls across 12 saved states with 48 candidate
lasers: p50/p95/max 8.184/11.140/11.307 ms per action offline, excluding native
acquisition and real-time scheduling. Another 47 historical no-laser states
preserve prior risk candidates and logits (max error 9.54e-7). Source unchanged.

Synthetic tests cover geometry, frame-coherent wrapping, observation shape,
sampling/likelihood agreement, actual fake-environment PPO gradients, Adam
migration/reload, task guards and complete-episode/menu-boundary training.
The full suite passes 280 tests; `runs/laser-training-tests.console.log` saves
output. None is a native win claim.

## First native smoke

Select this spell on Lunatic with the existing character/equipment, and start
at its result menu. The entrypoint sends game controls, retries complete
episodes and waits for focus. The 128 additional fresh-decision budget completes
the current episode before a batch update, so actual decisions may exceed 128.
Training is stochastic and may fail episodes.

```powershell
.\.venv\Scripts\python.exe -m scripts.train_laser --resume runs/ppo-laser-young-preflight/checkpoints/update-000003 --task stage06-young-demon-lord-character01-lunatic --decisions 128 --rollout-decisions 128 --run-dir runs/ppo-laser-young-smoke
```

Submit the printed `training.jsonl` and `training.console.log` paths. If a
read/frame/focus error or unsupported laser state stops collection, preserve
the actual logs and resume only a complete checkpoint. Inspect contexts,
durations, deaths, acquisition timings, optimization and the resulting checkpoint
before more training or independent evaluation. Native training remains pending.

## First native smoke reviewed

`runs/ppo-laser-young-smoke` collected two complete episodes before its only
PPO update: episode 1 failed with one recorded death at 368 frames/92 fresh
decisions; episode 2 succeeded at 1161 frames/291 fresh decisions. The user
observed the first player trapped between lasers and hit by a round bullet.
Training JSONL does not include full bullet trajectories or collision instants,
so that cause is retained as user observation. The second episode's 35 movement
choices in states with warning-phase lasers support the observed movement, but
do not establish a learned response: the gate explicitly anticipates warning
activation, and all samples preceded optimization.

For example, on episode 2's decision 74, 16 warning lasers were 10 frames from
the reference activation threshold. The combined gate offered only left; the
sampled action was left for four frames. This is observable gating behavior, not
proof the network learned to anticipate independently.

All 383 saved decision contexts reconstruct their physical bounds, laser
forecast from float32 rows/player, combined gaps, gate threshold and candidates;
every sampled action belongs. Bullet gaps are logged but full bullet geometry is
not; a full original-observation or old-log-probability replay is unavailable.
382 held actions lasted four frames, one terminal action one frame; five total
emergency decisions. The complete-episode collector overshot the 128 budget to
383 decisions and then made exactly one update: cumulative 1772, update 4.
Sampling took 29.224 s and optimization 4.328 s; terminal-only timing snapshots
do not establish per-step p95 or absence of all acquisition retries.

Source and published update-3 model/Adam tensors match exactly. Final update 4
has 12 finite Adam states at step 58 (source step 46) and changed policy/value
parameters, including new laser-input weights. Approximate KL=6.86e-7, clip
fraction=0, explained variance=-0.132; critic reliability remains unestablished.
Final model SHA256:
`49218dac301fa7127cb89cc9151fcd910f8bed1d8693690f9329f56e2e7d16be`.
Detailed analysis is `runs/ppo-laser-young-smoke/offline-review.json`.

## Independent evaluation

`scripts.evaluate_laser` now evaluates frozen laser checkpoints with the same
combined gate, float32 observation and physical held-action interruption. It
does not apply the external attack-return ranking or update weights. The trace
records full player, bullet and laser observations plus candidates, proposals,
actual actions and timing. The original training episodes are not post-update
results. Two new tests cover deterministic-policy agreement and fake-native
frozen evaluation; the full suite passes 282 tests, console output in
`runs/laser-evaluation-tests.console.log`.

On 24 saved manual-probe observations, actual update 4's deterministic actions
match the evaluation controller; model archive remains unchanged. This fixed-
observation check does not predict native outcomes, and those states are not
the missing full observations from the training collector.

Select the same young demon lord spell, character/equipment and Lunatic
difficulty; begin at its result menu and focus the game:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_laser --resume runs/ppo-laser-young-smoke/checkpoints/update-000004 --task stage06-young-demon-lord-character01-lunatic --episodes 3 --output runs/ppo-laser-young-smoke-eval.jsonl
```

Review the actual JSONL and console before more training. Evaluation is
deterministic while the training episodes used stochastic sampling. Manual
task selection and the observed laser/collision limitations remain in force.

## First post-update evaluation and candidate-slack experiment

Frozen update 4 finishes two successful episodes at 1291/1078 frames and one
failure with one death at 393 frames. The user reports the third player had
space but moved up-left into a round bullet; its screenshot is not synchronized
with a particular trace frame. All 1382 transition replays and loaded-policy
candidate/mode checks pass. Source update 3 differs in masked argmax on three
of these fixed observations, all from episode 1; no recorded episode-3 proposal
differs. This is not a native counterfactual or proof of equivalent outcomes.

At episode 3 frame 352, down and right have estimated joint clearances of
21.414 and 17.59 px. With the existing 1 px slack, threshold=20.414 and down is
the only candidate, despite right exceeding the 4 px minimum; the raw actor's
top direction is right. The resulting recorded trajectory reaches y=431.8.
At frame 384, the modeled down endpoint is outside y=432. All physically
allowed directions have negative predicted clearance; the emergency rule selects
up-left (-1.36 px) over right (-7.44 px). At frame 386, the current candidate
changes to stay, but up-left remains held; its newly estimated clearance is still
positive 7.30 px. At frame 392, up-left is chosen again, right has estimated
5.24 px but is below the near-best 11.81 threshold; death is observed by frame
393. The screenshot could correspond to either up-left decision. The precise
collision instant, action causality and alternative native outcomes are unknown.
The nearest observed bullet at frame 384 has radius 8, relative position
approximately (+6.14,-7.40) px; slots have no persistent identity.

Decision p50/p95/max are 3.418/7.722/8.693 ms, observation age
4.461/8.957/17.597 ms. Maximum 32 bullets are omitted elsewhere in the run;
at the frame-384 sample, 53 bullets are retained and none are omitted. The
geometry is a linear 12-frame estimate with approximate radii and a 4-frame
hold, not a survival probability.

An explicitly opt-in `--candidate-slack 4` evaluation experiment widens the
normal-safe band from best-minus-1 to best-minus-4 pixels, preserving the 4 px
minimum, emergency max-gap rule and physical boundary. It reuses the frozen
actor logits but is an **external evaluation gate**, not the policy's PPO
sampling distribution. Its header and collection explicitly identify that fact;
training and default evaluation remain at 1 px. It still holds actions for four
frames; risk changes do not interrupt the hold.

On the 692 fixed recorded fresh states, the widened gate changes proposals in
31/23/7 states by episode. At frames 344, 348 and 352 in the failure, it admits
the actor's rightward proposal; at the frame-384 emergency it still selects
up-left. These are offline same-state comparisons, not a re-simulated path or
proof that the alternate gate would have survived.

Two added tests cover safe-gate widening, physical boundary and emergency
precedence, invalid slack, and fake-native provenance. The full suite passes
284 tests (`runs/laser-slack-tests.console.log`). Analysis reports are adjacent
to the original trace with `-analysis.json` and `-slack4-fixed-states.json`.

For a bounded comparison, keep the same spell, difficulty, character/equipment,
checkpoint and result-menu start. Only add the external candidate slack:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_laser --resume runs/ppo-laser-young-smoke/checkpoints/update-000004 --task stage06-young-demon-lord-character01-lunatic --episodes 3 --candidate-slack 4 --output runs/ppo-laser-young-smoke-slack4-eval.jsonl
```

Inspect all outcomes, including deaths, before deciding whether the weaker
near-best constraint is helpful. No native alternate evaluation or new PPO
training was performed during this offline review.

## Slack-4 native review: room selection remains unresolved

The external slack-4 run finishes failure/success/success at 363/1063/1391
frames, one recorded death. The user reports episode 1 trapped in a tiny laser
compartment, and episode 3 surviving another narrow compartment by luck.
All 1410 transitions replay exactly under the recorded external gate; loaded
actor modes match that gate. Intrinsic policy masks are a subset of these
external candidates, not the same PPO sampling distribution. Fresh/emergency
decisions are 91/10, 266/1, 348/21. The prior slack-1 run also finished 2/3;
small unpaired runs do not establish equal reliability or improvement.

Episode 1 stays still at frame 260 with 42 frames before earliest warning
activation; the 12-frame gate reports all directions at capped clearance 24.
At frame 324 its best joint estimate is already 1.6 px with stay as the sole
candidate. The player remains at approximately (205.7,376.3) through repeated
emergency choices as all 48 beams activate. Slack changes do not alter the
emergency fallback. The problem includes selection of future compartments,
not just immediate directional clearance or proximity to the screen boundary.

`agent/laser_room.py` adds an offline grid diagnostic, not a live planner. It
treats all visible warning/firing static laser capsules as simultaneously closed
walls and measures a four-neighbor component containing the nearest player grid
cell. It ignores bullets, activation deadlines, future births and whether a
destination can actually be reached before closure. Invalid/moving candidates
are rejected. Blocked grid seeds are explicitly marked, not called zero true
area; grid resolution and approximate width can change topology.

At selected states, grids of 4/2 px estimate the first episode's tiny component
at 224/228 px squared, versus 7520/7524 in episode 2. Episode 3 includes blocked
seed estimates near the right-bottom edge despite native survival; this is
evidence against blindly using that approximate grid as a hard live constraint.
Reports are `*-analysis.json` and `*-rooms.json` beside the slack-4 trace.

The full suite passes 286 tests; the two new grid tests additionally pass after
invalid-geometry hardening. Keep both experimental results and frozen weights.
End scalar slack tuning and pause PPO budget increases. The next design task
is warning-period destination selection with timed reachability and local bullet
avoidance, validated offline before any live control experiment. No such live
planner is implemented or evaluated in this review.
