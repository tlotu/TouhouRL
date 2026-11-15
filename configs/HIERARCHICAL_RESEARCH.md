# Hierarchical risk-aware control

Research direction: analytic known-hazard forecasts, learned timed-region
selection, and a goal-conditioned worker initialized from the current BC/PPO
policy. Survival remains the primary objective. Prediction error trains a model
or confidence estimate; it is not automatically a control reward.

## Stage 1: offline forward model and planning probe

Implemented independently of native input and PPO:

- `agent/forward_model.py`: constant-velocity circles and timed static laser
  capsules. Collision checks cover entire movement segments, including activation
  within an action. Laser interval endpoints are conservative. Boundary tolerance
  is 1e-4 pixels for float32 encoding roundoff.
- `agent/timed_planner.py`: time-expanded reference and beam search over stay,
  up, right, down, left, using four-frame actions at two pixels/frame. Both methods
  share the same lattice. This action subset deliberately excludes diagonals;
  reference completeness is limited to this graph and supplied model.
- `TimedGoal`: a position, arrival window and tolerance radius. Arrival is checked
  at action boundaries. This contract is not yet an input to the trained worker.
- `scripts/probe_planning.py`: deterministic cases, optional frozen trace states,
  returned-route replay, JSON reports, SVG views and a saved console transcript.

Run from the Python repository:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.probe_planning --output runs/planning-micro
```

Inspect the existing laser trace without opening the game:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.probe_planning --trace runs/ppo-laser-young-smoke-slack4-eval.jsonl --state 0:130 --state 0:150 --state 0:162 --state 1:130 --horizon 96 --beam-widths 16 64 --repeats 3 --output runs/planning-stage1
```

Output directories are numbered rather than overwritten. Each contains
`report.json`, `planning.console.log`, one SVG and one numeric NPZ per case.
NPZ arrays store frame-indexed reference/beam reachability and physical grid
coordinates. One means model-reachable; zero is unmarked, not proof of danger.
They are not yet a calibrated remaining-survival-time representation. Raw source hashes,
observation omissions/clipping indicators, versions and timing samples accompany
the results. Metadata for a pre-action observation comes from the previous
transition when its next observation matches; post-action counts are not silently
assigned to the earlier state. Select fresh decisions only: partial action holds
are not modeled by this probe.

The reference enumerates reachable states at each time layer, not every safe
position in the field. Beam search preserves multiple first-action groups where
available; pruning can still lose the best route. No map cell receives a guaranteed
survival time. A path to a goal proves model-safe arrival only. A frontier route is
safe only up to its recorded endpoint time. Failure to find a route is not proof of
native inevitability.

For trace diagnostics, an explicit rule scores candidate positions by a laser-only
component area at the earliest time of maximum known simultaneous activation
within the horizon. It is a geometric baseline, not a learned selector or survival
probability. Future geometry at the endpoint can differ from this scoring partition.
It does not hard-filter blocked grid cells. Micro-cases instead prioritize goal
distance and use region areas only in reports. SVG panels show hazards at zero and
the requested horizon, with route vertices annotated by relative frame in titles.

Known limitations:

- No new circles/lasers, acceleration, moving laser axes or player-dependent fire.
- Policy observations omit excess circles and may clip positions/velocities.
- Spawned-but-inactive circles are not distinguished by the observation contract.
- Native collision widths, phase edges and execution timing remain approximate.
- Search timing excludes decoding, the reported region-setup time, input and game
  acquisition. Width bounds are not a hard wall-clock deadline.
- Native RNG is uncontrolled. Offline alternative routes are not native wins.

The first implementation must remain offline until its timing, action coverage,
model validity and failure behavior meet the deployment contract.

First trace probe: `runs/planning-stage1-002/report.json`, with four selected
states, 96-frame horizon and three timing samples per method. At episode 0,
frame 260, reference and beams found model-safe routes into a larger laser
component; at frame 324 the shared cardinal model found complete held prefixes
only through 32 frames. These are not native success/death predictions. Width-16
search medians ranged from 46.70 to 116.85 ms on the three states reaching the
full horizon; region setup added about 20-30 ms. This is not ready for live use.
Earlier probe directories preserve the initial horizon-partition experiment.
The final report uses peak activation to avoid scoring a fully faded laser field
as a large combat compartment. Artifact/source verification is saved separately
in `provenance.json`. The full offline suite passes 303 tests.

## Stage 2 offline short planner

`agent/swept_collision.py` uses time-layer swept boxes for circle and capsule
culling. It filters out-of-bounds edges before hazard evaluation and preserves
the full scan's clearance up to 24 pixels. `agent/short_planner.py` uses all
nine directions and four-frame holds with beam width 16. Angular cost is 0.5
per radian. The terminal score adds 0.25 per frame of stationary TTC, capped at
60 frames. This estimates time to contact while waiting at the leaf position.

The adaptive variant requests 60 frames when a known warning threatens a
nominal exit ray with at most 12 frames of slack. Slack accounts for the
estimated time to clear the beam and a four-frame reserve. This ray estimate
can miss a real exit or flag a ray unrelated to the eventual route. Smoothing
considers moving actions at most 45 degrees apart and selects a fully checked
alternative beginning with the previous action. It also requires equivalent
goal arrival and comparable score, clearance and TTC.

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.probe_short_planning --trace runs/ppo-laser-young-smoke-slack4-eval.jsonl --state 0:130 --state 0:150 --state 0:162 --state 1:130 --output runs/short-planning
```

`runs/short-planning-001/report.json` compares clearance-only short search,
short search with TTC, and adaptive search with TTC. Micro-case pass counts
were 7/9, 8/9 and 8/9. A pass requires finishing the model rollout and meeting
the scenario's arrival expectation. Each mode found at least 16 reference-
checked frames on all four frozen snapshots. The short TTC planner took
2.09–2.92 ms median per snapshot over seven repetitions, including scoring and
TTC. Timing excludes trace decoding, reference replay and game IO.

The static detour case still misses its arrival deadline. Known warning rays
do not trigger long search there. The comparison uses offline modeled rollouts
and cannot measure new projectiles or native clears. The 311-test suite and
output logs accompany the experiment.

## Stage 3 offline budgets and candidate coverage

The planner accepts an optional edge cap and a wall-clock limit checked between
batches. A cutoff returns the last checked prefix when available. An in-flight
batch can exceed the time limit. Logs distinguish budget stops, blocked routes,
and whether terminal TTC was evaluated.

Add `--audit-budget` to the Stage 2 command and select a new output directory.
The audit uses unpruned nine-direction search for the 16-frame reference and
records retained first actions by layer. Spare beam slots can favor distinct
8-pixel cells after the first-action quotas. The report also records candidate
routes and proposed and selected actions for a future replay inspector.

`runs/planning-budget-001/report.json` records 33/36 retained first actions for
width 16, 34/36 with spatial coverage, and 36/36 at width 48 across four trace
states. Width 48 took 4.86–7.26 ms median per state. The 378-edge short budget
completed 16 frames in all 28 trace calls. The 900-edge adaptive budget returned
28-frame checked prefixes on all 21 long-search calls. The static detour remains
an arrival failure. The full offline suite passed 314 tests.

## Stage 4 static subgoals and decision replay

`--guide-static` adds a checked cardinal-route guide to the short planner.
The guide splits turns and straight segments into timed subgoals. It applies
when persistent laser geometry blocks the direct goal and the static reference
finds a route. The detour reaches its goal at frame 28 of 32. The nine model
fixtures improve from 8/9 to 9/9.

Open `runs/planning-replay-001/replay.html` in a browser. The file under
`scripts/` is the source template. To export a fresh copy from a planning or
offline HRL report, use a new output filename.

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.replay_planning runs/planning-replay-001/report.json --output runs/planning-replay-001/replay-test.html
node tests/check_replay.cjs runs/planning-replay-001/replay-test.html
```

The page shows the current action, candidate directions, predicted clearance
status, stationary TTC after a held move, and search stop reason. Use Space to
play or pause and the arrows to step. Compare `detour_before_approach / short_ttc`
with `detour_before_approach / static_guided_ttc` to see the wait and the detour.
Frozen trace snapshots distinguish recorded
actions from predicted alternatives. The timeout example uses an injected clock.
The collision label refers to extensions rejected by the model or bounds. The
JavaScript harness checks controls and labels. Headless browser startup failed
on this machine.

Replay fresh decisions from the recorded Young Demon Lord evaluation.

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.replay_laser_trace runs/ppo-laser-young-smoke-slack4-eval.jsonl --window 0:260:360 --window 1:260:360 --window 2:260:360 --output runs/young-demon-lord-trace-replay
```

Open `replay.html` in the output directory. Use a new output directory for
each run. Windows use a zero-based episode and inclusive game-frame bounds.
Purple shows the recorded motion, and green shows the checked model route.
Switching modes within a window preserves the selected decision. Each forecast
starts from its recorded observation and does not affect later observations.

For a counterfactual, branch the player state and refresh hazards from each
logged sample. The end frame must be a logged sample endpoint.

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.replay_laser_counterfactual runs/ppo-laser-young-smoke-eval.jsonl --window 2:380:393 --keep-first-control --output runs/laser-counterfactual
```

The report separates collision, no valid plan, and the end of available data.
It includes contact intervals with 0, 1 and 2 px padding. The replay uses cyan
for branch motion. Recorded hazards can reflect the original player's position.

### Native shadow check

Select Young Demon Lord on Lunatic with character 01 and leave the game at
the spell result menu. Focus the game after launching the evaluation.

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_laser --resume runs/ppo-laser-young-smoke/checkpoints/update-000004 --task stage06-young-demon-lord-character01-lunatic --episodes 3 --planner-shadow --output runs/laser-planner-shadow.jsonl
.\.venv\Scripts\python.exe -m scripts.analyze_laser_shadow runs/laser-planner-shadow.jsonl --output runs/laser-planner-shadow-analysis.json
```

Shadow proposals are synchronous at fresh holds. Input still follows the
frozen policy and its original clearance gate. The planner checks its 900-edge
and 8 ms budgets between layers. The report separates planner time from
observation-to-input age and finds the first observed fully firing volley.
Input send completion is a host timestamp. Game consumption is unmeasured.

### Cached planner validation

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_laser --resume runs/ppo-laser-young-smoke/checkpoints/update-000004 --task stage06-young-demon-lord-character01-lunatic --episodes 10 --planner-cache-shadow --output runs/laser-planner-cache-shadow.jsonl
.\.venv\Scripts\python.exe -m scripts.analyze_laser_shadow runs/laser-planner-cache-shadow.jsonl --output runs/laser-planner-cache-shadow-analysis.json
```

The cached planner rechecks action suffixes against the current sample with
2 px of padding. Held steps only recheck cached motion. It compares TTC at a
common leaf time and checks the frame counter before proposing an override.
The 14 ms age limit reserves time for input dispatch. The frame read and key
send are separate operations.

After ten matching shadow episodes pass the 5 ms p95 gate, pass the raw
shadow JSONL to `--planner-intervene` and use a new output path for ten trials.
The gate checks both fresh-step and pooled work timings. Use the same checkpoint,
task, and candidate slack, then compare outcome and elapsed frames by result.

### Direction-key timing probe

From the same spell result menu, run the single-frame input probe and focus
the game. It holds slow mode and varies cardinal, diagonal, and neutral moves.

```powershell
.\.venv\Scripts\python.exe -m scripts.probe_input --switch-timing --output runs/demos/laser-switch-timing.jsonl
.\.venv\Scripts\python.exe -m scripts.analyze_input_switch runs/demos/laser-switch-timing.jsonl --output runs/demos/laser-switch-timing-analysis.json
```

The probe brackets each send sequence with the game frame counter and times
individual key events. The analyzer checks movement compatibility with key
states. It cannot directly observe the engine's per-frame input order.

Repeat the same schedule with `--switch-batch` and a separate output file to
test one `SendInput` call per direction change. This mode is probe-only.

Both probes showed the same per-frame direction sequence. To shadow the
transition check, add `--planner-switch-guard` to the cached planner run.

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_laser --resume runs/ppo-laser-young-smoke/checkpoints/update-000004 --task stage06-young-demon-lord-character01-lunatic --episodes 10 --planner-cache-shadow --planner-switch-guard --output runs/laser-planner-switch-guard-shadow.jsonl
```

The guard checks the proposed first hold with zero and one frame of start
delay and validates any queued tail. It defers overrides while earlier keys
may still be queued or the same-frame direction cannot be confirmed. The
cached route and leaf TTC retain their nominal motion model. Intervention
with the guard requires a matching shadow log.

## Stage 5 frozen-worker offline PPO

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.train_offline_hrl
```

The worker starts with checkpoint `update-000004` logits and a zero-initialized
goal adapter. The adapter fits safe local movement labels on 12 translated and
reflected static layouts. Eight held-out layouts gate manager training. The
manager chooses one of two lanes or the destination every eight frames. Its
observation includes the opening side and its reward uses bounded route progress.

The first guide changed goals before aligning with the far bank and failed the
gate. `runs/hrl-frozen-smoke-001/report.json` records the corrected run. After
8,192 manager decisions, held-out completion improved from 0/8 to 8/8. The
learned manager took 44–60 frames, compared with 28–32 for the corrected guide.
Source and worker digests were verified across training. This is a single-seed
offline static-task smoke test. Restore the adapter with the source archive
identified by its SHA-256 digest.

### Buffered waypoint PPO

```powershell
.\.venv\Scripts\python.exe -m scripts.train_buffered_hrl --steps 4096 --seed 7 --output runs/hrl-buffered-smoke
```

The buffered beam propagates queued direction events and tests each frame of
every expanded hold. A branch can use a different sequence or wait for a gate.
The leaf TTC includes releasing the current input before waiting.

The manager selects among three waypoints in synthetic timed laser arenas.
The existing goal adapter and source checkpoint stay frozen. The worker uses
the adapter's preferred action when a checked route from it reaches the
waypoint. Training varies the modeled start delay by episode. Held-out layouts
run at both delays and include a timed rule and a direct-goal reference. Use
`--hide-buffer` to remove buffer features and delay from the manager observation
while preserving the executor. Live evaluation retains the established cached
planner baseline. The switch guard remains opt-in.

For longer runs, use `--steps 51200 --eval-every 5120` with a new output
directory per seed. Curves are recorded after completed PPO updates on a
separate validation set. The final checkpoint is evaluated on fixed test layouts.

Profile the buffered search on the native input path in shadow mode.

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_laser --resume runs/ppo-laser-young-smoke/checkpoints/update-000004 --task stage06-young-demon-lord-character01-lunatic --episodes 3 --planner-buffered-shadow --output runs/laser-planner-buffered-timing-shadow.jsonl
.\.venv\Scripts\python.exe -m scripts.analyze_laser_shadow runs/laser-planner-buffered-timing-shadow.jsonl --output runs/laser-planner-buffered-timing-shadow-analysis.json
```

The host pipeline sums the accepted observation read, observation age at
input start, and input send time. The first transition has no aligned read
time. Shadow measures the baseline actor and buffered beam with settled-key
hypotheses. The synthetic manager has no native waypoint interface yet.

### Young Demon Lord log replay

```powershell
.\.venv\Scripts\python.exe -m scripts.train_replay_hrl --steps 4096 --seed 7 --output runs/hrl-young-replay-smoke-001
```

The manager ranks directional waypoints over a shared short-beam route pool.
Only 16-frame routes that pass the 2 px clearance check for all modeled delay
sequences are eligible. The frozen native actor contributes to route ranking.
The executor applies one hold and replans from the simulated endpoint and
the queue states consistent with that endpoint. Training samples 96-frame
windows. Evaluation uses a separate capture at fixed delays and checks selected
routes with the reference collision model.

The report records trace completion, modeled collisions, failed searches,
and offline planning time. Recorded hazards refresh every two frames and cannot
respond to simulated movement. The planner has a deterministic edge budget;
timings do not include native observation acquisition or input dispatch.

Use `--long-reward --exploration 0.10` to test delayed continuation rewards
and a 10% probability mass on distinct non-rule first moves. PPO uses the same
mixture for rollout and update likelihoods. Evaluation uses the learned policy
without the mixture. The reward checks the best available leaf TTC 32 frames
after a choice and confirms eligible 60-frame forecasts against replay survival
with the 2 px margin. These checks use subsequent observations only when they
arrive.

`--conditional-reward` applies long-horizon shaping only to decisions flagged
by a drop in escape coverage or TTC, low TTC, or an imminent laser with limited
escape directions. Other decisions earn a small below-boss alignment reward.
`--attack-reward` uses that alignment reward throughout the episode for comparison.

`scripts.benchmark_replay_search` compares both searches on identical replay
states and evaluates the early exit in closed loop. The shortcut requires
three distinct sustained directions to pass the full buffered clearance check
and have enough terminal TTC. Imminent laser warnings use the full search.

`scripts.train_replay_supervised` tests each verified first direction on
training captures at danger triggers. It clones successful choices and fits
the PPO critic to the conservative branch return. Closure comes from a known
laser crossing an escape ray, with at least five frames of follow-up.

`scripts.benchmark_replay_async` tests the four-frame speculative planner in
paced replay. The native `--planner-async-shadow` flag keeps baseline inputs and
logs timing and proposal status. A ready result still needs a matching
position and input queue and a fresh collision check.

On a miss, the async benchmark and native shadow use a four-frame buffered
clearance gate. Only the background worker runs beam search. The benchmark
reports boundary time including the fallback and next submission.

`scripts.train_replay_escape` samples prefixes 60–120 frames before rule
failures on training captures. It retains the waypoint actor and slot zero,
and matches additional verified directions to the remaining waypoint slots.
The escape bonus requires an actual action change and safe progress beyond
the matched rule stop. A connectivity forecast scans the current hazards over
at least 400 frames, opening candidates early when a contraction is forecast.
The grid component is a spatial estimate; execution still verifies 16 frames.

## Remaining stages

1. Validate timed candidates and forecast omissions across more recorded states.
   Retain the small-scene reference and measure bounded computation and
   uncertainty before live integration.
2. Validate worker tracking on dynamic hazards and measure old-task retention.
3. Compare the frozen-worker manager across seeds and layouts. Evaluate
   duration-aware returns when goal options can end at different times.
4. Add spawn priors, confidence feedback, event triggers and multiple waypoints as
   separate ablations. Preserve snapshots when comparing old forecasts to outcomes.

Required comparisons include a matched flat PPO, standard goal-conditioned HRL,
analytic candidates with rule selection, and the same candidates with learned
selection. Match state information, interaction budget and compute. Report survival,
task outcomes, attack contribution, recovery, multiple seeds, held-out patterns,
latency and omission/model-error diagnostics. Do not infer superiority from three
native episodes.

Detailed Chinese decisions and literature are in the workspace `note.md`, sections
16.63 onward. Native performance remains unmeasured for this controller.
