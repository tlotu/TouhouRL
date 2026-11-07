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

## Remaining stages

1. Validate timed candidates and forecast omissions across more recorded states.
   Retain the small-scene reference and measure bounded computation and
   uncertainty before live integration.
2. Add worker goal inputs and train safe short-range tracking from current weights.
   Measure goal success, tracking error, safety overrides and old-task retention.
   An unchanged frozen worker cannot follow a goal it does not observe.
3. Freeze the goal-conditioned worker and train a manager with duration-aware
   returns. Begin with one timed region; accumulate actual environment rewards.
4. Add spawn priors, confidence feedback, event triggers and multiple waypoints as
   separate ablations. Preserve snapshots when comparing old forecasts to outcomes.

Required comparisons include a matched flat PPO, standard goal-conditioned HRL,
analytic candidates with rule selection, and the same candidates with learned
selection. Match state information, interaction budget and compute. Report survival,
task outcomes, attack contribution, recovery, multiple seeds, held-out patterns,
latency and omission/model-error diagnostics. Do not infer superiority from three
native episodes.

Detailed Chinese decisions and literature are in the workspace `note.md`, sections
16.63 onward. No manager training or native improvement has been established.
