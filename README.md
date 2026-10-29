# TouhouRL

An early reinforcement-learning environment for **th06nc.exe** on Windows x64.
The current milestone is a real-time, single-life spell-practice loop. PPO and
screen capture are not implemented yet.

## Setup

Use 64-bit Python 3.10 or later. Run these commands from this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The acquisition/control/session tests use only the Python standard library.
The Gymnasium adapter test is skipped when Gymnasium is unavailable.

## Read-only check

Start the game manually, enter spell practice, and run:

```powershell
python -m scripts.sample_game --samples 30
```

This prints the executable hash, player state, and bullet count. It sends no keys.
Unknown builds fail with an error; they never fall back to old addresses.
Profiles are reference data, not proof that this machine has been validated.

## Input and failure-retry check

First verify the read-only values. Select one spell manually and confirm that
its failure menu defaults to **Retry**. Start the command, then focus the game
within 30 seconds and keep it focused:

```powershell
python -m scripts.sample_game --run --spell-retry --episodes 10 --output runs/retry-check.jsonl
```

The diagnostic sends random nine-direction movement with Z and Shift held.
It can lose the challenge. It releases owned keys on termination, errors,
focus loss, and Ctrl+C. It never selects difficulty, character, or spells.
`Game window lost focus` deliberately aborts the check. Initial focus is awaited;
mid-task focus loss never silently resumes the same episode. Existing log names
receive a numbered suffix automatically; the actual path is printed at startup.
The JSONL log records resets, episode endings, summaries, and errors.
Ten completed failure episodes with nine observed retries validate repeated
failure handling, not successful spell completion.
`--spell-retry` is an explicit task-context assertion: the available memory
flag cannot distinguish a normal Continue menu from a spell failure menu.
Do not use that option in a normal run or after moving the failure-menu cursor.

For a movement-only check without automatic retries:

```powershell
python -m scripts.sample_game --run --episodes 1
```

Each episode ends on a confirmed death or failure menu. Zero reserve lives
does not end an episode. Scene exits are reported without claiming success.
A frame limit truncates the episode and requires manual task selection;
resetting an ongoing task is not implemented.

## Modules

| Directory | Purpose |
| --- | --- |
| `env/` | Memory sampling, keyboard control, timing, retry logic, and Gymnasium adapter |
| `configs/` | Build-specific memory profiles and reference attribution |
| `scripts/` | Read-only and random-action live diagnostics |
| `tests/` | Offline unit tests with fake process, clock, and keyboard dependencies |
| `agent/` | Random, stationary, and reactive comparison policies |
| `models/`, `utils/` | Reserved; no runtime implementation yet |

## Current limits

- The user reported one successful failure-menu retry followed by firing and
  random movement. `runs/retry-check-02-002.jsonl` subsequently completed 10 failure
  episodes, 9 retries, and 4729 logic frames with no errors. This passes the first
  failure-retry milestone for the selected spell and baseline game build.
  Its 84 logged step samples had no radius fallbacks; intervening steps were not
  logged. Peak sampled bullet count was 68, with 4 omitted at capacity 64.
- Missing bullet dimensions now use the reference reader's 3 px estimate and
  are reported through `radius_fallback_count` / `radius_fallback_slots`.
  This estimate may differ from the actual hitbox; inspect frequent fallbacks
  before trusting collision-sensitive training or evaluation.
- Start and successful completion still require manual task selection. Automatic
  retries cover failed spell practice only, under the menu assumption above.
- External sampling cannot freeze the engine or guarantee exact action repeats.
  `delta_frames` measures elapsed logic frames, including input/acquisition delay.
- Observations are padded nearest-bullet arrays, not images. Enemies and lasers
  are not encoded. The nearest 64 bullets may omit important threats; inspect
  `dropped_bullets` before training on dense patterns.
- Each bullet row is `[dx/384, dy/448, vx/16, vy/16, radius/32, state/4, present]`.
  Values are clipped to the declared bounds. States 1-4 include live and spawning
  bullets; unused and despawning slots are omitted.
- The player vector is `[x/192-1, y/224-1, reserve_lives/8, bombs/8]`, clipped.
- Frame counts are bytes. Short counter wraps are handled; large jumps, stalls,
  and long gaps between calls invalidate the episode.
- PPO update-time pausing, native RNG seeding, and arbitrary active-task reset
  are not implemented. Do not start long unattended training yet.

## Baseline comparison

Run each policy on the same manually selected spell, character, and difficulty:

```powershell
python -m scripts.sample_game --run --spell-retry --episodes 10 --capacity 128 --policy random --output runs/random.jsonl
python -m scripts.sample_game --run --spell-retry --episodes 10 --capacity 128 --policy stay --output runs/stay.jsonl
python -m scripts.sample_game --run --spell-retry --episodes 10 --capacity 128 --policy reactive --output runs/reactive.jsonl
```

Compare `elapsed_frames` in episode endings, not just returns. Summaries now
count every transition and report peak bullet counts, dropped bullets, and radius
fallback steps. These metrics cover the complete run, unlike periodic step logs.
Game RNG is not controlled, so these are repeated trials, not paired seed tests.

Add `--trace` to save every `(observation, action, reward, next_observation,
terminated, truncated, info)` transition. Traces use the declared state-v1 schema;
they are useful for inspection and future imitation datasets, not directly as
on-policy PPO replay. JSONL traces can be large and add real-time I/O overhead.

Next: validate the reactive baseline and enemy/laser observations, then implement
verified pause/resume or episode-boundary updates before PPO training.

## Successful spell screens

The current memory fields do not distinguish spell result screens from active
play. The user confirmed that surviving a spell without defeating the enemy can
still fail the challenge. A previous apparent clear was incidental; the reactive
baseline only optimizes avoidance, not enemy damage or reliable spell clears.
`reactive-002.jsonl` ended with 1738 elapsed frames, including 180 quiet frames,
zero deaths, and a user-confirmed failed result. The result was not automatically
recognized. Do not compare result-screen waiting against random's mean 464.7
frames (7.745 seconds) as if both were measured combat survival.

Summaries report `death_or_retry_episodes` instead of the ambiguous legacy
`failed_episodes`. `unresolved_episodes` includes quiet stops, external time
limits, and scene exits. `success_rate` is null until reliable result detection
exists. `completed_episodes` counts closed environment episodes, not game wins.
Historical JSONL files keep their original fields and are not rewritten.

For a bounded manual check, add `--quiet-limit 180`. After bullets have first
appeared, 180 consecutive bullet-free logic frames truncate the episode with
`reason=quiet_after_combat`, release keys, and stop the run for manual review.
This option is disabled by default: gaps between waves and laser-only patterns
can also be quiet. It never awards a success bonus or navigates the success menu.
Select the spell manually and start a new run after reviewing the result.

```powershell
python -m scripts.sample_game --run --spell-retry --episodes 10 --capacity 128 --policy reactive --quiet-limit 180 --output runs/reactive.jsonl
```

Reliable spell-success detection and success-menu retry remain prerequisites
for automated success-rate evaluation and training on successful episodes.
Next, validate enemy position/HP telemetry and add a safety-aware attack-alignment
baseline. Enemy disappearance or an HP decrease alone must not label a clear:
phase changes, invulnerability, and practice timeouts need separate handling.

## Enemy telemetry validation

Enemy position/HP acquisition and offline tests are implemented. Both bundled
build profiles include `EM_RVA`. Live values still need verification:

```powershell
python -m scripts.sample_game --samples 300 --output runs/enemy-check.jsonl
```

This samples for approximately 30 seconds, sends no keys, and records all occupied
enemy slots. While it runs, switch to the game and manually observe movement,
shooting aligned with the boss, and a practice ending. Check whether boss x/y
follow the visible character and HP changes during effective attacks. Read-only
mode does not require game focus; the game may pause when unfocused.

`enemies_sampled=false` means the pool was intentionally skipped (such as in the
retry menu), not that all enemies were killed. Slot reuse, HP resets, invisibility,
and collision flags require interpretation. Enemy telemetry is diagnostic only;
the baseline policies, rewards, and automatic result handling are unchanged.
