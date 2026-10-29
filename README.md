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
| `agent/`, `models/`, `utils/` | Reserved; no runtime implementation yet |

## Current limits

- The user reported one successful failure-menu retry followed by firing and
  random movement. A later live log completed eight episodes and eight retries
  before a missing bullet-radius error. A full ten-episode check is still pending.
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

The next live acceptance check is 30 seconds of valid state reads, then three
failed spell-practice episodes with automatic retries and verified key release.
After that, validate enemy/laser observations and add a PPO baseline.
