# TouhouRL

An early reinforcement-learning environment for **th06nc.exe** on Windows x64.
The current environment is a real-time, single-life spell-practice loop. A small
CPU PPO training entry point is implemented; screen capture is not implemented.

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
| `agent/` | Random, stationary, reactive avoidance, and attack-alignment baselines |
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
- Initial task selection remains manual. The validated baseline build now
  supports recognized success/failure retries with `--spell-retry`.
- External sampling cannot freeze the engine or guarantee exact action repeats.
  `delta_frames` measures elapsed logic frames, including input/acquisition delay.
- Observations are padded bullet and enemy arrays, not images. Lasers and enemy
  body hitboxes are not encoded. The nearest 64 bullets may omit important threats; inspect
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
terminated, truncated, info)` transition. New traces use the declared state-v2 schema;
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
limits, and scene exits. `success_rate` is reported only when every closed episode
has a recognized result-menu outcome. Closed episodes are not automatically wins.
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
build profiles include `EM_RVA`. The baseline build was checked against a manual
run in `runs/enemy-check-001.jsonl`: 153 samples, 114 with Boss slot 0 occupied,
HP 747 down to 1, movement in both axes, then 39 empty-pool samples. All samples
remained in scene 2 without a retry flag or player death. Maximum recorded sample
time was about 4.14 ms. The final KeyboardInterrupt was a manual stop, not a read
failure. The newer build remains unverified.

HP did not visibly reach zero: it stayed at 1 while collision disabled, before
the slot disappeared. Do not implement `hp <= 0` as the sole completion rule.
Neither disappearance nor that transition is yet proven unique to a spell clear.
To collect more live evidence:

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
and collision flags require interpretation. Enemy HP remains diagnostic only.
Enemy positions and flags are now available to policies in state-v2; rewards and
automatic result handling are unchanged.

## Attack-alignment baseline

```powershell
python -m scripts.sample_game --run --spell-retry --episodes 3 --capacity 128 --policy reactive_attack --quiet-limit 180 --output runs/reactive-attack.jsonl
```

The new baseline tries to align below a visible, collidable boss while preserving
nearly optimal predicted bullet clearance. It follows a moving target but does
not predict target velocity. If no boss qualifies, it can target regular enemies
above the player. It does not chase invisible/non-collidable enemies or enemies
below the player. Safety estimates still omit lasers and enemy-body collisions.

State-v2 adds `enemies`, a `(16, 6)` padded array with rows
`[dx/384, dy/448, is_boss, collidable, visible, present]`. Relative coordinates
are clipped to [-1, 1]. Bosses sort first, then visible/collidable enemies and
distance; `info.dropped_enemies` reports omitted slots. The actor never receives
raw HP. The observation layout has changed: state-v1 traces/models are not
directly compatible. Random, stay, and reactive ignore the added enemy input.

Live check: `runs/reactive-attack.jsonl` completed 551 transitions (1102 logic
frames) with no errors, radius fallbacks, or dropped bullets. Sampled boss HP fell
from 900 to 1, then its slot disappeared. The user confirmed attack alignment,
evasion, and an explicit successful spell screen. This is one manually confirmed
clear, not an estimated success rate. The automatic result remains unresolved.

The user observed this success-menu order, initially highlighting the first item:

1. Return to spell card selection
2. Save replay
3. Retry

Success retry therefore needs two Down taps followed by confirmation from that
known initial selection. It must not reuse the failure-menu confirm-only route.
This navigation is now enabled on the validated baseline profile: it reads the
current selection and only confirms Retry in phase 1. Quiet stops still require
manual review. Boss disappearance alone does not create a win label.

## Result-menu discovery

Reference material does not yet identify a verified spell-success flag or menu
cursor field. `scripts.probe_menu` captures small, known memory neighborhoods
without sending input. Labels describe the screen observed by the operator;
they do not make the captured data an automatic classifier.

For each command below, switch back to the specified screen within five seconds,
then leave its selection unchanged for about one second. The command prints the
actual log path; existing paths receive numbered suffixes. Capture only menus,
not an actively running policy. Keep other input tools stopped.

```powershell
# Successful result, default Return to spell card selection highlighted.
python -m scripts.probe_menu capture --label success_default --output runs/menu-success-default.jsonl

# Same success screen, manually press Down twice to highlight Retry, without confirming.
python -m scripts.probe_menu capture --label success_retry --output runs/menu-success-retry.jsonl

# Failure after taking a hit, leave its default selection unchanged.
python -m scripts.probe_menu capture --label death_default --output runs/menu-death-default.jsonl

# Failure after surviving until the spell timer expires, leave its default unchanged.
python -m scripts.probe_menu capture --label timeout_default --output runs/menu-timeout-default.jsonl
```

The first two captures can come from the same successful result. Other captures
can be collected when those outcomes next occur; no need to repeat every case at
once. A `combat` label is available for a later control capture.

Compare captures using their actual paths:

```powershell
python -m scripts.probe_menu compare runs/menu-success-default.jsonl runs/menu-success-retry.jsonl
python -m scripts.probe_menu compare runs/menu-success-default.jsonl runs/menu-timeout-default.jsonl
```

The comparison excludes bytes changing within either capture and reports up to
64 stable differences per region. These are candidates only: unrelated game
state can also differ, and candidates must reproduce in independent trials.
The probe tool never sends input. Runtime result navigation is separately enabled
only by the validated baseline profile.

Probe evidence: `menu-success-default-001.jsonl` and `menu-success-retry.jsonl`
each contain 20 samples from the baseline build, with no region read errors.
There were zero stable differences in status, player_state, and gui_pointee.
The first captured bytes also matched exactly in all regions. The sampled
neighborhoods do not currently expose a cursor candidate; no menu offset should
be inferred from these captures. Locate the NC-specific menu code in the exact
executable before asking for broader memory snapshots. The TH06 reconstruction's
Gui/GuiImpl definitions are leads only, not verified NC result-menu layouts.

Subsequent static analysis located a different owner: the AsciiManager spell
result menu. Baseline-only candidate addresses are now included in probe captures.
See `configs/MENU_RESULT_EVIDENCE.md`. Repeat the two success captures with this
version; the command also prints unverified active/selection/phase/outcome values.
Expected selection 0 -> 2 is a hypothesis to test, not an enabled input rule.

Success captures `menu-success-default-002.jsonl` and
`menu-success-retry-001.jsonl` now confirm selection 0 -> 2, with active=1,
phase=1, and outcome_flag=1 in all 20 samples each. No read errors occurred.
Death and timeout captures have since completed the cross-check below.

## Recognized result retry

Baseline build `b0cc28ec904e9efce2439ad857c938b5` now supports recognized results.
Both failure captures contain 20 samples of active=1, selection=2, phase=1,
outcome_flag=0. Success captures have outcome_flag=1. See the evidence document.

```powershell
python -m scripts.sample_game --run --spell-retry --episodes 3 --capacity 128 --policy reactive_attack --output runs/result-retry.jsonl
```

Omit `--quiet-limit` for this check. Episode reasons are `spell_success` and
`spell_failure`; the result flag alone does not distinguish death from timer
expiry. Result activation releases combat keys. Transient result phases have up
to three seconds to settle. Reset reads selection before each bounded Down tap,
confirms only Retry (index 2), and waits for a new playable scene. Keep game focus.

Summaries add `successful_episodes` and `classified_failure_episodes`. Success
rate remains null if any ending is unresolved or classified only by legacy death
detection. No success reward bonus was added. Initial result screens can be
retried, but their outcomes are not counted as episodes. The newer build retains
legacy behavior.

Live acceptance passed in `runs/result-retry.jsonl`: three complete episodes,
one failure followed by two successes, with two inter-episode retries and no
errors or unresolved outcomes. All resets returned to playable state. The
observed 2/3 success rate is a three-trial smoke test, not a performance estimate.

The next training stage should collect complete episodes and update the policy
while the verified result menu remains active. Resume with reset only after the
update finishes. This avoids leaving movement held in a running battle during
optimization; engine pause/resume and fixed-step control remain unimplemented.

## Small PPO training

Install optional training dependencies, then run the offline tests first:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-training.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The two test_ppo_backend tests must run rather than skip. They check an actual
optimizer update, GAE boundaries, and checkpoint loading. All 120 tests now pass
with the training dependencies installed.

Select the spell manually and enter its success or failure RESULT menu. Then:

```powershell
.\.venv\Scripts\python.exe -m scripts.train --timesteps 512 --rollout-steps 512 --run-dir runs/ppo-smoke-v2
```

Keep the game focused after the prompt. This starts from random policy weights.
It collects at least 512 transitions, finishes the current episode, updates in
the result menu, saves a checkpoint, and leaves the menu open. The final complete
episode can exceed the requested budget. Existing run names get numeric suffixes.

Resume from the actual printed checkpoint path for additional transitions:

```powershell
.\.venv\Scripts\python.exe -m scripts.train --resume runs/ppo-smoke-v2/checkpoints/update-000001 --timesteps 2048 --run-dir runs/ppo-continued-v2
```

Evaluate without updating, again starting in the selected spell's result menu:

```powershell
.\.venv\Scripts\python.exe -m scripts.train --evaluate --resume runs/ppo-smoke-v2/checkpoints/update-000001 --episodes 3 --run-dir runs/ppo-evaluation-v2
```

The native smoke run completed 514 transitions across two failed episodes, one
PPO update, and checkpoint saving. Sampling averaged 24.42 transitions/second;
the update took 0.125 seconds. Continuation also passed: eight episodes, four
further updates, and 2684 cumulative transitions saved in
runs/ppo-continued/checkpoints/update-000005. All training episodes failed;
Historical deterministic evaluation failed all ten episodes at 438 frames each.
New training uses spell-v2 survival credit instead of spell-v1's time penalty;
old checkpoints are evaluation-only. Start a fresh run for the new reward.
See configs/TRAINING.md for reward semantics, fault handling,
and limits. No acceleration or remote machine is used.
