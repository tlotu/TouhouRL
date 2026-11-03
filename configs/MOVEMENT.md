# Movement mode diagnostics

Recorded width-comparison traces differ even after excluding stops and turns.
For at least three consecutive identical nonzero actions, narrow-band episodes
have cardinal/diagonal speed norms about 2 px/frame; wide-band episode 0 has
about 4, while its other episodes have about 2. Straight motion alone does not
explain the difference. See runs/movement-speed-stratified.json for grouped
histograms, actual frame deltas, examples and source hashes.

## Calibration

Start from the same spell's recognized result menu and focus the game. Do not
press movement or Shift during capture:

```powershell
.\.venv\Scripts\python.exe -m scripts.probe_input --movement-modes --output runs/demos/movement-modes.jsonl
```

This intentionally alternates requested slow/fast input for cardinal and diagonal
holds over 108 requested frames (about 1.8 seconds at 60 FPS). It sends keys,
including Shift release/press, without a model. Keys release at truncation;
gameplay continues, so return to the menu manually. Focus loss and input errors
retain the normal stop/cleanup behavior. This is not an avoidance evaluation.

Three distinct layers are logged in movement_diagnostic:

- requested_slow and owned_keys: emulator intent and local ownership;
- os_shift: GetAsyncKeyState high bits for generic/left/right Shift;
- game_candidates: reference-derived input mask, bit 2, focus byte and four
  speed parameters, with before/after frame counters.

The candidate baseline RVAs come from reference tools/offsets.py and
verify_focus.py: input=0xA6EC60, focus=Player+0x785C, four speeds at +0x7860
through +0x786C. They are available only for MD5
b0cc28ec904e9efce2439ad857c938b5 and the matching player base. They remain
verified=false until local calibration establishes their meaning. Reading
plausible values alone does not validate semantics. Candidate reads do not
change profiles, actor observations, policies or training rewards.

same_transition_frame requires both probe frame reads to match the transition
frame; otherwise do not pair those candidate values with that exact displacement.
Even equal counters do not prove atomic state or the engine input-poll instant.
Nonfinite values become null, implausible data is marked, and candidate read
errors remain explicit diagnostics. Menu samples are skipped.

## Intermittent reproduction

After calibration, --movement-probe can be added to evaluate_attack_zone:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_attack_zone --half-width 24 --hold-frames 4 --episodes 3 --task spell01-character01-normal --movement-probe --output runs/attack-zone-w24-movement.jsonl
```

This records the same layers without changing requested slow mode or adding an
automatic Shift repair. It adds some logging/read overhead. A short calibration
may pass even if the earlier intermittent behavior remains; distinguish scripted
mode changes from unintended mode changes in the rule run. No cause or fix is
established before inspecting the new native data.

## First native mode calibration

runs/demos/movement-modes-002.jsonl completed 108 one-frame transitions without
bullets or deaths. All candidate captures match their transition frame. Slow
cardinal motion is 2 px/frame (32 samples); fast is 4 (32). Diagonal norms are
approximately 2 and 4 (16 samples each), consistent with per-axis parameters
1.414214 and 2.828427. The 12 stationary samples do not move. All 107 nonterminal,
nontruncated records agree across requested/owned Shift, OS Shift, game input
bit, focus byte and measured speed. This supports the candidate semantics for
the selected baseline calibration; it does not establish the historical cause.

On the final truncated step, GameSession releases keys before diagnostic
annotation. OS/owned Shift is therefore released while the game fields still
describe the completed focused frame. New records explicitly tag
post_step_cleanup; do not interpret this timing difference as input loss.
Historical rows can use terminated/truncated for the same exclusion. Game
candidate verified=false remains a source-level diagnostic marker, not a claim
that this calibration failed. No production policy/profile is enabled from it.

The report is runs/demos/movement-modes-002-analysis.json. Continue with the
wide-band --movement-probe run above, which keeps slow requested throughout and
does not intentionally switch modes. This checks for intermittent differences
without a hidden Shift reassertion or changed prediction speed.

## Full rule-run telemetry

runs/attack-zone-w24-movement.jsonl completed three successes at 1045, 1001,
and 1019 frames with zero deaths, dropped bullets or radius fallbacks. Mean time
to result is 17.03 seconds. All 1531 playable telemetry records have aligned
candidate frame reads and agree across requested/owned Shift, OS Shift, input
bit and focus=1. Final result-menu transitions skip movement reads as designed.
All 204 eligible unchanged-action speed samples are about 2 px/frame; speed
parameters remain 4/2 and 2.828427/1.414214. No unintended fast mode was observed.

The report is runs/attack-zone-w24-movement-analysis.json. This validates the
current width-24 held rule under measured slow movement, without establishing
the cause of historical fast episodes. Stop repeating the same probe for this
milestone; preserve the uncertainty and proceed to explicitly versioned teacher
data/training integration. No evaluation trace becomes a cloning dataset by
renaming its collection type.
