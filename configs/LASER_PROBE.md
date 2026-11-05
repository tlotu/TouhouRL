# Read-only laser layout probe

Target: stage 6, user-reported "young demon lord": persistent static lasers
partition the field while large round bullets are emitted. Difficulty is a
manual label; keep the intended Lunatic selection explicit.

`scripts.probe_lasers` sends no input, shooting, retry, focus or pause commands.
It does not load a model or alter the policy state schema. Play and retry
manually during capture. The first objective is observability, not survival or
native control evaluation.

## Candidate layout and evidence status

The decoder is isolated in `env/laser_probe.py`. Sources are
`Reference/th06nc-autododge-main/tools/offsets.py`, `read_lasers.py` and
`sample_lasers.py`. No reference code is imported or executed.
The manager RVA matches the existing validated build profile, but laser layout
and phase semantics have not yet been independently validated in this project.
Every record says `verified=false`. Unsupported builds or a mismatched manager
RVA are rejected before the candidate pool is read.

| Field | Candidate offset / value |
| --- | --- |
| Build MD5 | `b0cc28ec904e9efce2439ad857c938b5` |
| Manager RVA | `0x3EC2A0` |
| Pool within manager | `0xF5018` |
| Slot count / stride | 64 / `0x298` |
| Width / phase timer | `0x004` f32 / `0x00C` i32 |
| Start / end axial offset | `0x010` / `0x014` f32 |
| Fade duration | `0x018` i32 |
| Angle / fire duration / speed | `0x260` f32 / `0x268` i32 / `0x26C` f32 |
| Warning duration / slot state | `0x270` i32 / `0x274` u8 |
| Origin x / y / z | `0x278` / `0x27C` / `0x280` f32 |
| Length limit / phase | `0x288` f32 / `0x290` u8 |

Nonzero slot state selects candidates. Phase hints from the reference are
0 warning, 1 firing, 2 fading, 3 tail; they do not assert verified damage state.
Candidate segment endpoints are origin + unit(angle) * start/end offset.
The origin is not assumed to be the segment start. No collision mask, player
clearance, static/rotating classification or future activation is inferred yet.
Raw duration fields are preserved without inventing remaining-time semantics.

Nonfinite or implausible candidate values remain visible as issue-tagged rows;
nonfinite floats become JSON null, with raw bytes preserved. Unknown phases do
not silently become safe. Candidate head/tail bytes are always logged; optional
full-pool base64 also retains inactive slots for independent offset review.

## Capture

Manually select the target spell. Run this command, switch to the game during
the delay, then manually start/play/retry to show warning, sustained lasers,
large round bullets and disappearance within the capture window:

```powershell
.\.venv\Scripts\python.exe -m scripts.probe_lasers --spell "stage06 young demon lord" --difficulty Lunatic --delay 8 --seconds 45 --interval 0.1 --raw-pool --output runs/demos/young-demon-lord-laser-probe.jsonl
```

The difficulty argument records a manual selection; it does not select or verify
the game setting. JSONL and `.console.log` paths, including automatic suffixes,
are printed and preserved. Full-pool capture can produce tens of MB.
Submit those paths with any observed activation/phase mismatch. A short optional
video can help align visual phases, but capture does not require video.

## Acquisition and interpretation

Regular state, all decoded circular bullets (up to the reader's 640-slot pool),
enemy HP/position and candidate laser pool are bracketed by the byte logic-frame
counter. Up to six attempts are allowed; exhausted races are skipped and counted.
Other memory errors stop capture and close the reader. Menus/loading states skip
the pool to avoid stale laser data. Matching byte counters reduce concurrent
ambiguity; they do not freeze the engine or prove consistency across a full wrap.

At 0.1-second intervals this is a sampled diagnostic, not every logic frame.
Paused/duplicate frames and missing intermediate phases must be interpreted
explicitly. Summary distinguishes candidates, no candidates in combat, no combat
samples, and no valid samples; none automatically verifies or rejects the layout.

Review field plausibility, segment placement, phase transitions and overlap with
large circular bullets first. Then verify against visible timing, optionally
static disassembly of the exact executable, and offline collision tests. Only
after that should laser observations or a new policy contract be designed.
Old model/controller results remain frozen; no laser support is claimed from
passing synthetic tests alone.

Eight tests cover origin/offset geometry, diagnostic invalid fields, build guards,
bounded races, raw byte recovery, short-read propagation, stale-menu avoidance,
read-only fake capture/cleanup, CLI guards and wrong-build cleanup.

## First native capture reviewed

`runs/demos/young-demon-lord-laser-probe.jsonl` records 449 accepted samples,
330 combat pools, no exhausted acquisition races, peak 48 candidates and 161
round bullets. All 330 full pools redecode to their saved candidate rows.
There are 8160 sampled slot records and no invalid-field issues. Seven observed
candidate bursts each contain 48 slots. Within each burst, origin, angle, width,
start/end offsets, speed and length limit remain unchanged. For 7248 adjacent
same-phase slot pairs, timer increments match the modulo byte-frame delta.

Width is consistently 20, start/end offsets 0/440, speed 0. Three raw duration
groups are warning/fire/fade = 60/90/16, 76/74/16 and 92/58/16, each represented
equally in the sampled records. Phase counts are 4224 warning-hint, 3360
firing-hint and 576 fading-hint slot samples; these are not frame durations or
unique lasers. Staggered transitions explain mixed phase-0/phase-1 samples.
One burst ends early near boss defeat; do not infer unconditional lifetime
semantics from every disappearance.

Laser candidates and circular bullets coexist on 168 samples (phase-1 candidates
and bullets on 89). Bullet radii are 8 or 16 with no fallback estimates. The
regular reader exposes all 161 peak bullets here; this is not a 128-capacity
policy trace. It confirms acquisition overlap without proving native hitboxes.

Observed menu-entry sequences are failure, success, failure, failure, consistent
with the user's manual one-win report. Capture begins mid-attempt and is sparse;
these are not four complete policy episodes or a measured model success rate.

`runs/review_laser_probe.py` produces the adjacent `-analysis.json` and
`-reconstruction.svg` from the original data. The reconstruction compares samples
136/151/161 from the same burst: reference phase 0/1/2. It overlays candidate
segments/widths, decoded circular bullets, player and boss in field coordinates.
It is a reconstruction, not a screenshot. Raw file hashes are preserved.

Current evidence establishes raw decoding consistency and strong structural /
temporal plausibility. All live records retain `verified=false`: visual segment
placement, native dangerous phase, collision width/endcaps and damage transitions
are still unverified. Ask the user to compare the reconstruction with the spell's
field partition before making an observation/controller contract. Do not repeat
the same bulk capture solely to accumulate samples. No policy or geometry
runtime changes are made in this review; training remains paused.

## Visual feedback and offline geometry hypothesis

The user reports that the reconstructed SVG approximately matches the game.
This supports approximate segment placement on this spell; it does not confirm
the exact damage width, endcaps, phase transitions, or other laser types.
Original capture records remain `verified=false`.

`agent/laser_geometry.py` provides an isolated offline hypothesis: static
segments, capsule distance using half the candidate width plus player radius,
reference phase/timer assumptions, and integer-frame prediction through 12
frames at 2 px/frame. A warning may become active during the horizon. Unknown
phases and nonzero reported speed raise errors. Rotation, axis translation,
bullet risk, walls, latency and subframe collision are not modeled. This is
not wired to a live controller or policy observation.

`runs/review_laser_geometry.py` applies the hypothesis to 170 sampled states
containing laser candidates. Seven have no action with 4 px estimated clearance;
122 allow all nine actions; minimum best clearance is -5.323 px. These figures
use human-play observations and the unverified geometry/phase assumptions, and
do not predict native survival or prove a controller's future action.
The adjacent `-geometry-hypothesis.json` preserves per-state gaps and limits.

Four new tests cover segment endpoints, phase onset/expiry, trajectory crossing
and fail-closed handling of unknown or moving lasers. The full suite passes
266 tests; output is in `runs/laser-geometry-tests.console.log`.

The existing PPO path accepts only its declared Normal training tasks and its
observation schema contains no laser input. Risk masking also needs to use the
same distribution during sampling and optimization. Do not label an old-policy
PPO run as laser learning. Before actual laser training, independently check
phase/collision behavior, version and connect laser observations, and ensure
the training action/likelihood contract matches any laser safety rules.

For a more closely sampled manual phase check, select the same spell and use:

```powershell
.\.venv\Scripts\python.exe -m scripts.probe_lasers --spell "stage06 young demon lord" --difficulty Lunatic --delay 8 --seconds 45 --interval 0.02 --raw-pool --output runs/demos/young-demon-lord-laser-timing.jsonl
```

Manually observe warning and bright phases and, if practical, note whether
touching a warning line or firing line caused a hit. The probe remains read-only;
sampling at 0.02 seconds does not guarantee every game frame is captured. The
user may retry manually. Preserve the prior 0.1-second capture separately.

The follow-up and separate static-laser PPO preparation are recorded in
[LASER_PPO.md](LASER_PPO.md). Raw probe records retain `verified=false`.
