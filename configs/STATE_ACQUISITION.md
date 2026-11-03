# Bounded recovery from frame-crossing snapshots

The interrupted masked spiral run completed seven successes at
1122/1267/1110/1671/1567/1445/1962 frames, zero deaths. Episode 7 then stopped
after its last logged transition at 956 frames with:

```text
Game advanced during every sample; retry acquisition
```

This indicates that all three internal snapshot attempts read different frame
counters before and after acquisition. It is not evidence of a collision,
invalid address, boundary-controller error or a classified episode failure.
The exact rejected frame pairs and cause of the scheduling delay were not
logged. Do not infer a particular host process or claim a measured speedup.

## Recovery behavior

SnapshotRaceError now distinguishes this transient case from MemoryReadError.
The reader still requires matching before/after frame counters, retains the
same three internal attempts and validates all original data. No addresses,
pool layouts, observation capacities or parsing semantics changed.

FrameStepper retries only SnapshotRaceError within the existing 0.5-second
step deadline, checking focus before every retry and yielding 1 ms after a race.
It counts failed calls in snapshot_calls and snapshot_seconds, and records
snapshot_race_retries in timing. That count is failed snapshot calls, not each
individual internal attempt. The next accepted frame determines actual elapsed
frames; it is never reported as a fictional two-frame step or replaced with an
old sample. Existing counter-jump and deadline checks still apply.

Reset uses its existing deadline, result settlement keeps its three-second
deadline, and update-boundary verification now has a 0.5-second acquisition
deadline. Short reads, invalid values, process errors and focus loss still abort;
continuous races or slow acquisition still time out. GameSession releases owned
keys on a failed step. This does not guarantee recovery from an extended stall.

Evaluation headers declare acquisition_contract=same-frame-bounded-retry-v1.
Control evaluation records interrupted_decision if an action was chosen but no
transition could be accepted. Its episode, step, last observed elapsed frames
and pending raw/masked/committed actions preserve the distinction. Generic
episode summaries now also expose incomplete_episodes; prior logs are unchanged.
An incomplete episode is excluded from classified success/failure counts.

Ten new offline tests cover recovery with measured overshoot, sustained/slow
race deadlines, fatal read propagation, focus loss, reset and update-boundary
recovery, step cleanup, result settlement, and pending-action diagnostics.
The targeted suite passed 52 tests; the full suite passed 227. Native recovery
under the updated acquisition path has not yet been observed.

## Preserve the partial experiment and finish separately

runs/bc-zone24-joint-stage06-spiral-boundary-repeat-analysis.json preserves the
seven completed episodes and the incomplete eighth separately. All 5552 recorded
transitions replayed correctly. The controller summary counts 5553 decisions:
one more action was chosen before acquisition failed, without a transition.
Do not invent that action's result or treat summary.last_reason as episode 7's
outcome; it refers to the last completed episode.

The seven complete episodes include 120 fresh mask replacements: 33 toward the
left boundary and 87 toward the right. Episodes 5/6 had 28/59 right-boundary
replacements and still succeeded; they spent 140/338 active frames at x>=368.
This is actual right-side intervention evidence, while the long dwell shows
that early retreat remains a separate issue. The incomplete prefix includes
another 17 right-side replacements, with no verified final outcome. None of
the recorded actions violated its modeled execution-horizon mask.

Return manually to the same spell's result menu; do not count continuation from
the interrupted live state as a new completed trial. Collect three additional
episodes with unchanged weights, controller, character, equipment and difficulty:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_control --resume runs/bc-zone24-joint/checkpoints/update-000000 --boundary-mask --hold-frames 4 --episodes 3 --task stage06-spiral-delayed-character01-normal --allow-task-transfer --transfer-reason "Spiral task; frozen actor with explicit hold-endpoint boundary mask" --output runs/bc-zone24-joint-stage06-spiral-boundary-repeat-remaining.jsonl
```

The new log records the acquisition revision; keep it separate from the older
partial run. Three future completions may fill the planned ten completed trials,
while the original aborted episode remains part of the run history. Source
model hash and masking contract stay unchanged. No game was driven or model
fitted during this repair.

The actual follow-up file, repeat-remaining-001.jsonl under the same run prefix,
completed all three episodes successfully with the new acquisition contract.
No errors or frame-race retries occurred. This verifies a clean native run of
the updated path, while recovery from an actual native race remains unobserved.
The old interrupted episode is retained. See BOUNDARY_CONTROL.md for completion
statistics and the survival-first interpretation of its upper-screen route.

Both later masked regressions added six complete successful episodes on the
trained tasks, without errors or race retries. The updated acquisition path
has clean native runs; actual recovery from a native race is still untriggered.
