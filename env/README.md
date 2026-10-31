# Game environment

- `memory_reader.py`: Immutable snapshots and bulk bullet reads. Rejects short,
  invalid, and frame-crossing samples. Resolves profiles by executable hash.
  Missing hitbox dimensions use the reference reader's 3 px radius estimate.
  Estimated bullets remain in observations and are marked in diagnostic info
  (`radius_fallback_count` and `radius_fallback_slots`). Invalid positions,
  velocities, and short reads still raise errors.
  Also bulk-reads 256 enemy slots per playable sample and preserves their raw
  signed HP, position, and flags. Occupancy uses flags1 bit7; invisibility and
  collision flags are diagnostic attributes, not reasons to discard a slot.
- `windows.py`: Lazy Win32 bindings for process discovery, read-only memory
  access, and foreground-only scan-code input. No game-file writes or injection.
- `action_emulator.py`: Nine directions, optional fixed Z/Shift, and owned-key
  release. Directions are stay, up, up-right, right, down-right, down, down-left,
  left, and up-left (IDs 0-8).
- `frame_stepper.py`: Bounded real-time waiting using the byte frame counter.
  Measures actual elapsed frames; does not provide deterministic engine stepping.
- `reset_controller.py`: Initial task attachment and opt-in failure-menu retry.
  Waits for menu settling and uses bounded confirmation retries while the menu
  stays visible. Stops confirming once a loading or playable state is observed.
  Rejects ordinary active-task resets and unsupported menus.
- `session.py`: Single-life episode loop, padded observations, event rewards,
  frame limits, fault handling, and cleanup. Usable without Gymnasium.
- `touhou_env.py`: Gymnasium adapter with `Discrete(9)` actions and float32
  dictionary observations. `seed` does not seed the native game.
- `menu_probe.py`: Bounded raw-state captures and stable-byte comparisons for
  manually labeled result menus. It neither classifies outcomes nor sends keys.
  The baseline profile also exposes statically derived result-menu candidates.
  Decoded candidate fields remain explicitly unverified and probe-only.

Reader, keyboard, clock, and sleep dependencies are replaceable for offline tests.
Enemy telemetry is in `Snapshot.enemies` and `info.enemies`. State-v2 also adds a
padded `(16, 6)` actor input: relative x/y, boss, collidable, visible, and presence.
HP stays in diagnostic info; rewards are unchanged. Pools are skipped outside playable scenes.
Check `enemies_sampled` before treating an empty list as a measured absence.
HP is not a normalized health bar; increases, nonpositive values, or reused slot
IDs must not be interpreted as damage, death, or spell results without validation.
Live construction is explicit through `windows.connect()`; imports never attach
to a process or send input.

Result-menu evidence: the user-confirmed success screen defaults to Return to
spell card selection, followed by Save replay and Retry. The failure-menu route
is not valid for this screen. The baseline profile now provides
`Snapshot.result_menu`, settled outcome classification, and selection-aware retry.
Quiet-combat heuristics never authorize menu input. Other profiles retain legacy
detection. Unexpected phases, invalid reads, and focus loss stop input.

`GameSession.verify_training_boundary()` requires an ended session and a settled
recognized result menu, releases owned keys, and checks foreground focus. The
PPO loop calls it before and after optimization; it does not advance or reset
the task. Model updates are never allowed inside an active episode.

Step info includes `timing`: age since the prior snapshot returned when input
starts, input-send duration, post-input frame-wait duration, snapshot read counts
and durations, and previous/current requested actions. Age includes logging and
decision computation between steps. Reset age begins when reset returns. Read
completion is not the engine's capture timestamp, and SendInput return does not
mean the game consumed the key. Terminal settling time is outside frame-wait
timing. These fields measure host stages rather than exact engine input latency.
Session/frame-stepper defaults use `time.perf_counter` for sub-frame measurements.
The older `time.monotonic` implementation on this Windows Python has 15.625 ms
resolution; zero values in historical logs must not be treated as zero cost.
