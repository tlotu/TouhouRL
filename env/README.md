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

Reader, keyboard, clock, and sleep dependencies are replaceable for offline tests.
Enemy telemetry is in `Snapshot.enemies` and `info.enemies`. State-v2 also adds a
padded `(16, 6)` actor input: relative x/y, boss, collidable, visible, and presence.
HP stays in diagnostic info; rewards are unchanged. Pools are skipped outside playable scenes.
Check `enemies_sampled` before treating an empty list as a measured absence.
HP is not a normalized health bar; increases, nonpositive values, or reused slot
IDs must not be interpreted as damage, death, or spell results without validation.
Live construction is explicit through `windows.connect()`; imports never attach
to a process or send input.
