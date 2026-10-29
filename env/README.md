# Game environment

- `memory_reader.py`: Immutable snapshots and bulk bullet reads. Rejects short,
  invalid, and frame-crossing samples. Resolves profiles by executable hash.
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
Live construction is explicit through `windows.connect()`; imports never attach
to a process or send input.
