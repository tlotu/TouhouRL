# Offline tests

Training checks:

- `test_training.py`: Complete-episode collection, transition alignment, reward
  profile, rejection of unknown/truncated endings, update boundaries, and budgets.
- `test_training_boundary.py`: Native session boundary guards.
- `test_checkpoints.py`: Publication, continuation metadata, duplicate names,
  and interrupted saves.
- `test_ppo_backend.py`: Actual optimizer update, model/optimizer restoration,
  and GAE episode boundaries. These two tests skip without Torch and SB3;
  install requirements-training.txt and run them before live training.

Run from the source directory:

```powershell
python -m unittest discover -s tests -v
```

- `fakes.py`: Deterministic clock, reader, keyboard, and snapshot fixtures.
- `test_memory_reader.py`: Binary decoding, bullet filtering, read failures,
  profile selection, invalid values, and frame-crossing samples.
- `test_controls.py`: Nine-direction input, key release, focus loss, frame
  timing, counter wrap, stalls, and bounded spell retries.
- `test_touhou_env.py`: Episode termination/truncation, retry cycles, fault
  cleanup, observation encoding, and the optional Gymnasium adapter.
- `test_windows.py`: Windows x64 structure sizes without live process/input calls.
- `test_diagnostics.py`: Ten consecutive fake episodes, partial summaries,
  initial focus waiting, and error logging.
- `test_baselines.py`: Seeded actions, reactive avoidance, wall handling,
  observation masks, transition alignment, and complete-run metrics.
- `test_quiet_stop.py`: Optional post-combat quiet limits, initial silence,
  new-wave counter resets, and death priority.
- `test_enemies.py`: Enemy-pool boundaries, flags, signed HP, fresh reads,
  non-finite coordinates, short/torn reads, menu skipping, and info isolation.
- `test_attack_policy.py`: Attack direction, target filtering, boss priority,
  clearance constraints, close targets, enemy padding, and policy registration.
- `test_menu_probe.py`: Stable-byte candidates, build/label checks, bounded
  region reads, invalid GUI pointers, and short-read reporting.
- `test_result_candidates.py`: Static candidate offsets, scalar decoding, and
  unavailable-data handling. These tests do not validate candidate semantics.
- `test_result_flow.py`: Result decoding, success/failure terminal states,
  transitional phases, cursor-aware retries, focus loss, unresponsive input,
  and a complete fake success-retry-failure cycle.

Tests do not start the game, send real keys, or require a GPU. Passing these tests
does not validate live addresses or the spell failure-menu default selection.
