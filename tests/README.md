# Offline tests

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

Tests do not start the game, send real keys, or require a GPU. Passing these tests
does not validate live addresses or the spell failure-menu default selection.
