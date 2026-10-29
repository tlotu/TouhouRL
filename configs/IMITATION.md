# Teacher demonstrations and behavior cloning

## Collect a fixed task

Select the same spell, character, equipment, and difficulty used for the
reactive_attack acceptance run. Start from its result menu and keep game focus.
The task label is a manual declaration, not an automatically verified spell ID.
Use a descriptive label instead of reusing it for a different task.

```powershell
.\.venv\Scripts\python.exe -m scripts.sample_game --run --spell-retry --policy reactive_attack --episodes 10 --capacity 128 --repeat 2 --trace --task spell01-character01-normal --output runs/demos/reactive-attack.jsonl
```

Each transition records the pre-action observation and teacher action. Transition
JSON goes only to the file; reset, periodic steps, episode results, and summary
remain visible. Files can be large. Existing filenames receive numeric suffixes;
use the actual printed path below. A focus/memory error stops capture, but earlier
complete episodes remain usable. Truncated or unfinished episodes are excluded.

## Fit offline

No game process, focus, keyboard input, or GPU is required. Training dependencies
are the same as PPO. At least two successful episodes are required; aim for ten
or more, including varied boss positions and active bullet patterns.

```powershell
.\.venv\Scripts\python.exe -m scripts.clone --demos runs/demos/reactive-attack.jsonl --epochs 30 --run-dir runs/bc-spell01
```

Multiple explicit paths may follow --demos. Schema, build, capacity, frame limit,
and task must match. The loader validates finite observation shapes, contiguous
steps, current teacher labels, and terminal episode boundaries. Duplicate files
are rejected. Successful episodes are used by default; --include-failures
explicitly includes complete classified failures for a separate experiment.
Old traces without a task label or complete observations must be recollected.

The split holds out 20% of entire episodes, with at least one on each side.
Repeated identical spell trajectories can still make this validation optimistic.
This measures imitation on teacher-visited states, not autonomous game success.
Inspect per-action counts and accuracy as well as overall accuracy; absent
actions have null accuracy. Action order is stay, up, up-right, right,
down-right, down, down-left, left, up-left.

The actor minimizes negative teacher log probability with a separate Adam
optimizer. The critic is not supervised. Epoch zero is also eligible; retain
the weights with the lowest held-out cross-entropy. The final checkpoint is
checkpoints/update-000000. Its PPO counters remain zero; supervised epochs and
teacher transitions are not native PPO interactions. Metadata stores source
SHA-256 hashes, episode split, action coverage, hyperparameters, and metrics.
PPO's optimizer starts fresh. There is no forced teacher action or safety filter
when evaluating the cloned policy.

## Evaluate, then fine-tune

First run autonomous evaluation on the manually selected task:

```powershell
.\.venv\Scripts\python.exe -m scripts.train --evaluate --resume runs/bc-spell01/checkpoints/update-000000 --episodes 3 --run-dir runs/bc-spell01-eval
```

Review the actual game behavior before PPO fine-tuning. If the clone fails,
inspect coverage and disagreement before adding PPO budget. Errors can compound
when the clone visits states absent from the demonstrations. Extra teacher
episodes may be needed; high held-out accuracy does not ensure a clear.

Once evaluation is satisfactory, start a short PPO continuation:

```powershell
.\.venv\Scripts\python.exe -m scripts.train --resume runs/bc-spell01/checkpoints/update-000000 --timesteps 2048 --run-dir runs/bc-spell01-ppo
```

The existing result-menu update boundaries and spell-v2 reward still apply.
Continuation preserves cloning provenance in every checkpoint. Keep pure PPO,
teacher performance, cloned evaluation, and PPO fine-tuning results separate.
Initial tests cover synthetic demonstrations and a real supervised/PPO optimizer
round trip. Native demonstration capture and cloned autonomous play remain
unvalidated until user-run acceptance.
