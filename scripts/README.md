# Entry points

Run commands from the TouhouRL source directory.

`python -m scripts.sample_game` performs read-only state sampling. `--samples`
sets the sample count; `--pid` selects an instance; `--profiles` selects a profile
file. The executable path is discovered from the running process.

`--run` waits up to `--focus-timeout` seconds (default 30) for game focus before
sending random nine-direction input. Mid-task focus loss still aborts the run.
`--spell-retry` explicitly enables one confirm press on a spell failure menu.
`--episodes`, `--repeat`, and `--max-frames` bound the diagnostic. Output is JSON
lines on stdout; instructions and errors go to stderr.
`--output runs/check.jsonl` saves events to a new file, including errors and a
partial summary when an episode check fails. Existing files are never overwritten.
`diagnostics.py` implements JSONL recording and the reusable multi-episode check.

The command stops on faults, unsupported reset states, or a non-death ending.
It is an integration check, not a training or evaluation benchmark.
