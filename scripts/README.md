# Entry points

Run commands from the TouhouRL source directory.

`python -m scripts.sample_game` performs read-only state sampling. `--samples`
sets the sample count; `--pid` selects an instance; `--profiles` selects a profile
file. The executable path is discovered from the running process.

`--run` enables random nine-direction input after a five-second focus countdown.
`--spell-retry` explicitly enables one confirm press on a spell failure menu.
`--episodes`, `--repeat`, and `--max-frames` bound the diagnostic. Output is JSON
lines on stdout; instructions and errors go to stderr.

The command stops on faults, unsupported reset states, or a non-death ending.
It is an integration check, not a training or evaluation benchmark.
