# Entry points

Run commands from the TouhouRL source directory.

`python -m scripts.sample_game` performs read-only state sampling. `--samples`
sets the sample count; `--pid` selects an instance; `--profiles` selects a profile
file. The executable path is discovered from the running process.

`--run` waits up to `--focus-timeout` seconds (default 30) for game focus before
sending random nine-direction input. Mid-task focus loss still aborts the run.
`--spell-retry` waits for a stable failure menu before confirming. If the menu
remains visible, it retries at most three presses, spaced by at least 0.75 seconds.
No more confirmations are sent after leaving that menu. Timeout errors include
the last scene, player state, retry flag, and confirmation count.
`--episodes`, `--repeat`, and `--max-frames` bound the diagnostic. Output is JSON
lines on stdout; instructions and errors go to stderr.
`--output runs/check.jsonl` saves events to a new file, including errors and a
partial summary when an episode check fails. Existing files are never overwritten.
If the requested path exists, a suffix such as `-001` or `-002` is added. The
actual log path is printed to stderr before connecting to the game.

`python -m scripts.probe_menu capture --label success_default --output runs/menu.jsonl`
waits five seconds, then takes 20 labeled raw captures over about one second.
No input is sent. Labels also include success_retry, death_default, timeout_default,
and combat. See the root README for the screen/selection checklist.
`python -m scripts.probe_menu compare LEFT RIGHT` prints stable-byte candidates
from two labels of the same build. Raw hex stays in files, not the console.
Baseline captures now include `result_candidates` and print the last sample's
unverified active/selection/phase/outcome flag. Other builds may lack candidates.
Read-only samples now include `enemy_count`, `enemies_sampled`, `enemies`, and
`sample_seconds`. Each enemy has slot, x/y, signed hp, is_boss, collidable, visible,
and raw flags1/2/3. Live episode info includes the same enemy records. The reader
does not estimate damage or identify successful spells from these fields.
Summaries distinguish `death_or_retry_episodes` and `unresolved_episodes`.
`successful_episodes` and `classified_failure_episodes` count recognized results.
`success_rate` is null unless every closed episode has a recognized result.
`diagnostics.py` implements JSONL recording and the reusable multi-episode check.

The command stops on faults or unresolved endings. Recognized success and failure
menus can both be retried with `--spell-retry` on the baseline build.
It is an integration check, not a training or evaluation benchmark.

`--policy random|stay|reactive|reactive_attack` selects a baseline. `--capacity` sets the padded
bullet limit (default 64, maximum 640). `--trace` requires `--run --output` and
records every transition, including terminal observations. Connected events
record the policy, capacity, seed, and observation schema. Summary peak/count
metrics cover every step, even when full tracing is disabled.
New logs use `touhourl-state-v2`, adding padded enemy-position/flag observations.
Old state-v1 traces do not have that input and must not be silently mixed with v2.

`--quiet-limit 180` optionally stops after three seconds of bullet-free logic
frames following observed combat. This is a manual-review truncation, not a
success detector. It releases keys and stops multi-episode execution without
sending success-menu input. Initial bullet-free frames do not trigger it.
