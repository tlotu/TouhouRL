# Held-decision PPO continuation

Use scripts.train_held for attack-zone-held-v1 clones or its own held-smdp-v1
checkpoints. The legacy scripts.train remains blocked for these models. Preserve
the successful clone and runs/bc-zone24/baseline.json for independent comparison.

## Sampling and time contract

Sample the network once at a fresh decision and retain that observation, action,
value and log probability. Execute exactly that sampled action on successive
two-frame session steps until at least the saved hold duration has elapsed or
the episode terminates. Intermediate observations are not separate PPO actions;
there is no ActionCommitment override, teacher or safety filter in collection.

The game is still real-time, so actual duration can exceed four frames. An early
terminal can end sooner. Every transition stores measured duration_frames.
Collect complete episodes and update only at verified result menus. Artificial
truncations, unknown endings, focus/read errors abort the batch. The last batch
row must be truly terminal; no artificial bootstrap or cross-episode GAE occurs.

The decision rate matches held evaluation logically, but intermediate neural
inference is omitted in training. Changed host timing can affect real-time native
trajectories; identical trajectories are not guaranteed.

## Frame returns

Define gamma_frame=sqrt(0.999) and lambda_frame=sqrt(0.95), preserving the old
two-frame reference decay. For duration d, bootstrap uses gamma_frame**d and
the GAE recursion uses (gamma_frame*lambda_frame)**d, with episode-boundary masks.
Advantages/returns are populated explicitly; SB3 still supplies PPO optimization.

Per observed frame, survival reward is 0.0005. Success adds 5 and failure -1 at
the final observed frame. Microstep rewards are discounted within the held
interval; GAE applies the remaining duration discount between intervals. The
version is spell-v2-frame-discounted. It is not an unmarked continuation of the
old reward stream. A clone has zero PPO history, so migration is explicit at
initialization; further resumes require matching contract, rewards and settings.
No HP, alignment or imitation regularization term is added.

## Conservative first update

| Setting | Value |
| --- | --- |
| Learning rate | 1e-5 |
| Optimization epochs | 2 |
| Minibatch | 64 |
| Entropy coefficient | 0 |
| Clip range | 0.1 |
| Target approximate KL | 0.005 |

The cloned critic was not supervised. These settings reduce update pressure but
do not prevent forgetting or guarantee improved play. Sampling is stochastic,
unlike deterministic baseline evaluation. Do not compare raw training success
fractions to the frozen baseline as a controlled improvement estimate.

Start from the same manually selected spell's result menu and keep focus:

```powershell
.\.venv\Scripts\python.exe -m scripts.train_held --resume runs/bc-zone24/checkpoints/update-000000 --task spell01-character01-normal --decisions 256 --rollout-decisions 256 --run-dir runs/ppo-zone24-smoke
```

Budget units are fresh held decisions, not two-frame substeps. The minimum batch
and final whole episode may exceed 256. This invocation produces one optimizer
update after its complete batch. It writes a starting checkpoint to a new run,
then update-000001, without modifying the source archive. Actual paths are printed
at start/end, with full training.jsonl and compact training.console.log.

Inspect metrics, successes, actual frame counts and checkpoint integrity before
another training run. Then evaluate the new archive through the existing held
entrypoint; keep the printed directory if a numeric suffix was added:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_control --resume runs/ppo-zone24-smoke/checkpoints/update-000001 --hold-frames 4 --episodes 3 --task spell01-character01-normal --output runs/ppo-zone24-smoke-eval.jsonl
```

Resume from the last completed held checkpoint using train_held and a new run
directory. --decisions is additional work. Policy, optimizer, settings, counters,
original cloning provenance and source model hash persist; game state and exact
native/Python random streams do not. A source clone may only initialize this
path with zero PPO counters. No live fine-tuning result has been recorded yet.

## Offline verification

Tests cover action/likelihood association, no intermediate resampling, short
terminal holds, frame overshoot, intra-hold rewards, duration GAE and episode
boundaries, aborted batches, menu updates, additional budgets, configuration
validation, real gradients, saved optimizer restoration, second updates, and CLI
checkpoint publication without touching the source. The real clone also passed
offline load/configuration with unchanged archive hash; see
runs/bc-zone24/held-ppo-preflight.json. This did not connect to the game or update
the baseline weights.
