# Risk-consistent held PPO

`scripts.train_masked --risk-filter` explicitly migrates physical-mask PPO or
resumes its own `risk-masked-held-smdp-v1` checkpoints. Omitting the flag retains
the old physical-only path, which rejects risk-policy checkpoints. Initial
migration is restricted to compatible physical-mask PPO sources; untouched raw
clones and unrelated PPO contracts are rejected.

## Distribution and timing

`RiskActorCriticPolicy` reuses the scalar `risk_candidates` predictor and physical
endpoint geometry. Forbidden logits become negative infinity; the remaining
distribution is normalized for sampling, saved likelihoods, entropy and PPO
action reevaluation. The existing pre-update likelihood check rejects batches
that do not match the current sampling policy.

Observations are explicitly canonicalized to float32, matching neural inputs.
New evaluation headers record `risk_observation_contract=float32-v1`; historical
external-controller runs retain their original native-float behavior. Evaluation
of an intrinsic risk checkpoint requires both `--boundary-mask --risk-filter`.
The controller uses the same float32 candidates and retains raw logits for
honest action-provenance diagnostics.

The four-frame macro is sampled once. Only physical remaining-hold checks can
end it early; the next macro samples a fresh action. Risk changes alone do not
interrupt holding. Actual durations, frame-discounted rewards/GAE, complete
episodes and result-menu-only updates retain their previous contracts. Each
held-decision log now includes sampling candidates, forecast gaps, mode and
player position under `sampling_context`. Full bullet traces are not recorded
by the training collector.

## Migration and offline evidence

Migration copies every learned tensor and Adam state. It preserves original
demonstration provenance, existing counters and an explicit source checkpoint /
hash chain. New candidate semantics are versioned in `config.risk_filter`.
Reward remains `spell-v3-masked-survival`; optimizer, width, capacity and hold
settings remain unchanged. The trainer selects one declared seen task per run;
automatic task scheduling and spiral adaptation are not implemented.

The prepared checkpoint is
`runs/ppo-risk-zone24-preflight/checkpoints/update-000003`, copied from the
reviewed physical-mask update 3. `offline-review.json` verifies exact parameter
and Adam equality, 1389 existing decisions, three published updates, six
optimization epochs, and twelve Adam states at step 46. Preparation makes no
native input or gradient update on this real model.

Across 4846 historical observations from the three risk-controller tasks,
candidate sets and deterministic proposals match. Float32 conversion changes
neither on this dataset; maximum conditional-probability error is 3.58e-7.
This does not guarantee identical live trajectories. The shared CPU predictor
adds collection/optimization work; actual timing remains measured.

The full suite passes 247 tests. New coverage includes candidate/controller
agreement, sampling/reevaluation likelihoods, nonempty Adam migration, actual
actor gradients, optimizer restoration and another update, probability rejection,
config/evaluation guards, fake-native training and canonical evaluation, and
source preservation. Historical spiral states are used only for offline
equivalence checks, not gradient fitting.

## First native smoke

Select the previously validated dense straight stage-6 spell, keep character,
equipment and difficulty fixed, and start at its result menu:

```powershell
.\.venv\Scripts\python.exe -m scripts.train_masked --risk-filter --resume runs/ppo-risk-zone24-preflight/checkpoints/update-000003 --task stage06-straight-dense-character01-normal --decisions 256 --rollout-decisions 256 --run-dir runs/ppo-risk-zone24-smoke
```

This is 256 additional fresh decisions, completed to an episode boundary.
Expect one batch update after the complete collection; inspect actual logs and
checkpoint numbers. Original sources remain frozen. `training.jsonl` and
`training.console.log` record the actual run directory, including any suffix.
The first completed native run is recorded below. Review training before final
checkpoint evaluation and old-task/spiral regressions.

The geometric gate still ignores lifecycle timers, future spawns and omitted
bullets. Restricting candidates does not guarantee survival or improvements over
the frozen actor/controller reference; some states have only one allowed action.

## First native update completed

`runs/ppo-risk-zone24-smoke` collects two successful pre-update episodes at
712/745 frames, zero deaths: 366 fresh decisions, 729 substeps and 1457 frames.
All sampled actions belong to their logged risk sets; candidates reconstructed
from stored gaps/player positions match every sampling context. There are 40
single-candidate decisions and two emergency decisions. One physical hold
interrupts after two frames; its previous action is infeasible in the next
recorded remaining window. Full bullet traces remain unavailable.

The update publishes `checkpoints/update-000004`, reaching 1755 cumulative
decisions, eight optimization epochs and Adam step 58 (source step 46).
Approximate KL=2.3431e-6, clip fraction=0, explained variance=-0.18616.
Collection takes 29.212 seconds; optimization 0.962 seconds. The critic is not
established as reliable. Both successes precede this optimization.

`offline-review.json` verifies exact starting policy/Adam equality, finite
parameters/state, actual actor changes, source hashes, matching resume settings
and the latest pointer. On historical fixed observations, final argmax differs
from risk update 3 at 1/1488 stage-1, 0/1078 dense and 10/2280 spiral states.
This is drift evidence, not a post-update survival estimate.

Hold further training. From the same dense straight spell's result menu, with
character/equipment/difficulty fixed, independently evaluate final update 4:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_control --resume runs/ppo-risk-zone24-smoke/checkpoints/update-000004 --boundary-mask --risk-filter --hold-frames 4 --episodes 3 --task stage06-straight-dense-character01-normal --output runs/ppo-risk-zone24-smoke-stage06-eval.jsonl
```

Inspect the result before old-task/spiral regressions or additional training.
Risk checkpoints require both flags so evaluation uses canonical float32 risk
candidates and physical hold interruption. No new native run was performed
during this offline review.

## Update 4 independent dense evaluation

`runs/ppo-risk-zone24-smoke-stage06-eval.jsonl` completes three successful
episodes at 731/732/742 frames, zero deaths and no incomplete episodes. Mean
time to result is 12.25 seconds at 60 Hz. The frozen model SHA256 is
`26e4050fd6a5b06be2bb3377a6fc921e1ffb84148f320c5abcc3003933166945`.

The adjacent `-analysis.json` verifies all 1103 transitions against canonical
controller replay and the loaded intrinsic policy's candidates/deterministic
proposals. All fresh choices satisfy their risk sets; all executed actions
satisfy the modeled physical window. There are 552 fresh choices, including
one emergency choice. Risk constraints affect 39 fresh choices relative to raw
logit ranking; this is not replacement of sampled PPO actions. No physical hold
interrupts. Risk changes alone retain the existing hold, including 48 held
substeps whose action is outside the newly computed candidate set.

Peak bullet count is 240 with 112 omitted; lifecycle/future-spawn limitations
remain. Three unpaired successes do not establish improvement. The model is
registered in `runs/ppo-risk-zone24-smoke/evaluated-model.json`; training stays
paused pending three stage-1 and three correctly selected spiral regressions.
Keep character, equipment and difficulty fixed; select each spell manually and
start from its result menu. Run separately, switching spells between commands:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_control --resume runs/ppo-risk-zone24-smoke/checkpoints/update-000004 --boundary-mask --risk-filter --hold-frames 4 --episodes 3 --task spell01-character01-normal --output runs/ppo-risk-zone24-smoke-stage01-eval.jsonl
```

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_control --resume runs/ppo-risk-zone24-smoke/checkpoints/update-000004 --boundary-mask --risk-filter --hold-frames 4 --episodes 3 --task stage06-spiral-delayed-character01-normal --allow-task-transfer --transfer-reason "Frozen risk-consistent PPO update 4; unseen spiral regression" --output runs/ppo-risk-zone24-smoke-spiral-eval.jsonl
```

## Update 4 regressions: hold training

The submitted stage-1 regression is 2/3 at 1502/1013/1680 frames. Its failed
episode records neither deaths nor dying flags; the final boss-tagged sample
retains 230 HP at frame 1558 while the player is near x=372.67. This is consistent
with insufficient damage and timeout, but no decoded timer confirms the cause.
`death_or_retry_episodes=1` counts the failure and must not be reported as a death.
Spiral completes 3/3 at 1022/1959/1277 frames. Resets show 900 HP at (192,96)
for stage 1 and 1600 HP at (192,96) for spiral, matching historical signatures;
spell selection remains manual. Dense remains 3/3, separately recorded.

Both adjacent `-analysis.json` reports verify 4228 transitions against canonical
controller replay and the loaded final policy's candidates and modes. Physical
windows and fresh risk membership pass. No physical holds interrupt, no snapshot
race retries occur, and both model identities match the frozen update 4.
Spiral omits up to 124 bullets (252 peak); stage 1 omits none (69 peak).

On these fixed observations, prepared risk update 3 differs from update 4 at
17/2098 stage-1 proposals (10 fresh) and 8/2130 spiral proposals (zero fresh).
The failed stage-1 episode has nine differences, five fresh. Its first difference
is at frame 616, after substantial rightward drift. At frame 600, all forecast
gaps cap at 24 and both models prefer right despite a leftward boss. Risk gating
does not resolve this preference when all directions pass. These comparisons
do not simulate the source model's native trajectory or establish PPO regression.

Freeze both models and run one bounded three-episode stage-1 source control.
Use the prepared intrinsic risk update 3 to preserve identical float32 candidate
semantics; it contains the pre-smoke weights/Adam, with no new risk update.
Keep character/equipment/difficulty fixed, select the original stage-1 spell,
and start at its result menu:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_control --resume runs/ppo-risk-zone24-preflight/checkpoints/update-000003 --boundary-mask --risk-filter --hold-frames 4 --episodes 3 --task spell01-character01-normal --output runs/ppo-risk-zone24-preflight-stage01-control.jsonl
```

Review this control before additional training or reward/controller changes.
Preserve the failure and the earlier reference; update 4 is not promoted.
