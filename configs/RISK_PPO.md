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

## Source control and opt-in attack return

The prepared risk update-3 source control also finishes stage 1 at 2/3:
1680-frame failure, then 1003/1025-frame successes, no recorded deaths/dying.
The user confirms the first episode timed out while repeatedly trying right.
Its last boss-tagged sample retains 225 HP at frame 1558, player x=364.73.
All 1855 transitions replay exactly; the loaded source policy matches candidates
and proposals. This demonstrates the failure exists before the risk update;
it does not establish equal reliability or rule out update effects.

`--attack-return` is a separate frozen-controller experiment, requiring both
existing risk and boundary flags. `safe-attack-band-return-v1` uses exactly the
existing risk candidates and physical hold checks. Outside the checkpoint's
attack half-width, normal-safe candidates rank by horizontal progress into the
band, vertical detour, stay, then retreat. Actor logits break ties in each group.
Visible/collidable targets above the player are boss-first, then nearest in x/y,
matching the feature target convention. No valid target, being inside the band,
or emergency mode preserves the original risk proposal. Unsafe approach never
overrides the gate; retreat remains possible if it is the only candidate.

This is local ranking, not a multi-step route planner. A vertical move may allow
later lateral recovery, but that path is not predicted. Absolute y is not
penalized, and homing damage/HP/timers are not new inputs. Risk changes and target
changes alone do not interrupt the four-frame hold.

Logs use `attack_return_control_evaluation` and a separate policy/contract tag.
They retain the raw, physical and original risk proposals, target dx, band
excess, return proposal and override status; summary counts separate proposed
changes from applied fresh choices. This externally ranked controller has not
been trained by PPO. Do not use its actions with the intrinsic policy's saved
sampling likelihoods or pool its results with the original controller.

`runs/ppo-risk-zone24-preflight-stage01-control-analysis.json` audits the source
and probes new rankings on its fixed observations: 198/928 fresh choices differ
(142/35/21 by episode), all inside unchanged risk sets; emergency proposals are
unchanged. This is not closed-loop success evidence. Seven added tests cover
safe recovery, detours, stay/retreat fallback, emergency precedence, target
eligibility, holds, CLI guards and fake-native provenance/source preservation.
The full suite passes 254 tests. Console output is saved in
`runs/attack-return-tests.console.log`.

Keep training paused. Select the original stage-1 spell and keep character,
equipment and difficulty fixed. From its result menu, evaluate the same frozen
source weights with only this ranking change:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_control --resume runs/ppo-risk-zone24-preflight/checkpoints/update-000003 --boundary-mask --risk-filter --attack-return --hold-frames 4 --episodes 3 --task spell01-character01-normal --output runs/ppo-risk-zone24-preflight-stage01-return-eval.jsonl
```

This native experiment is pending. Inspect survival, damage and return behavior
before old-task regressions or considering any training-distribution change.

## Attack-return first native result and expansion order

`runs/ppo-risk-zone24-preflight-stage01-return-eval.jsonl` completes 3/3,
zero deaths/dying, at 1115/995/1058 frames (17.6 seconds mean to result).
The same prepared risk update-3 archive remains frozen. All 1585 transitions
replay exactly; loaded intrinsic risk candidates/proposals match. Fresh actions
stay inside unchanged risk sets and modeled execution windows. Return ranking
affects 61/793 fresh choices (24/12/25); no physical hold interrupts occur.
Peak bullets=70, omitted=0, radius fallbacks=0, snapshot race retries=0.
The adjacent `-analysis.json` records hashes and per-episode trajectory ranges.
This is promising controller evidence, not a learned-weight gain, a paired
effect estimate, or evidence that Normal learning is exhausted.

Run one bounded compatibility check: three dense and three spiral episodes
using the same frozen source and the new controller. Switch spells manually,
keep character/equipment/difficulty fixed, and start at each result menu:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_control --resume runs/ppo-risk-zone24-preflight/checkpoints/update-000003 --boundary-mask --risk-filter --attack-return --hold-frames 4 --episodes 3 --task stage06-straight-dense-character01-normal --output runs/ppo-risk-zone24-preflight-stage06-return-eval.jsonl
```

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_control --resume runs/ppo-risk-zone24-preflight/checkpoints/update-000003 --boundary-mask --risk-filter --attack-return --hold-frames 4 --episodes 3 --task stage06-spiral-delayed-character01-normal --allow-task-transfer --transfer-reason "Frozen risk update 3 with attack-return ranking; spiral compatibility" --output runs/ppo-risk-zone24-preflight-spiral-return-eval.jsonl
```

If compatible, evaluate an explicitly identified Lunatic spell dominated by
straight round bullets as an unseen frozen task before adding training budget.
Record manual task/difficulty selection and inspect bullet omissions, emergency
frequency, measured timing, corner recovery, death versus timeout and remaining
boss HP. Failure can reveal representation/controller limits as well as policy
limits. Do not increase capacity without a separately versioned compatibility
change. Repeated easy-card evaluations are not the next training curriculum.

Laser support comes later: current memory acquisition/state space/clearance
forecast do not encode laser segments, widths or activation phases. First audit
read-only acquisition and collision geometry, then observation/model contracts
and offline tests, before treating laser deaths as policy evaluations. No new
Lunatic or laser native experiment has occurred during this review.

## Attack-return compatibility completed

The frozen prepared risk update 3 plus attack-return controller completes dense
3/3 at 724/841/711 frames and spiral 3/3 at 1255/1172/1075 frames, without
recorded deaths/dying. All 2891 new transitions replay exactly and loaded risk
policy candidates/proposals match. Fresh choices satisfy their risk sets;
execution windows remain feasible. Return ranking affects 35/570 dense and
69/876 spiral fresh choices, all in normal-safe mode. No physical holds interrupt
and no snapshot race retries occur. Dense peaks at 237 bullets, omitting 109;
spiral peaks at 258, omitting 130. Radius fallbacks remain zero.

Reset signatures match the previously checked spells: dense 750 HP at (192,128),
spiral 1600 HP at (192,96). Selection is still manual. Adjacent `-analysis.json`
reports preserve hashes and trajectories. The separate controller registry
`runs/ppo-risk-zone24-preflight/attack-return-evaluation.json` now records all
three tasks at 3/3 each and current source hashes. This bounded compatibility
check is complete; no more Normal repetitions are scheduled. These small,
unpaired samples do not establish superiority or general difficulty mastery.

Next, manually select one Lunatic spell dominated by straight round bullets,
without lasers; prefer a corresponding version of the earlier dense spell if
available. Keep character/equipment and all controller settings fixed. Start at
its result menu and run three frozen episodes:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_control --resume runs/ppo-risk-zone24-preflight/checkpoints/update-000003 --boundary-mask --risk-filter --attack-return --hold-frames 4 --episodes 3 --task lunatic-straight-round-probe01-character01-lunatic --allow-task-transfer --transfer-reason "Frozen Normal reference; first Lunatic straight round-bullet probe with manual spell selection" --output runs/ppo-risk-zone24-preflight-lunatic-round-probe01.jsonl
```

`probe01` is an experimental label, not a decoded spell ID. Report the actual
spell name or selection-screen screenshot with the logs, and retain that mapping
before comparisons or training. Do not pool different spells under this label.
The CLI cannot select or verify difficulty. Assess losses, remaining HP,
omissions, emergency frequency and timing before changing capacity, hold length,
reward or training budget. Lunatic outcomes remain pending; laser acquisition
and geometry remain a separate unimplemented phase.

## Lunatic probe 1: user-identified stage-3 spell

The user identifies `lunatic-straight-round-probe01-character01-lunatic` as
stage 3, "Dazzling Color Typhon" (spelling preserved as reported), with many
small straight bullets emitted in different directions. This mapping is a
manual report, not a decoded native spell ID. The actual suffixed files are
`runs/ppo-risk-zone24-preflight-lunatic-round-probe01-001.jsonl` and its console.
Preserve the original generic header; the adjacent analysis and controller
registry store the task mapping.

The frozen model/controller completes 3/3 at 1319/1402/1371 frames, zero
deaths/dying, mean 22.733 seconds to result. All 2045 transitions replay exactly;
loaded intrinsic risk candidates/proposals match (max logit error 6.68e-6).
All execution windows and fresh risk memberships pass. Return ranking affects
49/1023 fresh choices; risk gating affects 96. There are no emergency fresh
choices, physical hold interruptions or snapshot race retries. Player x stays
within approximately 144.49..310.08 across the three trajectories.

Peak bullets=361, omitted=233; 1497/2045 post-step samples omit bullets. Only
nearest 128 are encoded. Omitted geometry is unavailable for retrospective
collision forecasts, so omission counts alone prove neither safety nor causation.
Selected fresh-action forecast clearance has minimum 5.712 px; this is a
12-frame linear geometric estimate, not survival probability. Frame deltas are
2 on 2041 substeps, 1 on two, and 4 on two. Decision time p95=2.143 ms and
observation age p95=3.061 ms (max 14.927 ms). Requested two-frame timing is not
an exact engine lock. No retained velocity component reaches its clipping bound.

End this spell's first probe. Select a different Lunatic round-bullet spell
with a different emission mechanism, such as aimed bursts or rotating volleys
whose individual bullets remain straight; exclude lasers for now. Keep the
model, capacity, gate, hold and equipment frozen. Report the actual spell name
with the results, and use a new probe label to avoid pooling tasks:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_control --resume runs/ppo-risk-zone24-preflight/checkpoints/update-000003 --boundary-mask --risk-filter --attack-return --hold-frames 4 --episodes 3 --task lunatic-round-probe02-character01-lunatic --allow-task-transfer --transfer-reason "Frozen reference; second manually identified Lunatic round-bullet spell with different emission mechanics" --output runs/ppo-risk-zone24-preflight-lunatic-round-probe02.jsonl
```

Do not relabel this checkpoint as trained on Lunatic or infer full difficulty
mastery. No extra training, capacity change or laser implementation is made.

## Lunatic probe 2: VividChaoticDance

The user identifies probe 2 as stage-3 "VividChaoticDance", under the declared
Lunatic evaluation. Preserve that spelling and the original generic task tag.
Its detailed emission mechanism has not been supplied; do not infer it from the
name. `runs/ppo-risk-zone24-preflight-lunatic-round-probe02.jsonl` completes
3/3 at 1095/1154/1195 frames, no deaths/dying, mean 19.133 seconds to result.
Resets show 1400 HP at (192,144), distinct from probe 1's 2000 HP; clear times
across these spells are not a policy-improvement comparison.

All 1723 transitions replay exactly and loaded risk candidates/proposals match
(max logit error 3.81e-6). All fresh risk memberships and modeled execution
windows pass. There are 862 fresh choices, including 10 emergency choices
(3/7/0), and 74 applied return choices (14/22/38). One physical hold interrupts
in episode 1 at frame 582: previous down-left would cross y=432 within the
remaining two frames; the controller starts a fresh up action, which is the only
risk candidate. No risk/return override bypasses safety or the hold contract.

Peak bullets=347, omitted=219; 905 post-step samples omit bullets. No radius
fallbacks or snapshot race retries occur. Delta frames are 2 on 1721 substeps
and 1 on two; decision p95=2.216 ms, observation age p95=3.062 ms, max age
12.666 ms. Fixed-state held actions lie outside newly computed risk sets on
97 substeps, as permitted when only physical bounds interrupt holding.

Two distinct Lunatic probes have now passed their first three episodes each.
Keep their results separate; neither general difficulty mastery nor learned
Lunatic adaptation follows. End this bounded round-bullet probe phase; do not
schedule another arbitrary spell merely to accumulate wins. Training stays
paused, with the existing model and controller references preserved.

The next engineering phase is laser observability: choose and identify one
laser spell, establish read-only slot/phase/width/angle/segment diagnostics on
the validated build, verify against visible activation, and test dangerous
geometry offline. Only then extend the observation/policy contracts and run
control evaluations. Existing circular-bullet observations cannot establish
laser safety. Reference-project laser utilities are clues requiring independent
verification; no production laser reader or native laser test is implemented
in this analysis-only step.
