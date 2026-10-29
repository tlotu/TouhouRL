# Spell result menu: static candidates

Scope: read-only analysis of the user-provided `th06nc.exe`, MD5
`b0cc28ec904e9efce2439ad857c938b5`. All addresses below are RVAs, not absolute
process addresses. Success, death-failure, and timeout-failure captures now agree
with the static interpretation. A three-episode live check also passed. Other builds must
not reuse these addresses without separate analysis.

## Reference leads and binary evidence

The reference TH06 reconstruction places in-game menus in AsciiManager, not Gui.
Its `StageMenu.hpp` and `AsciiManager.hpp` provide ownership leads only: NC has
different menu sizes and fields. The earlier GUI-region probes did not change
between success-menu selections, consistent with probing the wrong owner.

The exact NC binary contains these localization keys:

| Key | String RVA |
| --- | --- |
| EXE_SPELL_PRACTICE_RETURN_SPELL_SELECT | 0x307F78 |
| EXE_SPELL_PRACTICE_SAVE_REPLAY | 0x307FA0 |
| EXE_SPELL_PRACTICE_RETRY | 0x307FC0 |

Three 64-bit pointers at RVA 0x308090 reference these keys in order.
RVA 0xC190 initializes three text entries from that table. The routine starting
at 0xC230 calls it, dispatches on `[this+0xD94]` (range 0-5), and initializes
`[this+0xB44]` to 2 when `[this+0xD98]` is zero, otherwise 0:

```text
41 8B C6                  mov eax, r14d       ; r14d was set to 2
40 38 B1 98 0D 00 00      cmp [rcx+0xD98], sil ; esi was zeroed
0F 45 C6                  cmovne eax, esi
89 81 44 0B 00 00         mov [rcx+0xB44], eax
```

This matches the reported default selections, but does not alone verify an
outcome enum. Save-replay states, transitions, and stale values remain possible.

The update callback at 0x90D0 calls 0xC230 with `AsciiManager+0x6E10`, gated by
the byte at 0x4F27B6 (`cmp` at 0x9231, call at 0x9241). Callback registration
passes a candidate AsciiManager base of 0x3DE620 (`lea rsi` at 0x9428 followed by
storing RSI into callback data). This yields menu base 0x3E5430.

## Candidate fields

| Candidate | RVA | Type | Validation needed |
| --- | --- | --- | --- |
| Result menu active | 0x4F27B6 | u8 | Combat versus all result types |
| Selection | 0x3E5F74 | i32 | Default success -> Retry should read 0 -> 2 |
| Menu phase | 0x3E61C4 | i32 | Identify settled interactive phase |
| Outcome-related flag | 0x3E61C8 | u8 | Success versus death and timeout failure |

The probe reads menu-relative `[0xB40, 0xDA0)`, plus the active byte. It does not
scan the entire process or enable input based on these fields. Captures expose
`result_candidates` with `verified=false` for manual validation.

First repeat success_default and success_retry captures. Then capture
timeout_default and death_default; compare the outcome-related flag and phase.
Repeated trials are necessary before creating a runtime menu classifier.

## Success-screen live check

`runs/menu-success-default-002.jsonl` and
`runs/menu-success-retry-001.jsonl` each contain 20 readable samples:

| Manual state | active | selection | phase | outcome_flag |
| --- | --- | --- | --- | --- |
| Success, Return to spell card selection highlighted | 1 | 0 | 1 | 1 |
| Same success screen, Retry highlighted | 1 | 2 | 1 | 1 |

All 20 samples in each file agree. Region-relative offset 0x4 changed from 0 to
2, matching the statically derived selection field. Offset 0xD also differed;
it is not used as a cursor/outcome field because stable differences alone do
not establish meaning. No reads failed.

Do not repeat these two captures now. Next obtain death_default and
timeout_default while their default menu items are highlighted, plus a later
combat control if needed. The expected outcome_flag=0 on failures remains a
hypothesis in those earlier captures; the subsequent checks are recorded below.

## Failure cross-check and runtime enablement

`menu-death-default.jsonl` and `menu-timeout-default.jsonl` each contain 20
samples with active=1, selection=2, phase=1, outcome_flag=0. All agree, contrasting
with outcome_flag=1 on the user-confirmed success screen. The baseline profile
now enables runtime decoding through `spell_result_verified=true`.

Only active phase 1 yields an outcome. Inactive fields are ignored; selection
must be 0-2 and the outcome flag must be binary. Input requires explicit spell
retry mode, stable phase/selection, bounded navigation, and foreground focus.
The newer profile is not enabled.

## End-to-end live acceptance

`runs/result-retry.jsonl` records three completed episodes with reasons
spell_failure, spell_success, spell_success; three playable resets and two
inter-episode retries. There were no errors, unresolved endings, dropped bullets,
or radius fallbacks. Both success screens reported active=1, phase=1, selection=0,
outcome_flag=1; the failed screen reported selection=2, outcome_flag=0. All reset
records returned to active=false. The user observed correct retry navigation.

This validates the selected spell's baseline-build result/retry loop. Two wins in
three episodes are a smoke-test result, not a reliable general success-rate claim.
