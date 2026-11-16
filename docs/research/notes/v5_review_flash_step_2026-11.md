# Flash-Next's decode step after v4.0: a review for v5 (2026-11-16)

Where does a decode step of Qwen3.8-Flash-Next go on Miner Llama v4.0 (UkisAI's Swift-1.5 GSQ-RCO Q2_0 with its MTP
draft of three tokens, nine P106-090 cards), and which rows shorten it? The method is
[the third review's](flash_next_review3_2026-11.md):
- "est." marks my arithmetic or inference, with its basis. Every other number is measured, from the linked folder or
  a row of [quants/experiments.md](../../../quants/experiments.md), one run or load each.
- v4.0 is the build `v4-rc2` (`eb82916d3`) with [the release settings](../../../release/flash-next-v4/README.md). A
  step is one verify pass of the main model plus the draft's passes.
- Nothing ran for this note: file reads only.

## The short answer

- **30 tok/s is est. 9.45 and 13.51 ms a step away.** A coding agent's step takes est. 118.74 and 121.71 ms for
  3.2786 and 3.2461 tokens (27.61 and 26.67 tok/s); 30 tok/s needs est. 109.29 and 108.20.
- **The largest known piece is depth.** The step grows est. 11.32 and 11.22 ms in the first 4,096 tokens of context,
  and an agent turn runs at 3 to 10 thousand. Under 8,204 cached cells a four-token verify pass still runs the
  dense attention kernel.
- **Every agent reply is a first request, and a first request generates 444.11 ms longer** than its repeat at 4,096
  tokens, with the same 79 steps. Nothing has named that time on v4.
- **Longer drafts are not the road yet.** A fifth verified token costs est. 30.06 and 25.76 ms where est. 12.73 and
  16.23 break even, with the 5-to-8-token kernels already on.
- **v5 starts with one rig row:** the step's map on v4.0 (R121 with R164 merged, est. 2 h). No part profile exists
  on v4.0 at an agent's depth.

## Where the time goes now

**The step by depth** on v4.0, two loads each
([2026-11-13_r157_release_deep](../../data/2026-11-13_r157_release_deep/README.md)):

| Context, tokens | Repeated request, ms a step | First request at that depth |
|---:|---|---|
| 0 | 98.24, 98.27 | 102.55, 102.59 |
| 4,096 | 109.56, 109.49 | 115.18, 113.94 |
| 8,192 | 114.14, 114.01 | 118.28, 118.44 |

The agent's step (`steps.tsv` in
[2026-11-13_r163_agent_draft4](../../data/2026-11-13_r163_agent_draft4/README.md)) sits at the first request's step
at 8,192 tokens: est. 16.19 ms above depth 0's (118.74 − 102.55).

**The parts of a shallow step.** The newest split with a host profile is the host ladder's last rung
([2026-11-12_r122_host_ladder](../../data/2026-11-12_r122_host_ladder/README.md)), on an older v4 build
(`d097d6d88`), before five kernel changes of v4.0:

| Part | Two loads, ms |
|---|---|
| the verify pass: GPU time of the nine cards | 91.71, 91.64 |
| the verify pass: hops, rows back, the acceptance test | 3.57, 3.57 |
| the draft: its catch-up, three passes, the server's turn before the next verify (1.61, 1.81) | 16.43, 16.62 |
| the step | 111.89, 111.99 |

- **What grows with depth is the verify pass's GPU time.** In that rung's first load its median is 88.27 ms at 79
  to 335 tokens and 95.15 at 2,760: est. 6.88 ms more, on every card. The draft's span is 16.00 and 16.86
  (`tables/split_f.md`).
- **The dense attention is the likely cause** (est.): a four-token verify pass goes to the sparse kernel only from
  8,204 cells ([2026-11-11_c103_union_gather](../../data/2026-11-11_c103_union_gather/README.md)). The lane's point
  at 8,192 is already past that switch. With v3.0's preset on v4.0's build the dense kernel was 7.46 ms a step
  slower there ([2026-11-13_r88_union_deep](../../data/2026-11-13_r88_union_deep/README.md)), so between 4,096 and
  8,204 cells the step peaks above both lane points (est.).
- **The draft** is est. 14.7 % of that step: 12.10 ms on its card and 4.31 in hops to it (medians). A drafting pass
  takes 2.51 ms at the median and 6.27 at the 90th percentile on the greedy request: after a token outside its
  shortlist the draft runs its full head. In the agent runs 0.7595 and 0.7489 of its guesses are accepted.

**The verify pass by group**, four tokens, 79 to 335 tokens of context, the same older build
([2026-11-12_r120b_kern_speed](../../data/2026-11-12_r120b_kern_speed/README.md), load 2, request 1):

| Group | ms a pass | Its bytes at 173 GB/s, est. ([chapter 2](../../book/02-how-an-llm-makes-text.md)) |
|---|---:|---|
| attention and recurrent projections | 23.518 | 8.05 (1,392 MB) |
| routed experts | 20.453 | 15.35 (four tokens, 664 MB each) |
| hyper-connections, BF16 | 12.894 | 7.42 (1,283 MB) |
| recurrent state and conv | 8.922 | no weights |
| other operations without weights | 7.167 | no weights |
| shared expert, router, output layer, attention, the rest | 13.017 | |
| the pass | 85.970 | |

v4.0's pass is est. 83.94 ms: that run's 87.760 without its switches, less five measured gains (2.498, 0.494, 0.235,
0.420, 0.174). Last measured on v3.0: the recurrent state's rollback snapshots, 5.52 and 5.54 ms a verify pass
([2026-11-10_served_verify_parts](../../data/2026-11-10_served_verify_parts/README.md)).

**The floor.** One token reads 4,155 MB: 24.0 ms. v4.0's token takes est. 36.22 and 37.49 ms (the step over its
tokens), est. 12.22 and 13.49 above it. A four-token verify pass reads the dense weights once and four tokens'
experts: est. 6,147 MB, 35.53 ms. So est. 48.41 ms of the pass is above its own floor: 16.089 in operations that
read no weights, est. 15.47 in the projections.

## What v4 proved and what lost here

- **Gained:** eight host switches between two steps, 111.94 ms against 119.15 (R122's folder). Kernels, ms a verify
  pass: the 3-bit products −2.498, the shared expert's one launch −0.494, the 6-bit products −0.235 (R120b's
  folder); the 32-bit gate with a row limit −0.420
  ([2026-11-12_c807_f32_router](../../data/2026-11-12_c807_f32_router/README.md)); the shared expert's last product
  −0.174 ([2026-11-13_c755_shexp_downs](../../data/2026-11-13_c755_shexp_downs/README.md)).
- **Lost:** the K tiles, +0.307 and, narrowed, +0.115
  ([2026-11-12_c808_ktiles_gain](../../data/2026-11-12_c808_ktiles_gain/README.md)); the union attention kernel, a
  step 3.31 ms slower at 4,096 tokens and 16.98 at 8,192
  ([2026-11-13_r88_union_deep](../../data/2026-11-13_r88_union_deep/README.md)); the one launch at five to eight
  tokens, est. 0.207 ms at five and 1.352 at eight
  ([2026-11-13_c793_glu_5to8](../../data/2026-11-13_c793_glu_5to8/README.md)); the better file `gmix-q2`, +18.06 ms
  a step ([2026-11-12_r115_gmix_step](../../data/2026-11-12_r115_gmix_step/README.md)).
- **Drafts of four lost** (R163's folder): +0.4519 and +0.5299 tokens a step for est. +33.70 and +29.40 ms, where
  the tokens pay for est. 16.37 and 19.87. Less the extra drafting pass (est. 3.645 ms), the fifth verify column
  costs est. 30.06 and 25.76 ms and breaks even at est. 12.73 and 16.23. `_RO8` and C794's tiles were on, so the
  5-to-8-token kernels in the build do not get there; C833 would save est. 0.046 ms. Even a column of 8 ms gives
  est. 28.61 tok/s (3.7305 tokens over 118.74 + 8 + 3.645 ms), not 30.
- **The length chosen while drafting lost**
  ([2026-11-13_r154_agent_adapt](../../data/2026-11-13_r154_agent_adapt/README.md)): 4.049 tokens a step at est.
  160.3 ms against 3.29 at 117.4. Its cost table
  ([2026-11-12_fn82_adaptive_draft](../../data/2026-11-12_fn82_adaptive_draft/README.md)) put a fifth column at
  est. 21.242 ms and left out the snapshots, which grow with the draft's limit.
- **The draft did not change.** Swift's own draft layer is the one in use (Q1168), token trees lose to a plain
  chain, and the first guess is right 0.8481 of the time
  ([2026-11-11_fn80_fn81_draft_yield](../../data/2026-11-11_fn80_fn81_draft_yield/README.md)).

## Candidates for v5

"%" is of the 118.74 ms agent step.

| # | Row | What | Evidence | Worth | Cost | Needs | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | R121 (Q1393) + R164 (Q1484) | the step's parts on v4.0 at depth 0, 4,096 and 8,192, verify widths 1, 4 and 5 | none on v4.0 | prices the rows below | no code; rig est. 2 h | v4.0's build | first |
| 2 | C647 (Q1133) | why a first request generates slower than its repeat | measured: 444.11 ms at 4,096, the same 79 steps (R157, `requests.csv`) | est. 3.85 ms a step (over est. 115.5 steps a reply), 3.2 % | a trace in row 1's session; fix unknown | the step tracer | first |
| 3 | R69 (Q1252) | the sparse kernel's switch point for a four-token verify: ratios 0.75, 0.625 | measured: sparse 7.46 ms a step faster than dense at 8,192 (R88) | up to est. 7.46 ms just under 8,204 cells, 6.3 %; nothing at 4,096 | no code; rig 1.1 h | v4.0's build | first |
| 4 | C839 (Q1473), then C746 (Q1318) | the union kernel timed alone, then scoring only its own token's cells | measured loss; 22.7 % of a union's cells are masked for a token | up to est. 11.32 and 11.22 ms at 4,096, 9.5 % | small, then large; rig 2 h | row 1 naming the attention | next |
| 5 | C133 (Q686), C665 (Q1159) | replay instead of rollback snapshots | 5.52 and 5.54 ms a pass, on v3.0 | est. up to 4.6 %, less the replay after a rejection | large, or upstream's merge | row 1 | next |
| 6 | C618 (Q1103), Q636 | a draft shortlist that misses less | a drafting pass's means less its medians (R122) | est. 2.14 ms a step on the greedy request, 0.36 on the sampled one: 1.8 % and 0.3 % | small; rig 50 min | an agent run's replies | next |
| 7 | C624 (Q1109) | the next window started on the first cards while the last ones verify | its own est. is v3.0's; its offline gate never ran | unpriced on v4.0 | gate: CPU only; code large | the gate at 10 % or more | next (gate) |
| 8 | C425 (Q969), C687 (Q1210) | the draft's passes chained on its card | 1.51 ms of host turnaround a step, on v3.0 | est. up to 1.3 % | medium | C687's check | later |
| 9 | C308 (Q868), C144 (Q683) | the 16.089 ms without weights: a leaner state kernel, fused small operations | measured group; older estimates | unpriced on v4.0 | large | row 1 | later |
| 10 | C816 (Q1425), C795 (Q1390) | the fifth column: BF16, router and Q6_K at 5 to 8 tokens | est. 30.06 and 25.76 ms a column | est. +1.0 tok/s at 8 ms a column, 3.6 % | large | row 1's width 5 | later |
| 11 | C751 (Q1326) | the BF16 kernel's four-token excess | ceiling 1.96 ms a pass, on v3.0 | est. up to 1.7 % | medium | a kernel bench | later |
| 12 | R152 (Q1460), C829 (Q1459) | the K tiles at one block a round; their slow launches | est. −0.05 and −0.18 ms a pass | est. 0.2 % | none, small; rig 50 min | — | later |

## Rows to close, merge or re-rank

- **R121 (Q1393):** stale text (18.4 ms of growth, a 131.6 ms step: the pre-ladder line's). Re-scope to v4.0.
- **R164 (Q1484):** merge into R121: the same profile at widths 4 and 5.
- **C797 (Q1394):** its four-token arm is R163, its rule arm R154a: losses. Close the rest until a fifth column is
  under est. 12.73 ms.
- **C815 (Q1424), R154b, R147 (Q1453), C833 (Q1466):** park; each acts only with longer drafts.
- **C816 (Q1425):** stale estimate (21.242 ms); R163 puts the column higher.
- **R124 (Q1403), C800 (Q1399):** stale (the step at 49,152 is 137.42 and 137.52 ms now), and for the
  131,072-token line only. Park.
- **FN83 (Q1404):** no longer a gate: v4.0 ships both kernels. Last.
- **C647 (Q1133):** stale (v3.0, when the host swapped). Re-price on R157. **FN86 (Q1418)** merges into it.
- **C739 (Q1308), C743 (Q1312):** est. 0.003 and 0.009 ms a step: no run of their own.
- **C810 (Q1416):** superseded (est.): its one-second pause is the wait C811 and C817 removed.
- **C841 (Q1482):** not a step row: nothing for Flash-Next.
- **R88 part 2b, R136 parts 2 and 3, R145 (Q1450):** the prompt path, not the step.

## New ideas

- **The agent's step by depth, from logs that exist:** R163's `steps.sh` over R156's server logs, binned by a
  reply's context. No rig time.
- **The first request with the draft's overlapped catch-up off:** one deep-lane pair. If the 444.11 ms moves to
  prompt time, it is bookkeeping.
- **The shortlist's fallback at one step:** the draft runs its full head for two steps after a miss, and the
  preset's third shortlist value is 2 (est.: my reading, not checked in the code).
- **Snapshots sized to the step's own draft:** today to the draft's limit, so a rule that allows six may pay six.
- **A draft temperature for the accept rule, priced with the replay tool:** the draft is pessimistic where it is
  unsure (under 0.5, mean 0.2367, accepted 0.5283 of the time). Q800 lists the fit only.

Training a better draft is the sister project's (C247, C65).

## What I could not settle from the files

- What the est. 11.32 and 11.22 ms between depth 0 and 4,096 are, and the step between 4,096 and 8,204 cells. The
  dense attention is my inference.
- Whether the first request's 444.11 ms is one-off per reply, a slower step throughout, or the draft's catch-up
  counted as generation.
- A fifth column's cost by part: est. 30.06 and 25.76 ms are differences between agent runs.
- Whether the read-once kernels lower the draft's yield: v4.0's 0.7595 and 0.7489 are on other text.
- The est. 16.19 ms pairs an agent run with a deep lane from another rig.
