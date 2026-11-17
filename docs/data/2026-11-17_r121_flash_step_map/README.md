# Flash-Next's v4.0 decode step taken apart (R121; Q1393 with Q1484, and Q1133's trace; 2026-11-17)

What this measured: where a decode step of Qwen3.8-Flash-Next goes on Miner Llama v4.0 with the released preset, part
by part and kernel by kernel, at 0, 4,096 and 8,192 tokens of context, for verify passes of 1, 4 and 5 tokens. It also
traces why the first request at a depth generates slower than its repeat. Nine P106-090 cards, UkisAI's Swift-1.5
GSQ-RCO Q2_0 file with its MTP draft, [release/flash-next-v4/settings.env](../../../release/flash-next-v4/settings.env)
unchanged, plus the profilers' own variables. Registry rows Q1393 (the map), Q1484 (merged in: the four-to-five-token
growth) and Q1133 (the first request) in [quants/experiments.md](../../../quants/experiments.md). The plan this run
serves is [the step review](../../research/notes/v5_review_flash_step_2026-11.md).

Each value is one load. Two values in a cell are the arm's two loads, in run order.

## Result

### The step at depth, and what grows

The preset's step (verify passes of 4 tokens: three drafted plus one), the repeated request, ms
([map.md](map.md), table 1):

| Context | The step, no profiler (`w4u1`) | The step, host profile (S37's split, median) | Verify pass: GPU time of the nine cards | Verify pass: the rest (hops, rows back, acceptance) | The draft's span | its GPU time | The server's turn before the next verify |
|---:|---:|---|---|---|---|---|---|
| 0 | 98.62 | 99.86, 99.80 | 84.02, 84.03 | 3.59, 3.59 | 12.13, 12.15 | 8.33, 8.36 | 0.96, 0.95 |
| 4,096 | 109.85 | 111.91, 111.92 | 94.36, 94.35 | 3.98, 3.97 | 13.59, 13.58 | 9.39, 9.39 | 0.96, 0.96 |
| 8,192 | 114.84 | 116.01, 116.16 | 97.43, 97.57 | 4.40, 4.43 | 14.17, 14.17 | 9.55, 9.52 | 0.97, 0.97 |

The unprofiled step is gen_ms over the reply's steps (256 tokens less the accepted drafts); R157 measured 98.24 and
98.27, 109.56 and 109.49, 114.14 and 114.01 on the same build and rig four days earlier
([2026-11-13_r157_release_deep](../2026-11-13_r157_release_deep/README.md)).

- **From depth 0 to 4,096 the step grows 12.05 and 12.12 ms:** the verify pass's GPU time 10.34 and 10.32, the draft
  1.46 and 1.43, the verify pass's host part 0.39 and 0.38. To 8,192: 16.15 and 16.36, of which the verify pass's GPU
  time is 13.41 and 13.54.
- **Kernel by kernel, the growth is the attention** (the group profile, all nine cards, ms a verify pass,
  [map.md](map.md) table 4a):

  | Launch group | Depth 0 | 4,096 | 8,192 | 4,096 − 0 | 8,192 − 0 |
  |---|---|---|---|---|---|
  | `FLASH_ATTN_EXT` (12 attention layers) | 0.97, 0.97 | 10.15, 10.15 | 12.66, 12.65 | 9.18, 9.18 | 11.68, 11.68 |
  | `LIGHTNING_INDEXER` (the QSA indexer's scores) | 0.12, 0.12 | 0.62, 0.62 | 1.15, 1.15 | 0.50, 0.50 | 1.03, 1.04 |
  | `TOP_K indexer_top_k` | 0.13, 0.14 | 0.53, 0.53 | 0.56, 0.55 | 0.40, 0.40 | 0.42, 0.42 |
  | every other launch group | 82.00, 81.96 | 82.12, 82.18 | 82.43, 82.49 | 0.12, 0.22 | 0.43, 0.53 |
  | **the verify pass** | 83.22, 83.19 | 93.42, 93.48 | 96.80, 96.84 | 10.20, 10.29 | 13.58, 13.66 |

  The attention is 9.18 and 9.18 ms of the verify pass's 10.20 and 10.29 ms of growth to 4,096 tokens, and 11.68 and
  11.68 of its 13.58 and 13.66 to 8,192. A four-token verify takes the sparse kernel only from 8,204 cells (the
  switch: `_MIN_RATIO` × about 2,051 kept cells per query token), so at 4,096 it runs the
  dense kernel over every cell, and at 8,192 the sparse one over the kept cells (est.: the switch rule, and the
  attention's flat cost of one-token passes, which take the sparse kernel from 2,051 cells: 2.90 and 2.90 ms at
  4,096, 2.91 and 2.93 at 8,192). The dense kernel at four tokens grows est. 2.24 ms a pass per 1,000 cells
  ((10.15 − 0.97) / 4.096), so it passes the sparse kernel's 12.66 ms at est. 5,220 cells and costs est. 19.3 ms at
  8,203: est. up to 6.7 ms a pass between 5,220 and 8,204 cells.
- **The draft grows too:** a drafting pass takes 2.49, 2.41 and 2.43 ms on its card at depth 0, 2.80, 2.80 and 2.79
  at 4,096, 2.87, 2.83 and 2.85 at 8,192 (the median of each pass, load `w4h1`), and the catch-up's turn 1.04, 1.40
  and 1.77 ms.

### The verify pass by part and width

GPU time of a verify pass by R41's parts, all nine cards, ms, the repeated request ([map.md](map.md), table 2):

| Part | 1 token, depth 0 | 4 tokens, depth 0 | 5 tokens, depth 0 | 4 tokens, 4,096 | 5 tokens, 4,096 | 4 tokens, 8,192 | 5 tokens, 8,192 |
|---|---|---|---|---|---|---|---|
| attention and recurrent projections | 15.73, 15.77 | 23.27, 23.22 | 36.19, 36.21 | 23.33, 23.34 | 36.10, 36.10 | 23.40, 23.34 | 36.32, 36.34 |
| routed experts | 7.17, 7.19 | 19.98, 19.97 | 24.17, 24.17 | 20.00, 20.00 | 24.49, 24.47 | 20.20, 20.22 | 24.66, 24.67 |
| hyper-connections (BF16) | 9.59, 9.66 | 12.83, 12.85 | 18.39, 18.38 | 12.82, 12.85 | 18.46, 18.45 | 12.83, 12.85 | 18.45, 18.46 |
| Gated DeltaNet state and conv | 2.65, 2.65 | 8.91, 8.91 | 11.01, 11.01 | 9.00, 8.99 | 11.06, 11.04 | 8.98, 9.02 | 11.03, 11.01 |
| other operations without weights | 5.19, 5.14 | 7.25, 7.17 | 7.85, 7.85 | 8.03, 8.02 | 8.90, 8.79 | 8.62, 8.57 | 9.51, 9.59 |
| attention (`FLASH_ATTN_EXT`) | 0.59, 0.58 | 0.97, 0.97 | 1.48, 1.48 | 10.15, 10.15 | 14.60, 14.61 | 12.66, 12.65 | 27.55, 27.56 |
| shared expert | 2.67, 2.64 | 3.58, 3.60 | 4.51, 4.49 | 3.65, 3.65 | 4.53, 4.53 | 3.66, 3.67 | 4.55, 4.51 |
| output layer | 2.86, 2.86 | 3.10, 3.12 | 8.72, 8.73 | 3.11, 3.12 | 8.71, 8.69 | 3.11, 3.11 | 8.71, 8.71 |
| router | 1.82, 1.83 | 2.44, 2.43 | 4.67, 4.67 | 2.42, 2.42 | 4.69, 4.68 | 2.43, 2.44 | 4.67, 4.67 |
| other weights | 0.57, 0.57 | 0.90, 0.94 | 1.10, 1.10 | 0.91, 0.95 | 1.12, 1.12 | 0.91, 0.97 | 1.11, 1.10 |
| **the pass** | **48.84, 48.90** | **83.22, 83.19** | **118.08, 118.09** | **93.42, 93.48** | **132.67, 132.49** | **96.80, 96.84** | **146.56, 146.62** |

One token at 4,096 and 8,192: 51.52 and 51.58, 51.69 and 51.75 ([map.md](map.md)). On card 8 the head's product
takes 3.10 and 3.12 ms at four tokens (the read-once Q5_K kernel, `_Q5K_MT`; 8.05 on v3.0, R49) and 8.72 and 8.73 at
five; the sampling chain after it 0.20, 0.91 and 1.06 ms at one, four and five tokens.

**What a fifth token costs** ([map.md](map.md) table 4b, loads paired `w4g1` → `w5g1` and `w4g2` → `w5g2`), ms a
verify pass at depth 0:

| Launch group | 4 tokens | 5 tokens | 5 − 4 | Why (est., from the switches' ranges in [docs/switches.md](../../switches.md)) |
|---|---|---|---|---|
| `output` Q5_K 2,560 × 248,320 | 3.10, 3.12 | 8.72, 8.73 | 5.62, 5.61 | `_Q5K_MT` reads the head once for 2-4 tokens only |
| routed experts' gate, up and SwiGLU (Q2_0) | 11.08, 11.07 | 13.60, 13.60 | 2.51, 2.53 | one more token's experts |
| `ssm_out` IQ4_XS | 1.89, 1.89 | 4.39, 4.39 | 2.49, 2.50 | `_DMT`: IQ4_XS and Q4_K read once for 2-4 tokens only |
| `attn_qkv` IQ4_XS | 1.85, 1.84 | 4.27, 4.27 | 2.41, 2.43 | the same |
| the router `ffn_gate_inp` F32 2,560 × 512 | 2.43, 2.43 | 4.67, 4.67 | 2.23, 2.23 | cuBLAS at both widths (C807's rule keeps the router there) |
| `GATED_DELTA_NET` | 8.91, 8.91 | 11.01, 11.01 | 2.10, 2.10 | a fourth rollback snapshot of the recurrent state |
| the four BF16 hyper-connection products | 10.30, 10.33 | 15.10, 15.10 | 4.80, 4.77 | `hc_ffn_down` +1.43, `hc_attn_down` +1.39, `hc_attn_up` +1.01, `hc_ffn_up` +0.97 (load 1) |
| `ssm_out` Q4_K | 1.55, 1.54 | 2.94, 2.94 | 1.39, 1.39 | `_DMT`, 2-4 tokens |
| routed experts' down (Q2_0) | 7.15, 7.16 | 8.45, 8.45 | 1.30, 1.29 | one more token's experts |
| **the pass** | 83.22, 83.19 | 118.08, 118.09 | **34.85, 34.90** | |

At 4,096 tokens the fifth token adds 39.24 and 39.01 ms a pass, the attention 4.46 and 4.46 of it; at 8,192 49.76
and 49.78, the attention 14.89 and 14.90: a five-token verify takes the sparse kernel only from 10,255 cells, so at
8,192 it still reads every cell (est., the switch rule). The step (S37's split) grows 39.18 and 39.42 ms at depth 0,
43.54 and 43.28 at 4,096 and 54.70 and 54.27 at 8,192; the draft's fourth pass is 3.64 and 3.65 of it at depth 0.

On this greedy reply drafts of four bring 3.765 tokens a step against 3.241 at 4,096 tokens (+0.524): at the
four-token step's 111.91 ms for 3.241 tokens, a fifth column breaks even at est. 18.1 ms a step, against the 43.54 and
43.28 it costs. Width 5 is slower at every depth in this run: 155.45 and 155.20 ms for 3.765 tokens (24.2 tok/s)
against 111.91 and 111.92 for 3.241 (29.0 tok/s) at 4,096 (est. from the split's medians).

**From one token to four** (table 4c), +34.38 and +34.28 ms a pass at depth 0: the routed experts' gate and up +7.47
and +7.45, `GATED_DELTA_NET` +6.26 and +6.26 (2.65 at one token, 8.91 at four, 11.01 at five: the drafted tokens'
snapshots), the experts' down +4.15 and +4.14, the Q3_K `attn_qkv` +1.37 and +1.36.

### The first request at a depth (Q1133)

**Today's first request at 4,096 tokens generates 310.85 ms slower than its repeat with no profiler** (`w4u1`:
8,989.20 against 8,678.35 ms, the same 231 drafted and 177 accepted tokens, 79 steps: 3.93 ms a step). With the host
profile and step trace it is 306.31 and 327.19 ms; R157 measured 444.11 and 351.48. The step trace
(`LLAMA_STEP_TRACE`) and host profile put it in three places ([first/first-w4h1.md](first/first-w4h1.md),
[first/first-w4h2.md](first/first-w4h2.md); pass 1 − pass 2, ms, over the 78 paired generation steps):

| Where | `w4h1` | `w4h2` |
|---|---:|---:|
| the first verify pass after the prompt | +106.64 | +123.78 |
| the reply's last pass (one token, no draft) | +129.0 | +127.4 |
| the 76 steps between | +68.24 | +73.57 |
| **all paired steps** (the server's gen_ms) | **+303.88** (+306.31) | **+324.75** (+327.19) |
| of which the verify passes | +256.47 | +276.95 |
| of which the main passes' GPU time (host profile, nine cards) | +202.83 | +225.46 |
| of which the host's queuing of the verify pass's graphs (`v.ub.in` → `v.ub.comp`) | +60.20 | +54.97 |
| of which the draft's passes (their GPU time) | +48.65 (+35.66) | +49.08 (+34.95) |
| of which the draft's catch-up | −1.41 | −0.93 |

- **The two slow passes take about twice as long on every card.** At 4,096 (`w4h1`), the first verify pass after the
  prompt takes 26.4, 27.6, 25.6, 26.9, 20.5, 27.1, 23.3, 25.3 and 17.3 ms on cards 0 to 8 (219.9 in all) against
  pass 2's 16.0, 15.6, 13.4, 15.5, 12.6, 13.5, 12.7, 13.3 and 9.0 (121.7), and its last pass 186.2 against 61.8. A
  verify pass soon after takes 13.6, 12.0, 11.2, 12.3, 9.4, 10.9, 10.4, 10.7 and 8.7 ms (99.2: pass 1's second).
- **Without the draft the cost is one pass.** At one token a step, pass 1's first decode after its 64-token prompt
  takes +124.91 and +128.39 ms at 4,096 tokens and +116.39 and +117.64 at 8,192. The steps after it match the
  repeat's in their median difference, 0.01 and 0.08 ms a step at 4,096; the second load also has one step of
  +95.3 ms (step 122) ([first/first-w1h1.md](first/first-w1h1.md), [first/first-w1h2.md](first/first-w1h2.md)). At
  five tokens a step: +132.26 and +132.33 at the first verify pass, and the last pass is not slower (−16.8 and −11.5).
- **Kernel by kernel, a slow pass is slow everywhere.** Under the group profile, which synchronizes after every
  evaluation, width 1's slow pass after the prompt (the second pass at 4,096) takes +107.69 and +106.15 ms, with 137
  of its 188 launch-group labels over 1.2 times the repeat's (small operations such as `FILL`, `SCALE`, `SILU` and
  `RMS_NORM` take 2.6 to 3.8 ms a pass instead of 0.03 to 0.3). Width 4's last pass takes +96.79 and +96.79 ms, with
  113 and 111 of 188 labels over 1.2 times ([first/passes-w1g1-d4096.md](first/passes-w1g1-d4096.md),
  [first/passes-w4g1-d4096.md](first/passes-w4g1-d4096.md)). Under that profiler width 4's first verify pass is not
  slow (−13.12 and −12.11 ms).
- **What it is not:** not the draft's catch-up (−1.41 and −0.93 ms), not a graph built again on the host (the main
  graph is built once in the generation of both requests), and not short host memory (the lowest MemAvailable of the
  width-4 loads 292, 309, 304, 283 and 319 MiB; no watchdog stop).
- **What it most likely is (est., from v4.0's code):** a pass whose graph properties are new runs its kernels one
  launch at a time; ggml captures a CUDA graph only after the warm-up of unchanged calls, and
  `GGML_CUDA_GRAPHS_FORCE` skips that warm-up only for graphs of more than 8 tokens (prompt micro-batches). The first
  pass of a width after the prompt, and the reply's shorter last pass, are new at each cache size; the repeat finds
  the graphs the first request left at the same cache size. A spread of every launch, not one kernel, is what eager
  launches look like. The check is a load with `GGML_CUDA_GRAPH_DEBUG=1` (R192).
- **What it costs an agent:** every reply is a first request at a new cache size, so est. 3.93 ms a step at 4,096
  (310.85 ms over 79 steps), 3.3 % of the agent's 118.74 ms step.

### What the map prices

| Row | What it changes | Before this run | Now (ms a step unless said) |
|---|---|---|---|
| C647 (Q1133) | the first request's cost | est. 3.85 (444.11 ms, R157) | 3.93 at 4,096 (310.85 ms, today, no profiler): two passes run kernel by kernel (est.) and 0.9 a step; the code row is C867 (Q1561) |
| C867 (Q1561), new | decode and verify graphs captured or patched at a new cache size | none | up to est. 3.93 (3.3 %) |
| C839, C746 (Q1473, Q1318) | the attention at depth | "row 1 naming the attention" | named: 9.18 of the verify pass's 10.20 ms of growth to 4,096 (the step grows 12.05), 11.68 of 13.58 to 8,192; ceiling 9.18 a pass at 4,096 |
| R69 (Q1252) | where the sparse kernel takes a four-token verify | up to 7.46 near 8,204 cells | est. crossover 5,220 cells; est. up to 6.7 a pass between 5,220 and 8,204 cells, nothing at 4,096 (dense 10.15 < sparse est. 12.66). The same ratio moves five-token passes: 14.89 a pass at 8,192 |
| C133, C665, C847 (Q686, Q1159, Q1508) | replay instead of rollback snapshots | 5.52 and 5.54 a pass (v3.0) | `GATED_DELTA_NET` 8.91 a pass at 4 tokens, 2.65 at 1, 11.01 at 5 (2.10 a snapshot); dearer than v3.0's figure |
| C816, C795 (Q1425, Q1390) | the fifth column | est. 30.06 and 25.76 (agent runs) | measured +34.85 and +34.90 a verify pass at depth 0, +39.24 and +39.01 at 4,096, +49.76 and +49.78 at 8,192; break-even est. 18.1 at 4,096. Kernels to reach first: the head (5.62), IQ4_XS (4.90), BF16 (4.80), routed (3.81), router (2.23), the snapshot (2.10), Q4_K (1.39), and the attention's switch at 5 tokens (14.89 at 8,192) |
| C751 (Q1326) | the BF16 kernel's four-token excess | ceiling 1.96 (v3.0) | the hyper-connections take 12.83 and 12.85 at 4 tokens against 9.59 and 9.66 at 1: 3.24 and 3.19 (dearer) |
| C425, C687 (Q969, Q1210) | the draft's passes chained on its card | 1.51 of host turnaround (v3.0) | the turns between drafting passes 0.93 and 0.89 at 4,096 (`w4h1`, medians): est. up to 1.82, plus the catch-up's turn 1.40 |
| C308, C144 (Q868, Q683) | operations without weights | 16.089 (R120b) | other operations without weights 7.25 and 7.17 at depth 0, 8.62 and 8.57 at 8,192, plus `GATED_DELTA_NET`'s 8.91 |
| FN110 (Q1535) | plain host arithmetic | needs 5 ms or more | the verify pass's host part 3.59 and 3.59 at depth 0, 4.40 and 4.43 at 8,192, and the server's turn 0.96: no 5 ms of host arithmetic; stays parked |

## What ran

- **Rig and build:** a nine-card P106-090 rig (Manli M-P106L9-N6G, the stock 4 GB host), nothing else running;
  Miner Llama v4.0 (`pascal-p106-v4` `eb82916d3`, the tag `miner-llama-v4.0`; the rig build's `llama-server` reports
  `eb312b7bc`, branch `pascal-p106-v4-b12360-c828`, whose tree is v4.0's), md5 of `llama-server`
  `e7afd26db01f4c671549ef12c504ada3`.
- **Model and line:** UkisAI's Swift-1.5 GSQ-RCO Q2_0 of Qwen3.8-Flash-Next (`PRESET_MODEL`, the n-gram table in the
  third shard) with the MTP draft, the v4 preset exactly ([loads.tsv](loads.tsv) has each load's line).
- **The driver:** [bench/deep_lane](../../../bench/deep_lane/README.md) through [run/r121.sh](run/r121.sh), one
  detached run, 10:39:21 to 12:28:55. Per depth (0, 4,096, 8,192) a load sends a prime (the prompt without its 64-token
  tail), decode pass 1 (the first request at the depth: 256 tokens at temperature 0, end of text ignored), decode
  pass 2 (the same request again: it restores a checkpoint four tokens before the end) and a sampled pass (the
  preset's sampler, seed 1). The client waits 4 s before every request (`PAUSE=4`), so each profile cuts into one piece
  per request.
- **The arms, each its own load:**

  | Arm | Width | Profiler (the variables added to the preset's) |
  |---|---|---|
  | `w4u` | 4 (the preset) | none: the warm-up load (depth 0 only, untimed) and the last load `w4u1` |
  | `w4h1`, `w4h2` | 4 | `GGML_CUDA_PROFILE_HOST` (each card's host and GPU time per evaluation, nothing synchronized) and `LLAMA_STEP_TRACE` (the server's, drafter's and contexts' host marks per step) |
  | `w4g1`, `w4g2` | 4 | `GGML_CUDA_PROFILE` with `GGML_CUDA_PROFILE_MODE=group` (every launch group timed with events, the stream synchronized after every evaluation) |
  | `w5h1`, `w5h2`, `w5g1`, `w5g2` | 5 | the same, with `--spec-draft-n-max 4` |
  | `w1h1`, `w1h2`, `w1g1`, `w1g2` | 1 | the same, without the MTP draft (the deep lane's `DRAFT=off` drops `-md` and every `--spec-*` flag) |

  Order: `warm`, `w4h1 w4g1 w5h1 w5g1 w1h1 w1g1`, the same six again, `w4u1`. Before every load: no llama process and at
  most 100 MiB on any card; every load under the RAM watchdog (stop below 60 MiB available), which stopped none.
- **The replies:** every load of a width wrote the same reply at each depth and pass (one md5 per width, depth and
  pass in [requests.csv](requests.csv)), so the profilers changed no token. Width 1 writes another greedy reply than
  width 4 and 5.
- **Host memory, lowest MemAvailable per load:** 292 and 309 MiB (`w4h`), 304 and 283 (`w4g`), 319 (`w4u1`), 289 and
  293 (`w5h`), 279 and 270 (`w5g`), 688 and 688 (`w1h`), 673 and 665 (`w1g`).
- **What the profilers cost:** the host profile with the trace adds 0.70 and 0.65 ms a step at depth 0, 0.90 and 0.49
  at 4,096, −0.13 and 0.07 at 8,192 against `w4u1` (server ms a step, the repeated request); the group profile 31.86
  and 30.59 at depth 0. Its GPU times per launch group are what this folder uses from it.

## How it was read

On the rig's CPU after the run, at `nice -n 19`, with no model loaded ([run/tables.sh](run/tables.sh)):

1. [run/split_trace.py](run/split_trace.py) cuts each profile file into one file per generation request, streaming
   (a pause of more than 3 s cuts; a piece with at least 45 main passes is a generation request), so a reader holds one
   request in memory on the 4 GB host.
2. Per host-profile request, S37's
   [step_split.py](../2026-11-10_served_step_split/run/step_split.py): the step, the verify pass's GPU time and the
   rest, each draft pass's lead, GPU time and turn, over the typical steps.
3. Per group-profile request, R49's [verify_parts.py](../2026-11-10_served_verify_parts/run/verify_parts.py) with
   R41's parts ([kpass_parts.py](../2026-11-04_kpass_profile/run/kpass_parts.py), 7.6 µs subtracted per launch
   group): the typical verify passes by part, bucket and label.
4. The first request: R86's [align_t0.py](../2026-11-11_r86_step_trace/run/align_t0.py) puts each host profile on the
   step trace's clock; [run/first_request.py](run/first_request.py) pairs pass 1's generation steps with pass 2's, step
   by step, by stage and by mark pair, with the main passes' and the draft's GPU time per step; C405's
   [step_gap.py](../2026-11-11_c405_server_gap/run/step_gap.py) gives each load's waterfall;
   [run/first_passes.py](run/first_passes.py) compares pass 1's first, early, typical and last passes with pass 2's,
   launch group by launch group, from the group profiles.
5. [run/step_map.py](run/step_map.py) joins them per width, depth and request ([map.md](map.md): the repeated request;
   [map_p1.md](map_p1.md): the first; [map_s.md](map_s.md): the sampled).

## Commands

```bash
# from the repository root on a machine that runs only ssh and tar; the rig's paths in the environment
RIG=<rig alias> WORK=<run folder on the rig> BIN=<v4.0 build on the rig>/bin/llama-server \
  MODEL=<models>/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf \
  MODEL_DRAFT=<models>/mtp-Qwen3.8-Flash-Next-draft.gguf \
  TEXT_LOCAL=<the release check's proxy.jsonl.bodies.gz> DRY=1 STEP=start \
  bash docs/data/2026-11-17_r121_flash_step_map/run/r121.sh
# then the same without DRY=1, then STEP=collect OUT=<folder> (about 1 h 50 min on the rig)

# the tables, on any machine with Python 3 (here the rig's CPU, the rig idle): S this repository, OUT the run's
# results (requests.csv, requests-<load>.csv, hp-*.csv, st-*.jsonl, gp-*.csv), RES the tables (about 23 minutes)
S=<this repository> OUT=<results> RES=<tables> bash docs/data/2026-11-17_r121_flash_step_map/run/tables.sh
```

The six group-profile traces (1.07 GB together) are not published. Their sizes and md5s:

| Trace | Bytes | md5 |
|---|---:|---|
| `gp-w1g1.csv` | 310,532,773 | `1de53e0d1a8c67909e50975ce804f74a` |
| `gp-w1g2.csv` | 310,547,279 | `25a0c6a84bb3a6ac148e42153f598d47` |
| `gp-w4g1.csv` | 120,832,069 | `0d2481b476ccaf2f929c9a0ad4edfab9` |
| `gp-w4g2.csv` | 120,832,009 | `7cb94baeba5282a60032b6b89ca116b2` |
| `gp-w5g1.csv` | 107,765,458 | `ea48cfb9ce8c921e06ae8b95d61af5de` |
| `gp-w5g2.csv` | 107,763,196 | `2a4fcaa9a5f0564c4562f95183ed5c78` |

Every table here comes from them, or from the host profiles and step traces, through the scripts above.

## Files

| File | What it holds |
|---|---|
| [map.md](map.md), [map_p1.md](map_p1.md), [map_s.md](map_s.md) | `step_map.py`: the step, the verify pass by part and bucket per width and depth, and what grows with depth and width by label, each load by value, for the repeated, the first and the sampled request |
| `map.tsv.gz`, `map_p1.tsv.gz`, `map_s.tsv.gz` | every value of those tables in long form: table, width, depth, request, load, name, value |
| [requests.csv](requests.csv) | every request of every load: prompt and generation times, drafted and accepted tokens, the reply's md5 |
| [loads.tsv](loads.tsv) | every load's environment and server line as it ran (paths as `<OUT>`, `<MODELS>`, `<BIN>`) |
| [table.md](table.md), [deep_classes.tsv](deep_classes.tsv) | the deep lane's own tables of the run |
| `first/first-<load>.md`, `first/steps-<load>.tsv`, `first/pairs-<load>.tsv` | `first_request.py` per host-profile load: requests, pass 1 against pass 2 by stage and mark pair, every generation step's stages, the pairs' sums |
| `first/passes-<load>-d<depth>.md` (and `.tsv`) | `first_passes.py` per group-profile load and depth: every main pass's GPU time, the first, early, typical and last pass by label and per card |
| `first/gap-<load>.md`, `first/align-<load>.md` | `step_gap.py`'s waterfall and `align_t0.py`'s clock offset per host-profile load |
| `split/<load>.tsv`, `split/<load>-<request>.md` | `step_split.py`: every generation step's parts, and the tables per request |
| `verify/parts.tsv`, `verify/accounting.tsv`, `verify/per_card.tsv`, `verify/cut.tsv`, `verify/labels.tsv.gz`, `verify/last_card_groups.tsv.gz`, `verify/passes.tsv.gz` | `verify_parts.py`'s tables for every group-profile load and request, joined with a load and a request column |
| [run/r121.sh](run/r121.sh) | the run: the arms on the deep lane, start, wait, status, collect |
| [run/tables.sh](run/tables.sh), [run/split_trace.py](run/split_trace.py), [run/first_request.py](run/first_request.py), [run/first_passes.py](run/first_passes.py), [run/step_map.py](run/step_map.py) | the reading |

## Limits

- **One reply per width and depth, greedy.** The drafts' acceptance and so the step's draft passes are this reply's;
  the v4.0 agent runs accept 0.7595 and 0.7489 of the draft (the step review), this one 177 of 231 at 4,096 (0.766).
- **The group profile synchronizes after every evaluation**, so its passes run back to back with the host idle: its
  GPU times by launch group are this folder's kernel figures, its step times are not the preset's (31.86 and 30.59 ms
  a step slower at depth 0).
- **Which kernel `FLASH_ATTN_EXT` ran** (dense or sparse) is not in the trace's labels; the split above follows the
  switch's rule.
- **The cause of the first request's slow passes is inferred**, from where the time sits (every launch of a pass, on
  every card) and from v4.0's code; R192's debug load counts the eager, captured and replayed evaluations.

## Provenance and licence

Code: Miner Llama (MIT, like llama.cpp), v4.0. Model: Swift-1.5 GSQ-RCO Qwen3.8-Flash-Next Q2_0
([ukisai](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF), Swift Open License 1.0 and
qwen-community-1.0) with its MTP draft file; the prompts are the deep lane's recorded agent text
([bench/deep_lane](../../../bench/deep_lane/README.md)). The tables hold times, counts, kernel labels and md5s, no
text. Measured on a Manli M-P106L9-N6G (nine P106-090).
