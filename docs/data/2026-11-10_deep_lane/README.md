# The deep lane on the rig: the sparse Pascal attention off and on, and the MTP draft on and off, at 0 to 49,152 tokens of context (Q1143, Q230, Q1153; 2026-11-10)

What this is: the first timing of the sparse Pascal attention (`GGML_CUDA_FATTN_SPARSE_PASCAL=1`, off by default) on
a real model. The line is the Miner Llama v3 release preset
([release/flash-next-v3/settings.env](../../../release/flash-next-v3/settings.env): Qwen3.8-Flash-Next, Swift Q2_0, the
MTP draft of three tokens, a 65,536-token context) on nine P106-090 cards, build `pascal-p106-v3-b12076-ladder5`
`1551f5fe1`. [bench/deep_lane](../../../bench/deep_lane/README.md) holds the context at 0, 8,192, 16,384, 32,768 and
49,152 tokens and generates 256 tokens after each, with the switch off (`dense`) and on (`sparse`), two server loads
an arm. Registry rows Q1143 (the lane) and Q230 (the kernel) in
[quants/experiments.md](../../../quants/experiments.md).

A second run on the same day put the line with its MTP draft against the same line without it, at the same depths
and with the sparse attention on in both arms (`draft` and `nodraft`; registry row Q1153). Its result is
[the second pair](#the-second-pair-the-mtp-draft-on-against-off-q1153) below.

Why: on a P106 llama.cpp's attention reads the whole cache for every token, so generation slows as an agent's
context grows, and real coding attempts write 40,000 tokens and more. The model's attention layers keep about 2,051
cells per query token; the sparse kernel reads only those. It had passed its cases natively on a P106
([2026-11-10_replay_checks_rig2](../2026-11-10_replay_checks_rig2/README.md)) and written the dense kernels' text on
the whole model on newer cards ([2026-11-09_sparse_attn_whole_model](../2026-11-09_sparse_attn_whole_model/README.md)),
and had never been timed.

These are single-request diagnostics at a known depth. They do not replace the agent check.

## Result: the sparse attention off and on

**The sparse attention is a gain from 16,384 tokens of context on in tokens a second, and from 8,192 on in the time
of a step. With it a step takes 150.40 ms at 8,192 tokens, 149.73 at 16,384, 165.86 at 32,768 and 180.77 at 49,152
(load `sparse1`, the repeated request); with the dense kernel 153.00, 182.25, 241.65 and 301.90 (load `dense1`). At
49,152 tokens the line generates 17.81 and 14.68 tok/s with it against 10.59 and 11.54 without. At depth 0 and at
8,192 its tok/s is negligible. Prompt reading does not change.**

### Generation: tokens a second and ms a token

Per load: pass 1, pass 2 → the load (its two requests' tokens over their time), then the load's ms a token. Pass 1
is the first request at a new depth, as every turn of an agent is; pass 2 is the same request again. Spread: the
difference between an arm's two loads. Class of `sparse` against `dense`: gain when its mean is above `dense`'s by
more than the larger spread, loss when below by more than it, else negligible
([dense_sparse/table.md](dense_sparse/table.md), [dense_sparse/classes.tsv](dense_sparse/classes.tsv)).

| Depth (tokens) | Arm | Load 1, tok/s | ms a token | Load 2, tok/s | ms a token | Spread, tok/s | `sparse` against `dense` | | Class |
|---:|---|---|---:|---|---:|---:|---:|---:|---|
| 0 | dense | 26.38, 27.70 → **27.03** | 37.00 | 25.77, 27.70 → **26.70** | 37.46 | 0.33 | | | |
| | sparse | 25.81, 27.67 → **26.71** | 37.44 | 25.78, 27.67 → **26.69** | 37.47 | 0.02 | −0.16 tok/s | −0.6 % | negligible |
| 8,192 | dense | 14.69, 19.68 → **16.82** | 59.44 | 16.63, 24.98 → **19.97** | 50.08 | 3.14 | | | |
| | sparse | 17.29, 25.40 → **20.58** | 48.60 | 17.50, 25.42 → **20.73** | 48.25 | 0.15 | +2.25 tok/s | +12.3 % | negligible |
| 16,384 | dense | 15.33, 19.51 → **17.17** | 58.25 | 17.45, 19.70 → **18.51** | 54.03 | 1.34 | | | |
| | sparse | 18.19, 23.75 → **20.60** | 48.54 | 18.51, 24.25 → **20.99** | 47.64 | 0.39 | +2.96 tok/s | +16.6 % | gain |
| 32,768 | dense | 9.26, 10.39 → **9.79** | 102.12 | 10.44, 11.35 → **10.88** | 91.94 | 1.08 | | | |
| | sparse | 14.79, 19.29 → **16.75** | 59.71 | 17.07, 19.31 → **18.12** | 55.18 | 1.38 | +7.10 tok/s | +68.7 % | gain |
| 49,152 | dense | 10.45, 10.73 → **10.59** | 94.45 | 11.15, 11.96 → **11.54** | 86.64 | 0.95 | | | |
| | sparse | 17.06, 18.63 → **17.81** | 56.13 | 13.91, 15.55 → **14.68** | 68.11 | 3.13 | +5.18 tok/s | +46.8 % | gain |
| all five | dense | **14.23** | | **15.62** | | 1.38 | | | |
| | sparse | **19.97** | | **19.48** | | 0.48 | +4.80 tok/s | +32.1 % | gain |

Tokens a second mixes two things here: the time of a step and the tokens a step yields. A step is one pass of the
main model; it yields one token of its own and the draft tokens it accepts. From 8,192 tokens on, two loads of one
arm write different greedy replies (see "The replies"), and the share of the draft a reply accepts differs with it.
At 8,192 the load `dense1` wrote a reply that accepted 0.626 of the drafted tokens while the other three loads
accepted 0.964, which is `dense`'s whole spread of 3.14 tok/s there. At 32,768 the dense loads accepted 0.508 and
0.575 and the sparse loads 0.746 and 0.732, so a part of the +68.7 % is the text and not the kernel. The next table
takes the text out.

### The time of a step

The request's generation time over its steps (generated tokens minus accepted draft tokens), per load, with the
number of steps. For a time, less is the gain; the class rule is the same.

**Pass 2, the repeated request:**

| Depth (tokens) | `dense1` | `dense2` | Spread | `sparse1` | `sparse2` | Spread | `sparse` against `dense` | | Class |
|---:|---|---|---:|---|---|---:|---:|---:|---|
| 0 | 123.21 ms (75 steps) | 123.23 (75) | 0.02 | 123.36 (75) | 123.38 (75) | 0.02 | +0.15 ms | +0.1 % | loss |
| 8,192 | 153.00 (85) | 152.97 (67) | 0.03 | 150.40 (67) | 150.34 (67) | 0.06 | −2.62 ms | −1.7 % | gain |
| 16,384 | 182.25 (72) | 182.98 (71) | 0.74 | 149.73 (72) | 150.80 (70) | 1.07 | −32.35 ms | −17.7 % | gain |
| 32,768 | 241.65 (102) | 242.49 (93) | 0.84 | 165.86 (80) | 165.74 (80) | 0.13 | −76.27 ms | −31.5 % | gain |
| 49,152 | 301.90 (79) | 301.48 (71) | 0.42 | 180.77 (76) | 180.96 (91) | 0.19 | −120.82 ms | −40.0 % | gain |

**Pass 1, the first request at a new depth:**

| Depth (tokens) | `dense1` | `dense2` | Spread | `sparse1` | `sparse2` | Spread | `sparse` against `dense` | | Class |
|---:|---|---|---:|---|---|---:|---:|---:|---|
| 0 | 124.41 ms (78 steps) | 127.38 (78) | 2.97 | 127.14 (78) | 127.32 (78) | 0.18 | +1.34 ms | +1.1 % | negligible |
| 8,192 | 193.62 (90) | 229.73 (67) | 36.10 | 221.00 (67) | 218.37 (67) | 2.63 | +8.01 ms | +3.8 % | negligible |
| 16,384 | 228.81 (73) | 206.68 (71) | 22.14 | 195.46 (72) | 194.84 (71) | 0.62 | −22.60 ms | −10.4 % | gain |
| 32,768 | 268.32 (103) | 258.10 (95) | 10.22 | 216.30 (80) | 185.10 (81) | 31.20 | −62.51 ms | −23.7 % | gain |
| 49,152 | 306.33 (80) | 323.28 (71) | 16.95 | 197.40 (76) | 202.27 (91) | 4.87 | −114.96 ms | −36.5 % | gain |

What the tables show:

- **On the repeated request the step repeats to within 1.07 ms between two loads of an arm**, whatever reply the
  load wrote (85 and 67 steps at 8,192 took 153.00 and 152.97 ms each). So the step's time is the clean measure of
  the kernel, and its classes hold at every depth.
- **The dense step grows by about 3.6 ms per 1,000 tokens of context:** `dense1` takes 153.00 ms at 8,192, 182.25 at
  16,384, 241.65 at 32,768 and 301.90 at 49,152. **The sparse step grows by less than 1 ms per 1,000 tokens:**
  `sparse1` takes 149.73 ms at 16,384, 165.86 at 32,768 and 180.77 at 49,152.
- **At 8,192 tokens the kernel saves 2.62 ms a step, and the reason is its threshold, not its speed.** In the
  release build's source, mode 1 leaves a call to the dense kernel while the cache holds fewer than 1.5 × the kept
  cells × the call's query tokens, because every query token reads its own list while a dense kernel reads the
  cache once for all of them. With 2,051 kept cells a verify pass of four tokens reaches the sparse kernel at
  12,306 cells (of five tokens at 15,383). So at 8,192 the verify passes of both arms ran the dense kernel, and at
  16,384 the sparse arm's step (149.73 ms) is the same as at 8,192 (150.40). The threshold is
  `GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO`; registry row Q1212 times it lower at 4,096 to 12,288 tokens, where an
  agent's everyday turns are.
- **At depth 0 the switch costs 0.15 ms a step on the repeated request** (123.36 and 123.38 ms against 123.21 and
  123.23), beyond the loads' spread of 0.02 ms and 0.1 % of the step. In tokens a second depth 0 is negligible
  (26.71 and 26.69 against 27.03 and 26.70).
- **What still grows with the switch on** is 27 ms between depth 0 and 8,192 (the dense kernel below the threshold)
  and 31 ms between 16,384 and 49,152 (`sparse1`). Registry row Q1213 profiles a step at depth to name its parts.
  The candidates: the draft's own passes, which read their layer's whole cache (Q1153); the indexer, which scores
  every cell; and the kernel's list of kept cells, made from a mask as long as the cache.

### The first request at a new depth

Pass 1 is slower than pass 2 in every load and both arms, for the same reply and often the same number of steps:

| Depth (tokens) | Load | Steps, pass 1 and 2 | Step on pass 1, ms | Step on pass 2, ms |
|---:|---|---|---:|---:|
| 8,192 | `dense2` | 67, 67 | 229.73 | 152.97 |
| 8,192 | `sparse1` | 67, 67 | 221.00 | 150.40 |
| 16,384 | `dense2` | 71, 71 | 206.68 | 182.98 |
| 16,384 | `sparse1` | 72, 72 | 195.46 | 149.73 |
| 32,768 | `sparse1` | 80, 80 | 216.30 | 165.86 |
| 49,152 | `dense2` | 71, 71 | 323.28 | 301.48 |
| 49,152 | `sparse1` | 76, 76 | 197.40 | 180.77 |

At 8,192 tokens `sparse1` generated the same 256 tokens in the same 67 steps in 14.81 s on pass 1 and 10.08 s on
pass 2. Pass 1 reads the prompt's last 64 tokens on a live cache and pass 2 its last 4 from a checkpoint; the
generation time is counted after the prompt in both. This is the first-request cost of registry row Q1133 on
Flash-Next. It is not the sparse kernel's: both arms pay it. Every turn of an agent is a first request at a new
depth, so pass 1 is the speed an agent sees: at 49,152 tokens 17.06 and 13.91 tok/s with the switch on against
10.45 and 11.15 with it off.

### Prompt reading

The tokens each prime request read over their time: a chunk on top of the cache of the depth before.

| Depth (tokens) | Read, on cached | `dense1` | `dense2` | `sparse1` | `sparse2` | `sparse` against `dense` | Class |
|---:|---|---:|---:|---:|---:|---:|---|
| 8,192 | 8,128 on 0 | 129.7 tok/s | 127.4 | 128.7 | 130.1 | +0.7 % | negligible |
| 16,384 | 8,708 on 7,612 | 109.8 | 110.2 | 109.6 | 109.6 | −0.4 % | negligible |
| 32,768 | 16,388 on 16,316 | 91.2 | 91.2 | 90.6 | 90.5 | −0.8 % | loss |
| 49,152 | 16,900 on 32,188 | 70.5 | 70.3 | 70.4 | 70.2 | −0.2 % | negligible |

- Depth 0 reads the 64-token tail alone, at 32.1 to 32.5 tok/s in the four loads.
- **The switch does not change prompt reading.** The kernel takes calls of at most 8 query tokens
  (`GGML_CUDA_FATTN_SPARSE_PASCAL_NQ`), and a prompt is read in micro-batches of 512. The one class that is not
  negligible, at 32,768, is 0.7 tok/s (0.8 %) beyond a spread of 0.1.
- **Prompt reading itself slows with depth in both arms:** 129.7 tok/s on an empty cache, 70.5 on 32,188 cached
  tokens (`dense1`). Reading 16,900 tokens there took 239.7 s. The dense kernel computes every query token of a
  micro-batch against every cell. Registry row Q1214 puts prompt micro-batches through the sparse kernel.

### The replies

- At depth 0 all four loads wrote the same greedy reply on both passes.
- At 8,192 the four loads wrote three different replies (`dense` two, `sparse` one); at 16,384, 32,768 and 49,152
  four different replies. Every load wrote its own reply again on pass 2.
- That is not a verdict on the kernel in either direction. On this line two loads of one arm write different
  replies from 8,192 tokens on (registry rows Q575 and Q1175), as `dense`'s own two loads do here. The kernel's
  correctness on a P106 is the replay check linked above.

### What it decides

- The rule set before the run: a gain from 8,192 tokens on makes the switch the default for this model. In the
  time of a step it is a gain from 8,192 on. In tokens a second it is negligible at 8,192 and a gain from 16,384
  on, and nothing is measurably slower at any depth. The repeated request's step at depth 0 is 0.15 ms longer. The
  preset is not changed by this run.
- For the lane's second pair, the draft on against off (Q1153), the faster setting is the switch on.
- The step still pays the dense kernel between about 3,077 and 12,306 tokens (Q1212), and prompts pay it at every
  depth (Q1214).

**Limits.**
- Two loads an arm, one day, one rig. One tail: every prompt ends in the same recorded turn, and each request
  generates 256 tokens.
- A deep prompt is recorded sessions cut at a token count with the tail's 64 tokens behind them, not a session an
  agent wrote ([bench/deep_lane](../../../bench/deep_lane/README.md), "Run against llama-server"). The step's time
  does not depend on what is written; tokens a second does, through the draft's acceptance.
- The server's log does not say which kernel took a call. That the sparse kernel ran at 16,384 and beyond is read
  from the step's time; that it did not take the verify passes at 8,192 is read from the source and fits the time.
- No quality is measured here.

## The second pair: the MTP draft on against off (Q1153)

The same line and depths with the sparse attention on in both arms: `draft` is the preset with
`GGML_CUDA_FATTN_SPARSE_PASCAL=1`, `nodraft` the same server line without the draft model and its flags. Two loads an
arm after a warm-up load. The question: is there a depth from which the draft stops paying? In other engines'
published runs an MTP draft loses at depth, and off the rig this draft's acceptance does not fall with depth
([2026-11-10_draft_acceptance_by_depth](../2026-11-10_draft_acceptance_by_depth/README.md)), so what was left to
measure is the time of a step with and without it.

**In generation the draft is not a loss at any depth up to 49,152 tokens. It is a gain at depth 0, 16,384 and 49,152
and negligible at 8,192 and 32,768, where one of its two loads wrote a reply that accepted little of the draft. On
the repeated request a draft step costs as much as 2.40 tokens of the line without the draft at depth 0 and 2.68,
2.55, 2.57 and 2.66 at 8,192, 16,384, 32,768 and 49,152, so the tokens a step must yield to pay do not rise with
depth. What depth costs the draft is prompt reading: with the draft the line reads a prompt 28.4 % slower at 8,192
tokens and 41.3 % slower at 49,152.**

### Generation: tokens a second and ms a token

Per load: pass 1, pass 2 → the load, then the load's ms a token, as in the first pair's table. Class of `draft`
against `nodraft` by the same rule ([draft_nodraft/draft_pair.md](draft_nodraft/draft_pair.md),
[draft_nodraft/draft_pair_classes.tsv](draft_nodraft/draft_pair_classes.tsv)).

| Depth (tokens) | Arm | Load 1, tok/s | ms a token | Load 2, tok/s | ms a token | Spread, tok/s | `draft` against `nodraft` | | Class |
|---:|---|---|---:|---|---:|---:|---:|---:|---|
| 0 | draft | 26.38, 27.71 → **27.03** | 37.00 | 25.79, 27.69 → **26.71** | 37.44 | 0.32 | +7.61 tok/s | +39.5 % | gain |
| | nodraft | 19.04, 19.49 → **19.26** | 51.92 | 19.04, 19.48 → **19.26** | 51.92 | 0.00 | | | |
| 8,192 | draft | 11.57, 20.16 → **14.70** | 68.02 | 17.16, 24.11 → **20.05** | 49.88 | 5.35 | +0.45 tok/s | +2.6 % | negligible |
| | nodraft | 16.03, 17.82 → **16.87** | 59.26 | 16.22, 17.82 → **16.98** | 58.89 | 0.11 | | | |
| 16,384 | draft | 18.43, 24.06 → **20.87** | 47.91 | 20.61, 23.92 → **22.14** | 45.16 | 1.27 | +4.70 tok/s | +28.0 % | gain |
| | nodraft | 16.66, 16.94 → **16.80** | 59.53 | 16.67, 16.96 → **16.81** | 59.47 | 0.02 | | | |
| 32,768 | draft | 16.80, 18.86 → **17.77** | 56.27 | 12.76, 15.45 → **13.98** | 71.55 | 3.79 | +0.55 tok/s | +3.6 % | negligible |
| | nodraft | 15.26, 15.65 → **15.45** | 64.71 | 14.96, 15.43 → **15.19** | 65.82 | 0.26 | | | |
| 49,152 | draft | 17.86, 17.98 → **17.92** | 55.80 | 16.83, 17.45 → **17.13** | 58.36 | 0.79 | +3.32 tok/s | +23.4 % | gain |
| | nodraft | 14.03, 14.53 → **14.28** | 70.04 | 13.76, 14.54 → **14.14** | 70.73 | 0.14 | | | |
| all five | draft | **18.87** | 53.00 | **19.06** | 52.48 | 0.19 | +2.63 tok/s | +16.1 % | gain |
| | nodraft | **16.37** | 61.09 | **16.30** | 61.37 | 0.07 | | | |

- **On pass 1 alone,** the speed an agent's turn sees, the classes are the same: gain at depth 0 (+37.0 %),
  negligible at 8,192 (−10.9 %: 11.57 and 17.16 against 16.03 and 16.22 tok/s), gain at 16,384 (+17.1 %), negligible
  at 32,768 (−2.2 %: 16.80 and 12.76 against 15.26 and 14.96) and gain at 49,152 (+24.8 %). On pass 2 alone 8,192
  is a gain too (+24.2 %) and 32,768 stays negligible (+10.4 %).
- **The line without the draft repeats itself:** its two loads are 0.00 to 0.26 tok/s apart at every depth, because
  it yields one token a step whatever it writes. Every spread that decides a class here is the draft arm's.
- **Two single loads of `draft` ran below both loads of `nodraft`:** `draft1` at 8,192 (14.70 against 16.87 and
  16.98) and `draft2` at 32,768 (13.98 against 15.45 and 15.19). The other load of the arm ran above them at both
  depths (20.05 and 17.77).
- **Without the draft the line slows from 19.26 tok/s at depth 0 to 14.28 at 49,152** (`nodraft1`): 51.92 ms a token
  at depth 0, 59.26 at 8,192, 59.53 at 16,384, 64.71 at 32,768 and 70.04 at 49,152.

### What a step costs and what it must yield

Tokens a second of the draft line is the tokens a step yields over the time of a step. A step of `nodraft` is one
token. So the draft pays when a step yields more tokens than its time is in tokens of `nodraft`. That ratio is the
break-even, here from the means of each arm's two loads; the share to break even is the break-even minus one, over
the draft tokens the arm offered a step (2.93 to 2.96). The step's time does not depend on the reply; the yield does
([draft_nodraft/draft_pair_breakeven.tsv](draft_nodraft/draft_pair_breakeven.tsv)).

**Pass 2, the repeated request:**

| Depth (tokens) | Step of `draft1`, ms | `draft2` | Token of `nodraft1`, ms | `nodraft2` | Tokens a step to break even | Share accepted to break even | Yielded: `draft1`, `draft2` | Accepted: `draft1`, `draft2` |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0 | 123.20 | 123.26 | 51.32 | 51.32 | 2.40 | 0.478 | 3.41, 3.41 | 0.823, 0.823 |
| 8,192 | 151.17 | 149.56 | 56.13 | 56.13 | 2.68 | 0.573 | 3.05, 3.61 | 0.696, 0.894 |
| 16,384 | 149.89 | 150.72 | 59.05 | 58.95 | 2.55 | 0.527 | 3.61, 3.61 | 0.894, 0.881 |
| 32,768 | 165.57 | 165.72 | 63.88 | 64.80 | 2.57 | 0.533 | 3.12, 2.56 | 0.719, 0.527 |
| 49,152 | 184.90 | 181.11 | 68.81 | 68.80 | 2.66 | 0.563 | 3.32, 3.16 | 0.792, 0.729 |

**Pass 1, the first request at a new depth:**

| Depth (tokens) | Step of `draft1`, ms | `draft2` | Token of `nodraft1`, ms | `nodraft2` | Tokens a step to break even | Share accepted to break even | Yielded: `draft1`, `draft2` | Accepted: `draft1`, `draft2` |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0 | 124.42 | 127.24 | 52.52 | 52.53 | 2.40 | 0.475 | 3.28, 3.28 | 0.777, 0.777 |
| 8,192 | 248.63 | 207.18 | 62.39 | 61.66 | 3.67 | 0.913 | 2.88, 3.56 | 0.637, 0.876 |
| 16,384 | 195.63 | 174.94 | 60.01 | 59.99 | 3.09 | 0.711 | 3.61, 3.61 | 0.894, 0.881 |
| 32,768 | 185.78 | 194.76 | 65.54 | 66.84 | 2.87 | 0.634 | 3.12, 2.49 | 0.719, 0.502 |
| 49,152 | 186.13 | 187.79 | 71.27 | 72.66 | 2.60 | 0.542 | 3.32, 3.16 | 0.792, 0.729 |

- **On the repeated request the break-even is flat from 8,192 on:** 2.68, 2.55, 2.57 and 2.66 tokens a step, or
  0.527 to 0.573 of the drafted tokens. The draft's step and the plain token grow with depth in the same proportion:
  the step from 151.17 to 184.90 ms between 8,192 and 49,152 (`draft1`), the token from 56.13 to 68.81
  (`nodraft1`).
- **On the first request the break-even is highest at 8,192 (3.67) and falls with depth to 2.60 at 49,152.** The
  first request's extra time falls on the draft's step far more than on a plain token; see "The first request and
  the host's memory" below.
- **The loads yielded 2.49 to 3.61 tokens a step on the first request.** The two loads that ran below `nodraft`
  are the two whose reply yielded least: `draft1` at 8,192 (2.88, with 0.637 of its drafts accepted) and `draft2` at
  32,768 (2.49, with 0.502).

### How much of the difference between two loads is the text

From 8,192 tokens on each load writes its own greedy reply (see "The replies"). The difference in tok/s between the
two loads of `draft` is the product of two factors, the tokens a step and the step's time; both passes of a load
together.

| Depth (tokens) | tok/s: `draft1`, `draft2` | `draft1` against `draft2` | Tokens a step: `draft1`, `draft2` | From the tokens a step | Step, ms: `draft1`, `draft2` | From the step's time | Replies |
|---:|---|---:|---|---:|---|---:|---|
| 0 | 27.03, 26.71 | +1.2 % | 3.35, 3.35 | 0.0 % | 123.82, 125.29 | +1.2 % | the same |
| 8,192 | 14.70, 20.05 | −26.7 % | 2.96, 3.58 | −17.3 % | 201.31, 178.57 | −11.3 % | different |
| 16,384 | 20.87, 22.14 | −5.7 % | 3.61, 3.61 | 0.0 % | 172.76, 162.83 | −5.7 % | different |
| 32,768 | 17.77, 13.98 | +27.1 % | 3.12, 2.52 | +23.8 % | 175.68, 180.45 | +2.7 % | different |
| 49,152 | 17.92, 17.13 | +4.6 % | 3.32, 3.16 | +5.2 % | 185.52, 184.45 | −0.6 % | different |

- **At 32,768 the accepted share explains the difference:** 23.8 of the 27.1 % between the two loads are the tokens
  a step (0.719 against 0.502 of the drafts accepted on the first request).
- **At 8,192 it explains the larger part:** 17.3 % from the tokens a step, 11.3 % from the step's time, which is
  the first request's (248.63 against 207.18 ms).
- **At 16,384 it explains nothing:** both loads yielded 3.61 tokens a step, and the 5.7 % are the first request's
  step (195.63 against 174.94 ms).
- So the two negligible classes come from one low-yield reply each, not from the depth: the same arm's other load
  ran 20.05 tok/s at 8,192 and 17.77 at 32,768, above both loads of `nodraft`.

### At the yields measured off the rig (est.)

A greedy reply accepts more of the draft than a sampled one, and an agent runs with the preset's sampler. The
acceptance run linked above counted tokens a step on newer cards, pooled over three prompt tails per depth: with the
preset's sampler 2.23 at depth 0 (the 64-token tail alone), 2.87 at 8,192, 3.24 at 16,384, 2.84 at 32,768 and 2.88
at 49,152; greedy 2.96, 2.95, 3.54, 3.20 and 3.41. At those yields and this run's step times the draft against no
draft would be (the yield over the break-even, minus one; estimates, not measured speeds):

| Depth (tokens) | Yield | Tokens a step | Break-even, first request | Est. | Break-even, repeated request | Est. |
|---:|---|---:|---:|---:|---:|---:|
| 0 | the sampler | 2.23 | 2.40 | est. −7.1 % | 2.40 | est. −7.3 % |
| | greedy | 2.96 | 2.40 | est. +23.8 % | 2.40 | est. +23.5 % |
| 8,192 | the sampler | 2.87 | 3.67 | est. −22.0 % | 2.68 | est. +7.0 % |
| | greedy | 2.95 | 3.67 | est. −19.6 % | 2.68 | est. +10.3 % |
| 16,384 | the sampler | 3.24 | 3.09 | est. +5.0 % | 2.55 | est. +27.2 % |
| | greedy | 3.54 | 3.09 | est. +14.6 % | 2.55 | est. +38.9 % |
| 32,768 | the sampler | 2.84 | 2.87 | est. −1.1 % | 2.57 | est. +10.5 % |
| | greedy | 3.20 | 2.87 | est. +11.3 % | 2.57 | est. +24.3 % |
| 49,152 | the sampler | 2.88 | 2.60 | est. +10.7 % | 2.66 | est. +8.1 % |
| | greedy | 3.41 | 2.60 | est. +31.4 % | 2.66 | est. +28.3 % |

At the sampler's yields the estimate is below zero at depth 0 and on a first request at 8,192 and 32,768, and it
shows no trend with depth either.

### Prompt reading

The tokens each prime request read over their time, with the ms a prompt token; the class of `draft` against
`nodraft` ([draft_nodraft/draft_pair_prompt.tsv](draft_nodraft/draft_pair_prompt.tsv)).

| Depth (tokens) | Read, on cached | `draft1` | `draft2` | `nodraft1` | `nodraft2` | `draft` against `nodraft` | Class | The draft adds, ms a prompt token |
|---:|---|---:|---:|---:|---:|---:|---|---:|
| 8,192 | 8,128 on 0 | 127.7 tok/s (7.83 ms) | 127.2 (7.86) | 179.4 (5.57) | 176.6 (5.66) | −28.4 % | loss | +2.23 |
| 16,384 | 8,708 on 7,612 | 107.6 (9.29) | 109.3 (9.15) | 164.9 (6.07) | 166.6 (6.00) | −34.6 % | loss | +3.19 |
| 32,768 | 16,388 on 16,316 | 90.3 (11.07) | 91.0 (10.99) | 148.3 (6.74) | 148.1 (6.75) | −38.8 % | loss | +4.28 |
| 49,152 | 16,900 on 32,188 | 69.9 (14.30) | 70.3 (14.22) | 119.5 (8.37) | 119.4 (8.37) | −41.3 % | loss | +5.89 |

- **The draft is a loss in prompt reading at every depth, and the loss grows with depth:** it adds 2.23 ms to every
  prompt token on an empty cache and 5.89 ms on 32,188 cached tokens. Reading 16,900 tokens there took 241.6 and
  240.4 s with the draft and 141.4 and 141.5 s without. The four prompt chunks of a load, 50,124 tokens, took 567.6
  and 563.9 s with the draft and 350.0 and 350.4 s without.
- The draft model reads every prompt token too, in micro-batches of 128 tokens on this line
  (`--spec-draft-n-ubatch 128`). Which part of that grows with depth is not measured here; registry row Q1215 times
  it. Registry row Q1105 is a cheaper catch-up, one that writes only the draft layer's cache.
- At depth 0 the 64-token tail reads at 32.9 and 32.7 tok/s with the draft against 33.4 and 33.1 without.
- Without the draft prompt reading still slows with depth: 179.4 tok/s on an empty cache, 119.5 on 32,188 cached
  tokens (`nodraft1`).

### A whole turn: the prompt it reads against the tokens it writes

A turn of an agent reads new prompt tokens (the tool's output) and then generates. The draft saves time per
generated token and costs time per prompt token, so in a turn it pays up to a certain number of new prompt tokens per
generated token: the saving over the cost. The saving here is the difference between the arms' mean ms a token on the
first request ([draft_nodraft/draft_pair_turn.tsv](draft_nodraft/draft_pair_turn.tsv)); the rows marked est. use the
yields measured off the rig instead.

| Depth (tokens) | The draft adds, ms a prompt token | Saved, ms a generated token (this run's loads) | Prompt tokens per generated token to break even | At the sampler's yield (est.) | At the greedy yield off the rig (est.) |
|---:|---:|---:|---:|---:|---:|
| 0 | +0.41 | +14.18 | 34.76 | none: est. −4.00 ms saved | est. 24.71 |
| 8,192 | +2.23 | −10.33 | none: nothing saved | none: est. −17.50 | none: est. −15.13 |
| 16,384 | +3.19 | +8.61 | 2.70 | est. 0.89 | est. 2.40 |
| 32,768 | +4.28 | −2.74 | none: nothing saved | none: est. −0.71 | est. 1.57 |
| 49,152 | +5.89 | +14.26 | 2.42 | est. 1.18 | est. 2.92 |

At 49,152 tokens of context the draft paid in this run's loads for a turn that reads up to 2.42 new prompt tokens per
token it writes; a turn that reads a 2,000-token tool output and writes 256 tokens reads 7.8. How many of an agent's
real turns are on which side is registry row Q1216.

### The first request and the host's memory

The step on the first request at a depth minus the step on its repeat (for `nodraft` the token):

| Depth (tokens) | `draft1`: pass 1, pass 2, ms | `draft2` | `nodraft1` | `nodraft2` |
|---:|---|---|---|---|
| 0 | 124.42, 123.20 (+1.22) | 127.24, 123.26 (+3.98) | 52.52, 51.32 (+1.20) | 52.53, 51.32 (+1.21) |
| 8,192 | 248.63, 151.17 (+97.46) | 207.18, 149.56 (+57.62) | 62.39, 56.13 (+6.26) | 61.66, 56.13 (+5.53) |
| 16,384 | 195.63, 149.89 (+45.74) | 174.94, 150.72 (+24.22) | 60.01, 59.05 (+0.96) | 59.99, 58.95 (+1.04) |
| 32,768 | 185.78, 165.57 (+20.21) | 194.76, 165.72 (+29.04) | 65.54, 63.88 (+1.66) | 66.84, 64.80 (+2.04) |
| 49,152 | 186.13, 184.90 (+1.23) | 187.79, 181.11 (+6.68) | 71.27, 68.81 (+2.46) | 72.66, 68.80 (+3.86) |

The host's memory during the run, from the RAM watchdog's samples, one a second
([draft_nodraft/host_memory.tsv](draft_nodraft/host_memory.tsv)):

| Load | MemAvailable after loading | Lowest MemAvailable | Most swap in use | Swap moved during the first request at 0, 8,192, 16,384, 32,768, 49,152 | Highest pressure during a request |
|---|---:|---:|---:|---|---:|
| `draft1` | 876 MiB | 76 MiB | 524 MiB | 0, +209, +128, −52, +20 MiB | 11.68 % |
| `draft2` | 907 | 96 | 1,136 | 0, +178, +238, 0, +18 | 9.94 % |
| `nodraft1` | 1,001 | 258 | 37 | 0, 0, 0, 0, 0 | 1.17 % |
| `nodraft2` | 969 | 275 | 447 | 0, 0, 0, 0, +30 | 4.00 % |

- **The first request's extra time (registry row Q1133) is the draft line's:** 20.21 to 97.46 ms a step at 8,192,
  16,384 and 32,768 against 0.96 to 6.26 ms a token without the draft.
- **Where it is largest the host was swapping.** With the draft the 4 GB host falls to 76 and 96 MiB of available
  memory, and during the first request at 8,192 and 16,384 it moved 128 to 238 MiB to swap, under a memory pressure
  of up to 11.68 % (the kernel's "some" figure, a 10-second average). Without the draft the lowest available memory
  is 258 and 275 MiB, and no swap moved during a decode request before 49,152.
- **Swap is not all of it:** at 32,768 no swap moved in either load of `draft` during the first request, and its
  step was still 20.21 and 29.04 ms longer than on the repeat. This run does not separate the parts.
- **Prompt reading does not feel the swap:** at 16,384 `draft1` read its chunk at 107.6 tok/s with no swap moving and
  `draft2` at 109.3 while 342 MiB went to swap; `nodraft2` read at 49,152 at 119.4 tok/s while 405 MiB went to swap,
  `nodraft1` at 119.5 with none.
- The first pair's four loads, all with the draft, show the same: 71 to 304 MiB of available memory at the lowest
  and 49 to 224 MiB to swap during the first request at 8,192
  ([dense_sparse/host_memory.tsv](dense_sparse/host_memory.tsv)).

### The same line twice

`draft` here and `sparse` in the first pair are the same setting, run on the same morning: four loads of one line. Per
load: tok/s on pass 1, pass 2 → the load with its ms a token; the share accepted and the tokens a step on pass 1;
the step on pass 1 and pass 2; the first six characters of the reply's hash on pass 1.

| Depth (tokens) | Load | tok/s (ms a token) | Share accepted | Tokens a step | Step on pass 1, ms | Step on pass 2, ms | Reply |
|---:|---|---|---:|---:|---:|---:|---|
| 0 | `draft1` | 26.38, 27.71 → **27.03** (37.00) | 0.777 | 3.28 | 124.42 | 123.20 | 08569d |
| | `draft2` | 25.79, 27.69 → **26.71** (37.44) | 0.777 | 3.28 | 127.24 | 123.26 | 08569d |
| | `sparse1` | 25.81, 27.67 → **26.71** (37.44) | 0.777 | 3.28 | 127.14 | 123.36 | 08569d |
| | `sparse2` | 25.78, 27.67 → **26.69** (37.47) | 0.777 | 3.28 | 127.32 | 123.38 | 08569d |
| 8,192 | `draft1` | 11.57, 20.16 → **14.70** (68.02) | 0.637 | 2.88 | 248.63 | 151.17 | 28577f |
| | `draft2` | 17.16, 24.11 → **20.05** (49.88) | 0.876 | 3.56 | 207.18 | 149.56 | 8b3879 |
| | `sparse1` | 17.29, 25.40 → **20.58** (48.60) | 0.964 | 3.82 | 221.00 | 150.40 | a84769 |
| | `sparse2` | 17.50, 25.42 → **20.73** (48.25) | 0.964 | 3.82 | 218.37 | 150.34 | a84769 |
| 16,384 | `draft1` | 18.43, 24.06 → **20.87** (47.91) | 0.894 | 3.61 | 195.63 | 149.89 | e766b3 |
| | `draft2` | 20.61, 23.92 → **22.14** (45.16) | 0.881 | 3.61 | 174.94 | 150.72 | 41dce8 |
| | `sparse1` | 18.19, 23.75 → **20.60** (48.54) | 0.876 | 3.56 | 195.46 | 149.73 | 625c60 |
| | `sparse2` | 18.51, 24.25 → **20.99** (47.64) | 0.881 | 3.61 | 194.84 | 150.80 | 58515e |
| 32,768 | `draft1` | 16.80, 18.86 → **17.77** (56.27) | 0.719 | 3.12 | 185.78 | 165.57 | 71c146 |
| | `draft2` | 12.76, 15.45 → **13.98** (71.55) | 0.502 | 2.49 | 194.76 | 165.72 | a940e1 |
| | `sparse1` | 14.79, 19.29 → **16.75** (59.71) | 0.746 | 3.20 | 216.30 | 165.86 | 76507e |
| | `sparse2` | 17.07, 19.31 → **18.12** (55.18) | 0.732 | 3.16 | 185.10 | 165.74 | f22b9e |
| 49,152 | `draft1` | 17.86, 17.98 → **17.92** (55.80) | 0.792 | 3.32 | 186.13 | 184.90 | 5dfb04 |
| | `draft2` | 16.83, 17.45 → **17.13** (58.36) | 0.729 | 3.16 | 187.79 | 181.11 | 22ebf9 |
| | `sparse1` | 17.06, 18.63 → **17.81** (56.13) | 0.807 | 3.37 | 197.40 | 180.77 | 5fe0ff |
| | `sparse2` | 13.91, 15.55 → **14.68** (68.11) | 0.616 | 2.81 | 202.27 | 180.96 | 30320e |

- **The two runs agree on the step.** On the repeated request the four loads are within 0.18 ms at depth 0, 1.60 at
  8,192, 1.07 at 16,384, 0.29 at 32,768 and 4.14 at 49,152, where `draft1` took 184.90 ms and the other three
  181.11, 180.77 and 180.96.
- **Tokens a second differs with the reply, not with the run.** At depth 0 the four loads wrote one reply and ran
  27.03, 26.71, 26.71 and 26.69 tok/s. From 8,192 on they wrote three or four replies a depth and accepted 0.502 to
  0.964 of their drafts, and their tok/s is 6.02 apart at 8,192, 1.54 at 16,384, 4.15 at 32,768 and 3.24 at 49,152.
- Of the four loads of this line, one a depth ran below the line without the draft at 8,192 (`draft1`) and at
  32,768 (`draft2`), and none at 0, 16,384 or 49,152.
- The first request's step is the part that does not repeat: 174.94 to 195.63 ms at 16,384 and 185.10 to 216.30 at
  32,768 over the four loads.

### The replies

- At depth 0 both loads of `draft` wrote the reply all four loads of the first pair wrote, and both loads of
  `nodraft` wrote one other reply. A line with the draft and a line without it write different greedy text by
  construction: a pass over several tokens is another arithmetic than a pass over one
  ([2026-11-10_draft_line_exactness](../2026-11-10_draft_line_exactness/README.md)).
- From 8,192 on each of the four loads wrote its own reply, in both arms.
- Every load wrote its own reply again on pass 2 except `nodraft2` at 32,768, which wrote another one. It is the
  only load of the two pairs that did so.

### What it decides

- The rule set before the run: a depth from which the draft is a loss gets a gate on depth before any other work on
  the draft. No depth shows a loss in generation, and the tokens a step the draft needs do not rise with depth, so
  this run gives no gate on depth.
- The draft's margin in generation is the reply's yield against a break-even of 2.40 to 2.68 tokens a step on a
  repeated request and up to 3.67 on a first one. Work that raises the yield or shortens the step is measured
  against those numbers.
- Depth costs the draft in the prompt, 28.4 to 41.3 % of prompt speed, and on the first request of a turn, where the
  4 GB host swaps. Registry rows Q1215 (the draft's prompt cost by part), Q1216 (the draft's worth per recorded turn),
  Q1105 (a cheaper catch-up) and Q1133 (the first request).
- The preset is not changed by this run.

**Limits.**
- Two loads an arm, one day, one rig, one tail, greedy replies of 256 tokens. The two negligible classes rest on one
  low-yield reply each; more loads, or prompts that end in whole recorded turns (registry row Q1176), would narrow
  them.
- The rows marked est. combine this run's step times with yields counted on other cards and other tails.
- A deep prompt is recorded sessions cut at a token count with the tail behind them, and the model's reply to such a
  prompt accepts its own share of the draft.
- The host's memory is read from one sample a second; the run does not show what was swapped.
- No quality is measured here.

## Method

- **The line:** the preset as it is on nine P106-090 cards (`-ts 6,6,6,6,5,6,5,6,3`, `-c 65536`, `-ub 512`, the MTP
  draft with `--spec-draft-n-max 3`), with one variable added per arm: `GGML_CUDA_FATTN_SPARSE_PASCAL=0` or `=1`.
  [dense_sparse/loads.tsv](dense_sparse/loads.tsv) has each load's line. After loading the cards held 5,059, 5,095,
  4,963, 5,111, 4,875, 4,979, 4,381, 4,991 and 4,565 MiB in every load of both arms: the switch takes no memory at
  load.
- **The loads:** one untimed warm-up load of `dense`, then `dense1`, `sparse1`, `dense2`, `sparse2`, each a fresh
  server start (224 to 227 s to load). Before each, the quiet check: no llama process, 0 MiB on every card. The
  host has 4 GB of RAM; the server ran under [scripts/ram_watchdog.sh](../../../scripts/ram_watchdog.sh), which
  did not act. The run took 70 minutes.
- **The prompts:** [deep_client.py](../../../bench/deep_lane/deep_client.py) on the release check's recorded
  requests of the arm `t1swift` (12 sessions), rendered by the server's chat template with their tools and
  tokenized into one stream. A prompt of depth D is the stream's first tokens and a tail of 64 tokens, D tokens
  together; depth 0 is the tail alone. The tail is the end of the recorded request with the longest reply. Every
  load sent the same token ids.
- **The requests** per depth: a prime (the prompt without its tail, read on top of the cache), then the whole
  prompt twice at temperature 0 with 256 tokens out and end of text ignored. The primes at 16,384, 32,768 and
  49,152 continued from a context checkpoint.
- **The tables:** [deep_table.py](../../../bench/deep_lane/deep_table.py) on the run's `requests.csv`. The step
  tables and the ms a token were added to it after the run, and the tables here were rebuilt with it on another
  machine from the same file; the tok/s and the classes are the ones the rig printed at the end of the run.
- **The second pair's line:** the preset with `GGML_CUDA_FATTN_SPARSE_PASCAL=1` added for both arms. `nodraft`
  drops the draft model and its flags (`-md`, `--spec-type draft-mtp`, `--spec-draft-n-max 3`,
  `--spec-draft-n-ubatch 128`, `-otd token_embd=CUDA4`) and changes nothing else;
  [draft_nodraft/loads.tsv](draft_nodraft/loads.tsv) has each load's line. Both pairs sent the same prompts, token
  for token.
- **The second pair's loads:** one untimed warm-up load of `draft`, then `draft1`, `nodraft1`, `draft2`, `nodraft2`
  (224 and 225 s to load with the draft, 213 and 211 s without), each after the quiet check. A load with the draft
  took 16.1 minutes and a load without it 12.5; the run took 62 minutes. The watchdog did not act.
- **The draft's memory:** after loading the cards held the first pair's memory with the draft and 5,013, 5,061,
  4,917, 5,077, 4,191, 4,933, 4,355, 4,945 and 2,411 MiB without it. The draft takes 3,116 MiB over the nine cards,
  2,154 of them on the last card and 684 on the fifth, which holds its token embeddings.
- **The second pair's tables:** the driver's own, written on the rig ([draft_nodraft/table.md](draft_nodraft/table.md);
  its control is `draft`, so its classes read `nodraft` against `draft`). [run/draft_pair.py](run/draft_pair.py)
  turns the same `requests.csv` into the tables of this page: the class of `draft` against `nodraft` per pass, the
  break-even, the two factors between loads, prompt reading, the turn, and the first pair's `sparse` loads beside
  `draft`. [run/host_memory.py](run/host_memory.py) reads the watchdog's samples. Both ran on another machine.

## Files

| File | What |
|---|---|
| [dense_sparse/table.md](dense_sparse/table.md) | the driver's tables: generation per depth, arm, load and pass with ms a token, the step on pass 1 and pass 2, prompt reading, the draft's counters per load, the checks |
| [dense_sparse/classes.tsv](dense_sparse/classes.tsv) | every class as a table: metric (`generation`, `generation_all`, `step_pass1`, `step_pass2`, `prompt`), depth, each load's value, spread, difference, percent, class |
| [dense_sparse/requests.csv](dense_sparse/requests.csv) | one line per request of the five loads: prompt tokens sent, reused and read with their time, tokens generated with their time, the draft's counters, hashes of the prompt and the reply |
| [dense_sparse/loads.tsv](dense_sparse/loads.tsv) | one line per load: arm, start, seconds to load, the server line as it ran |
| [dense_sparse/host_memory.tsv](dense_sparse/host_memory.tsv) | the host's memory per load and per request, from the RAM watchdog's samples: lowest MemAvailable, most swap in use and its change, highest pressure |
| [draft_nodraft/table.md](draft_nodraft/table.md), [classes.tsv](draft_nodraft/classes.tsv) | the second pair's driver tables and classes, `nodraft` against `draft` |
| [draft_nodraft/draft_pair.md](draft_nodraft/draft_pair.md) | the second pair read part by part: `draft` against `nodraft` per pass, the break-even, the yields measured off the rig (est.), the two loads of `draft` against each other, prompt reading, the turn, and the four loads of the line |
| [draft_nodraft/draft_pair_classes.tsv](draft_nodraft/draft_pair_classes.tsv), [_breakeven.tsv](draft_nodraft/draft_pair_breakeven.tsv), [_prompt.tsv](draft_nodraft/draft_pair_prompt.tsv), [_turn.tsv](draft_nodraft/draft_pair_turn.tsv), [_loads.tsv](draft_nodraft/draft_pair_loads.tsv) | the same as tables: the classes (generation on both passes, pass 1, pass 2, all depths, and the prompt), the break-even per depth and pass, the prompt's ms a token per arm, the turn's break-even, and one line per load, depth and pass of both runs |
| [draft_nodraft/requests.csv](draft_nodraft/requests.csv), [loads.tsv](draft_nodraft/loads.tsv), [host_memory.tsv](draft_nodraft/host_memory.tsv) | the second pair's requests, loads and host memory, as for the first pair |
| [run/draft_pair.py](run/draft_pair.py), [run/host_memory.py](run/host_memory.py) | the two table scripts of the second pair (Python 3.9 or newer, standard library only) |

Not published: the prompts' token ids, the replies' first characters and the server logs (recorded text and paths),
and the raw telemetry (the cards every 10 s, the watchdog's samples, the quiet and fit checks, which the text above
summarizes).

## Commands

From a machine that only runs ssh, in a checkout of this repository. `RIG` is the rig's ssh alias; `BIN`, `MODEL`
and `MODEL_DRAFT` are paths on the rig; `TEXT_LOCAL` is the release check's recorded requests (`proxy.jsonl` beside
`proxy.jsonl.bodies.gz`), copied over by `start`:

```bash
RIG=rig BIN=<build>/bin/llama-server \
MODEL=<models>/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf \
MODEL_DRAFT=<models>/mtp-Qwen3.8-Flash-Next-draft.gguf \
TEXT_LOCAL=<recorded>/proxy.jsonl.bodies.gz TAG='^t1swift/' WD_ENV=MIN_AVAIL_MB=60 \
ARMS="dense sparse" ARM_ENV_dense=GGML_CUDA_FATTN_SPARSE_PASCAL=0 ARM_ENV_sparse=GGML_CUDA_FATTN_SPARSE_PASCAL=1 \
bash bench/deep_lane/remote.sh start
```

```bash
RIG=rig bash bench/deep_lane/remote.sh wait
```

```bash
RIG=rig OUT=<results folder> bash bench/deep_lane/remote.sh collect
```

`DRY=1` in front of the first command prints the two server lines and starts nothing. Nothing else ran on the rig
between `start` and the end of `wait`. The tables again from a results folder:

```bash
python3 -I bench/deep_lane/deep_table.py <results folder> --out table.md --tsv classes.tsv
```

The second pair is the same three commands with other arms in the first:

```bash
RIG=rig BIN=<build>/bin/llama-server \
MODEL=<models>/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf \
MODEL_DRAFT=<models>/mtp-Qwen3.8-Flash-Next-draft.gguf \
TEXT_LOCAL=<recorded>/proxy.jsonl.bodies.gz TAG='^t1swift/' WD_ENV=MIN_AVAIL_MB=60 \
ARMS="draft nodraft" ARM_DRAFT_nodraft=off ENV_ADD=GGML_CUDA_FATTN_SPARSE_PASCAL=1 \
bash bench/deep_lane/remote.sh start
```

Its tables, from the two results folders (the yields are the pooled tokens a step of
[2026-11-10_draft_acceptance_by_depth/pooled.tsv](../2026-11-10_draft_acceptance_by_depth/pooled.tsv), the sampler
pass and the first greedy pass):

```bash
python3 -I docs/data/2026-11-10_deep_lane/run/draft_pair.py <results folder of the second pair> \
  --second <results folder of the first pair>:sparse --second-name "the first pair" \
  --yield "the sampler off the rig:0=2.226,8192=2.866,16384=3.241,32768=2.844,49152=2.876" \
  --yield "greedy off the rig:0=2.965,8192=2.954,16384=3.539,32768=3.200,49152=3.413" \
  --out draft_pair.md --tsv draft_pair
python3 -I docs/data/2026-11-10_deep_lane/run/host_memory.py <results folder> > host_memory.tsv
```

## Provenance and licence

The recorded requests are this project's release check of 2026-11-05
([docs/data/2026-11-05_release_flashnext](../2026-11-05_release_flashnext/README.md)): the agent's sessions on 12
bug-fix tasks (Exercism exercises, MIT). The files here hold token counts, times, the server's counters and hashes;
no prompt or reply text. The scripts are under this repository's licence.
