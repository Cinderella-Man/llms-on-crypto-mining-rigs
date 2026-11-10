# The deep lane on the rig: the sparse Pascal attention off and on, and the MTP draft on and off, at 0 to 49,152 tokens of context, a 131,072-token line to 120,000 tokens, the draft's micro-batch in prompt reading, and prompt micro-batches through the sparse kernel (Q1143, Q230, Q1153, Q1154, Q1215, Q1214; 2026-11-10)

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

A third run put the same line, with and without the draft, at a 131,072-token context (`-c 131072 -ub 256`) and
held it at 49,152, 65,536 and 120,000 tokens, after a projection and a real load of each line's memory (registry row
Q1154). Its result is [the 131,072-token line](#the-third-run-a-131072-token-line-q1154) below.

A fourth run timed the draft model's micro-batch in prompt reading: `--spec-draft-n-ubatch` 128 (the preset), 256
and 512 against the line without the draft, on the second pair's line and prompts (registry row Q1215). Its result
is [the draft's micro-batch](#the-fourth-run-the-drafts-micro-batch-in-prompt-reading-q1215) below.

A sixth run read the prompts in micro-batches of 128 tokens through the sparse kernel: the preset's `-ub 512`
(control), `-ub 128`, and `-ub 128` with the switch at 2 and `GGML_CUDA_FATTN_SPARSE_PASCAL_NQ=128`, at 0, 16,384,
32,768 and 49,152 tokens (registry row Q1214). Its result is
[prompt micro-batches through the sparse kernel](#the-sixth-run-prompt-micro-batches-through-the-sparse-kernel-q1214)
below. The fifth deep-lane run on the rig profiled a step at depth and has its own page
([2026-11-10_step_parts_depth](../2026-11-10_step_parts_depth/README.md)).

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

## The third run: a 131,072-token line (Q1154)

Real coding attempts write 40,000 tokens and more, and a benchmark that keeps an attempt's whole history cannot fit
a typical attempt into the preset's 65,536 tokens. The question: does the same line fit at `-c 131072` on the nine
6 GB cards and the 4 GB host, and how fast is it at 65,536 and 120,000 tokens? The line is the preset with
`GGML_CUDA_FATTN_SPARSE_PASCAL=1`, `-c 131072` and `-ub 256` in place of `-ub 512` (the micro-batch the projection
of registry row Q213 allows with the draft at this context), with the draft (`draft`) and without it (`nodraft`).
First a projection per card of each line, then one real load of each at depth 0 (`load_check`), then the deep lane
at 49,152, 65,536 and 120,000 tokens, one load an arm after a warm-up load.

**The line fits and holds to 120,000 tokens with the draft and without it. With the draft the fullest card held
5,419 of its 6,068 usable MiB during the run, the host's available memory fell to 175 MiB at the lowest while
1,118 MiB went to swap during the first long prompt, and the RAM watchdog did not act. At 120,000 tokens the line
generates 15.24 tok/s with the draft and 10.96 without (one load each). At 49,152 tokens it is as fast as the
65,536-token preset: a draft step takes 181.81 ms on the repeated request, against 184.90, 181.11, 180.77 and
180.96 ms in R39b's and R39a's four loads. With the draft it reads prompts a third slower: 49.0 against 73.6 tok/s
on 65,212 cached tokens.**

### Memory on the cards

MiB in use per card: the projection of [scripts/fit_probe.sh](../../../scripts/fit_probe.sh) (llama.cpp's `--fit`
step with no weights loaded, seconds a line), the real use right after loading, and the most each card held during
the load (the cards every 10 s). The last column is what is left on the fullest card, card 3, of the 6,068 MiB a
P106-090 offers a process (6,144 MiB minus the driver's reserve of 76;
[ctx131k/card_fit.tsv](ctx131k/card_fit.tsv), made by [run/card_fit.py](run/card_fit.py)).

| Line | | Card 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | Left on card 3 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 131,072, `-ub 256`, draft | projected | 5,088 | 5,386 | 5,092 | 5,402 | 5,004 | 5,109 | 4,672 | 5,120 | 4,892 | 666 |
| | after loading (three loads) | 5,145 | 5,373 | 5,081 | 5,389 | 4,993 | 5,097 | 4,659 | 5,109 | 4,887 | 679 |
| | highest, `draft1` | 5,187 | 5,403 | 5,111 | 5,419 | 5,019 | 5,127 | 4,689 | 5,139 | 4,935 | 649 |
| 131,072, `-ub 256`, no draft | projected | 5,040 | 5,348 | 5,046 | 5,365 | 4,321 | 5,062 | 4,644 | 5,073 | 2,541 | 703 |
| | after loading (two loads) | 5,099 | 5,339 | 5,035 | 5,355 | 4,309 | 5,051 | 4,633 | 5,063 | 2,529 | 713 |
| | highest, `nodraft1` | 5,141 | 5,369 | 5,065 | 5,385 | 4,339 | 5,081 | 4,663 | 5,093 | 2,559 | 683 |
| 65,536, `-ub 512`, draft (R39b) | after loading | 5,059 | 5,095 | 4,963 | 5,111 | 4,875 | 4,979 | 4,381 | 4,991 | 4,565 | 957 |
| | highest, `draft1` | 5,101 | 5,137 | 5,005 | 5,153 | 4,913 | 5,021 | 4,423 | 5,033 | 4,625 | 915 |
| 65,536, `-ub 512`, no draft (R39b) | after loading | 5,013 | 5,061 | 4,917 | 5,077 | 4,191 | 4,933 | 4,355 | 4,945 | 2,411 | 991 |

- **The projection matches the real load** within 13 MiB on cards 1 to 8, and on card 0 it is 57 MiB under it with
  the draft and 59 MiB without. llama.cpp
  counts the probe's own 45 MiB context as taken and prints 621 MiB free on card 3 with the draft and 658 without.
  Its `--fit` target of 1,024 MiB free per card is met by none of the four lines probed, the 65,536-token preset's
  included; the preset runs with `-fit off`.
- **What the doubled context costs** after loading, against the preset with its `-ub 512`: 86 MiB on card 0, 278 on
  cards 1, 3 and 6 (the three cards with two attention layers), 118 on cards 2, 4, 5 and 7, and on card 8 118 without
  the draft and 322 with it. That is 1,510 MiB over the nine cards without the draft and 1,714 with it.
- During the requests the fullest card grew by 30 MiB in both lines (5,389 to 5,419 MiB with the draft, 5,355 to
  5,385 without).
- Registry row Q213 had projected 309 MiB free on the fullest card for a 128k line with the draft, on v2.0's build
  and another split; on this build and the preset's split the projection is 621 MiB (llama.cpp's figure).

### The host's memory

From the RAM watchdog's samples, one a second ([ctx131k/host_memory.tsv](ctx131k/host_memory.tsv),
[ctx131k/load_check/host_memory.tsv](ctx131k/load_check/host_memory.tsv)):

| Load | MemAvailable after loading | Lowest MemAvailable | During | Most swap in use | Moved to swap | Highest pressure |
|---|---:|---:|---|---:|---|---:|
| `load_check`: `draft1` (depth 0) | 908 MiB | 437 MiB | the decode at depth 0 | 20 MiB | none | 0.00 % |
| `load_check`: `nodraft1` (depth 0) | 970 | 610 | the decode at depth 0 | 20 | none | 0.00 % |
| `warm` (draft, depth 0) | 887 | 441 | the decode at depth 0 | 20 | none | 0.00 % |
| `nodraft1` | 947 | 190 | the decode at 120,000 | 20 | none | 0.18 % |
| `draft1` | 878 | 175 | the prime to 49,152 | 1,223 | +1,118 MiB in the prime to 49,152, +134 in the first request at 49,152 | 9.70 % |

- **With the draft the host gave 1,118 MiB to swap while the line read its first 49,028 prompt tokens**, under a
  memory pressure of up to 9.70 % (the kernel's "some" figure, a 10-second average). The most written to swap in any
  10 seconds was 162 MiB, against the watchdog's limit of 600. From then on at least 725 MiB stayed available to the
  end of the load.
- **Without the draft no swap moved.** The lowest available memory fell with depth: 253 MiB during the decode at
  49,152, 223 at 65,536 and 190 at 120,000.
- On the 65,536-token line R39b's loads fell to 76 and 96 MiB with the draft and 258 and 275 without, so this line
  is not tighter on the host than the preset. The watchdog did not act in any load.

### Generation: tokens a second and ms a token

Per load: pass 1, pass 2 → the load, with the load's ms a token ([ctx131k/table.md](ctx131k/table.md),
[ctx131k/draft_pair.md](ctx131k/draft_pair.md)).

| Depth (tokens) | `draft1`, tok/s (ms a token) | Accepted on pass 1 (tokens a step) | `nodraft1`, tok/s (ms a token) | `draft` against `nodraft` |
|---:|---|---:|---|---:|
| 49,152 | 15.01, 18.05 → **16.39** (61.01) | 0.771 (3.28) | 14.12, 14.55 → **14.33** (69.78) | +14.4 % |
| 65,536 | 17.09, 18.82 → **17.92** (55.82) | 0.930 (3.71) | 13.39, 13.55 → **13.47** (74.25) | +33.0 % |
| 120,000 | 15.20, 15.29 → **15.24** (65.60) | 0.964 (3.82) | 10.85, 11.07 → **10.96** (91.25) | +39.1 % |
| all three | **16.44** (60.81) | | **12.75** (78.42) | +29.0 % |

- **One load an arm, so no class.** Two loads an arm were est. 2 h 18 min from R39b's prompt speeds; the run took 68
  minutes. Registry row Q1218 is the second load of each arm.
- Every pass 2 repeated its pass 1's reply, and the draft line's replies here accepted much of the draft: 0.930
  and 0.964 of the drafted tokens at 65,536 and 120,000, more than any of R39b's loads at any depth.
- Without the draft the line slows from 14.33 tok/s at 49,152 to 10.96 at 120,000: 69.78, 74.25 and 91.25 ms a
  token.

### The step and what it must yield

The draft's step against a token of the line without it, and their ratio, the tokens a step the draft must yield to
pay ([ctx131k/draft_pair_breakeven.tsv](ctx131k/draft_pair_breakeven.tsv)). The last two columns are an estimate:
the tokens a step counted off the rig with the preset's sampler at 49,152 tokens, 2.88
([2026-11-10_draft_acceptance_by_depth](../2026-11-10_draft_acceptance_by_depth/README.md); nothing deeper was
counted), at this run's step times.

| Depth (tokens) | Step of `draft1`, ms: pass 1, pass 2 | Token of `nodraft1`, ms: pass 1, pass 2 | Break-even, tokens a step: pass 1, pass 2 | At 2.88 tokens a step (est.): pass 1, pass 2 |
|---:|---|---|---|---|
| 49,152 | 218.69, 181.81 | 70.84, 68.72 | 3.09, 2.65 | −6.8 %, +8.7 % |
| 65,536 | 217.07, 197.12 | 74.68, 73.81 | 2.91, 2.67 | −1.1 %, +7.7 % |
| 120,000 | 251.36, 249.94 | 92.15, 90.34 | 2.73, 2.77 | +5.4 %, +4.0 % |

- **The break-even barely rises to 120,000 tokens:** 2.65, 2.67 and 2.77 tokens a step on the repeated request,
  against R39b's 2.40, 2.68, 2.55, 2.57 and 2.66 at 0 to 49,152.
- **With the sparse attention the step grows by under 1 ms per 1,000 tokens of context:** from 65,536 to 120,000
  the draft's step on the repeat grows by 52.82 ms (0.97 ms per 1,000 tokens) and a plain token by 16.53 ms (0.30).
  R39a's dense step grew by 3.68 ms per 1,000 tokens from 32,768 to 49,152; continued to 120,000 that would be
  est. 562 ms a step.
- **The first request:** at 49,152 the draft's step is 36.88 ms longer on pass 1 than on the repeat (a plain token
  2.12 ms), right after the prime that moved 1,118 MiB to swap; at 65,536 19.95 ms (0.87) and at 120,000 1.42 ms
  (1.81). Registry row Q1133.

### Prompt reading

The tokens each prime read over their time, with the ms a prompt token
([ctx131k/draft_pair_prompt.tsv](ctx131k/draft_pair_prompt.tsv)):

| Depth (tokens) | Read, on cached | `draft1` | `nodraft1` | `draft` against `nodraft` | The draft adds, ms a prompt token |
|---:|---|---:|---:|---:|---:|
| 49,152 | 49,028 on 60 | 100.5 tok/s (9.95 ms) | 146.2 (6.84) | −31.3 % | +3.11 |
| 65,536 | 16,644 on 48,828 | 66.2 (15.10) | 93.7 (10.68) | −29.3 % | +4.42 |
| 120,000 | 54,724 on 65,212 | 49.0 (20.41) | 73.6 (13.58) | −33.5 % | +6.83 |

- The three primes, 120,396 tokens, took 1,856.3 s with the draft and 1,256.3 s without.
- On 65,212 cached tokens the draft adds 6.83 ms to a prompt token; in R39b it added 5.89 on 32,188 (another
  micro-batch and other chunks, so not a pair).
- At these speeds a 2,000-token tool output at 120,000 tokens of context reads in est. 40.8 s with the draft and
  27.2 s without.
- In a whole turn the draft paid in this run's loads up to 1.35, 3.66 and 3.86 new prompt tokens per generated token
  at 49,152, 65,536 and 120,000; at the sampler's 2.88 tokens a step est. nowhere at 49,152 and 65,536 and up to
  0.70 at 120,000 ([ctx131k/draft_pair_turn.tsv](ctx131k/draft_pair_turn.tsv)). Registry row Q1216 counts which side
  real turns fall on.

### At 49,152 tokens beside the 65,536-token line, and at depth 0

All loads with the sparse attention on. Tok/s is both passes of a load; the share accepted and the tokens a step are
pass 1's.

| Line | Load | tok/s (ms a token) | Share accepted | Tokens a step | Step on pass 1, ms | Step on pass 2, ms |
|---|---|---|---:|---:|---:|---:|
| 131,072, `-ub 256` | `draft1` | **16.39** (61.01) | 0.771 | 3.28 | 218.69 | 181.81 |
| 65,536, `-ub 512` (R39b) | `draft1` | **17.92** (55.80) | 0.792 | 3.32 | 186.13 | 184.90 |
| | `draft2` | **17.13** (58.36) | 0.729 | 3.16 | 187.79 | 181.11 |
| 65,536, `-ub 512` (R39a) | `sparse1` | **17.81** (56.13) | 0.807 | 3.37 | 197.40 | 180.77 |
| | `sparse2` | **14.68** (68.11) | 0.616 | 2.81 | 202.27 | 180.96 |
| 131,072, no draft | `nodraft1` | **14.33** (69.78) | | 1 | 70.84 | 68.72 |
| 65,536, no draft (R39b) | `nodraft1` | **14.28** (70.04) | | 1 | 71.27 | 68.81 |
| | `nodraft2` | **14.14** (70.73) | | 1 | 72.66 | 68.80 |

- **At the same depth the larger context costs nothing in generation:** on the repeated request the draft's step is
  181.81 ms against 184.90, 181.11, 180.77 and 180.96, and a plain token 68.72 ms against 68.81 and 68.80.
- The 49,152-token prompt here is other text than R39's: the stream is cut from both arms' recorded sessions
  (`t1swift` and `t1base`, 24 sessions) and the tail is another session's turn with an equally long recorded reply.
  So the replies and their acceptance are not paired with R39's; the step times are.
- Pass 1's 218.69 ms is the first request after a prime of 49,028 tokens on an empty cache; R39b's first requests at
  49,152 followed a prime of 16,900 tokens.
- **At depth 0** (`load_check`, [ctx131k/load_check/table.md](ctx131k/load_check/table.md)) the line with the draft
  ran 25.75, 27.65 → **26.67** tok/s (37.50 ms a token), its step on the repeat 123.46 ms; without the draft 18.96,
  19.48 → **19.22** (52.03), a token on the repeat 51.33 ms. The 65,536-token line in R39b: 27.03 (37.00) and 26.71
  (37.44) tok/s, steps of 123.20 and 123.26 ms; 19.26 (51.92) twice, tokens of 51.32 and 51.32 ms. The load check's
  draft load was the first load of the weights on the rig in 25 minutes, with no warm-up load before it (registry row
  Q1055).

### What it decides

- **A 131,072-token line is usable on the rig as sold, with the draft.** It fits the cards with 649 MiB left on the
  fullest one at its highest, it holds the 4 GB host to 120,000 tokens, it is as fast as the 65,536-token preset at
  the same depth, and at 120,000 tokens it still generates 15.24 tok/s with the draft (10.96 without). The draft
  line was the faster one in generation at all three depths in this run's loads.
- **What the draft costs at this depth is prompt reading:** 31.3 %, 29.3 % and 33.5 % of prompt speed at 49,152,
  65,536 and 120,000 tokens, 6.83 ms a prompt token on 65,212 cached tokens. At the yield counted with the preset's sampler its gain in generation at 120,000 is
  est. +4.0 % on the repeat and +5.4 % on a first request. So over a whole long session the line without the draft
  may be the faster one; registry row Q1216 decides that from recorded turns.
- **`-ub 256` costs nothing in generation** at depth 0 or 49,152. Its cost in prompt reading against `-ub 512` is not
  measured here; on stock llama.cpp's agent check it read prompts at 50.3 against 53.7 tok/s (registry row Q1128).
  Whether one 131,072-token preset can replace the 65,536-token one is the agent check of both lines (registry row
  Q1220).
- Next: the second load of each arm for classes (Q1218), the same probe and lane at 196,608 and 262,144 tokens
  (Q1219), and a second preset file for this line (Q1221). The preset is not changed by this run.

**Limits.**
- One load an arm, one day, one rig: no class.
- Greedy replies of 256 tokens at the end of recorded sessions cut at a token count; at depth the model often says
  the context looks cut. The draft's yield here, 3.28, 3.71 and 3.82 tokens a step, is that of three such replies.
- The estimates use tokens a step counted on other cards at 49,152 tokens for all three depths.
- The host's memory is one sample a second; the run does not show what was swapped.
- No quality is measured here.

## The fourth run: the draft's micro-batch in prompt reading (Q1215)

The second pair found that with the draft the line reads prompts 28.4 % slower on an empty cache and 41.3 % slower
on 32,188 cached tokens. The draft model reads every prompt token too, in micro-batches of 128 tokens
(`--spec-draft-n-ubatch 128`), a quarter of the main model's 512. That value was set so that the draft would fit the
4 GB host: by llama-server's projection the draft's compute buffer in host memory is 25 MiB at 128 against 101 MiB at
512 (registry row Q558). This run timed the draft's micro-batch at 128 (the preset, `draft`), 256 (`dub256`) and 512
(`dub512`, which is llama.cpp's default: without the flag the draft takes the main model's `-ub`) against the line
without the draft (`nodraft`). It used the second pair's line, depths and prompts, two loads an arm, and only 16
tokens a decode request, since the question is prompt reading.

**A draft micro-batch of 512 reads prompts faster than the preset's 128: a gain at 16,384, 32,768 and 49,152 tokens
(+4.2 %, +3.6 % and +3.2 %) and over a load's four prompt chunks (91.8 and 91.5 against 88.0 and 88.7 tok/s, +3.7 %;
546.2 and 548.0 s against 569.6 and 565.2 s), negligible at 8,192. The 4 GB host held: its lowest available memory
was 263 and 238 MiB against 71 and 81 MiB with the preset, and the RAM watchdog never acted. But 512 takes away only
9.3 % of what the draft costs in prompt reading. 90.7 % is left, and the part of it that grows with depth does not
change with the micro-batch.**

### Prompt reading: tok/s (ms a prompt token)

Per load, the tokens its prime read over their time. Each arm is classed against `draft` by the rule of the first
pair. The draft adds: the arm's mean ms a prompt token minus `nodraft`'s. Taken away: the share of what the draft adds
in `draft` that the arm no longer adds ([draft_prompt_cost/draft_ubatch.md](draft_prompt_cost/draft_ubatch.md),
[draft_prompt_cost/draft_ubatch_prompt.tsv](draft_prompt_cost/draft_ubatch_prompt.tsv)).

| Depth (tokens) | Read, on cached | Arm | Load 1 | Load 2 | Spread, tok/s | Against `draft` | Class | The draft adds, ms a prompt token | Taken away |
|---:|---|---|---|---|---:|---:|---|---:|---:|
| 8,192 | 8,128 on 0 | draft (128) | 122.5 (8.17) | 129.2 (7.74) | 6.8 | | | +2.31 | |
| | | dub256 | 130.2 (7.68) | 129.0 (7.75) | 1.2 | +3.0 % | negligible | +2.08 | 10.2 % |
| | | dub512 | 132.8 (7.53) | 131.9 (7.58) | 0.9 | +5.2 % | negligible | +1.92 | 17.1 % |
| | | nodraft | 177.1 (5.65) | 177.5 (5.63) | 0.4 | +40.9 % | gain | | |
| 16,384 | 8,708 on 7,612 | draft (128) | 107.8 (9.28) | 108.9 (9.18) | 1.2 | | | +3.23 | |
| | | dub256 | 110.7 (9.03) | 110.8 (9.02) | 0.1 | +2.2 % | gain | +3.03 | 6.2 % |
| | | dub512 | 113.2 (8.84) | 112.7 (8.87) | 0.5 | +4.2 % | gain | +2.85 | 11.6 % |
| | | nodraft | 167.0 (5.99) | 166.2 (6.02) | 0.8 | +53.8 % | gain | | |
| 32,768 | 16,388 on 16,316 | draft (128) | 90.6 (11.04) | 90.4 (11.07) | 0.2 | | | +4.32 | |
| | | dub256 | 92.4 (10.83) | 92.1 (10.86) | 0.3 | +1.9 % | gain | +4.11 | 4.8 % |
| | | dub512 | 93.8 (10.66) | 93.6 (10.69) | 0.2 | +3.6 % | gain | +3.94 | 8.8 % |
| | | nodraft | 148.5 (6.73) | 148.7 (6.73) | 0.1 | +64.2 % | gain | | |
| 49,152 | 16,900 on 32,188 | draft (128) | 70.0 (14.29) | 70.1 (14.26) | 0.1 | | | +5.91 | |
| | | dub256 | 71.1 (14.07) | 70.9 (14.10) | 0.2 | +1.4 % | gain | +5.72 | 3.2 % |
| | | dub512 | 72.4 (13.81) | 72.2 (13.85) | 0.2 | +3.2 % | gain | +5.46 | 7.6 % |
| | | nodraft | 119.5 (8.37) | 119.6 (8.36) | 0.1 | +70.7 % | gain | | |
| all four | 50,124 tokens | draft (128) | 88.0 (11.36) | 88.7 (11.28) | 0.7 | | | +4.34 | |
| | | dub256 | 90.1 (11.10) | 89.8 (11.13) | 0.3 | +1.8 % | gain | +4.14 | 4.7 % |
| | | dub512 | 91.8 (10.90) | 91.5 (10.93) | 0.3 | +3.7 % | gain | +3.94 | 9.3 % |
| | | nodraft | 143.3 (6.98) | 143.3 (6.98) | 0.0 | +62.2 % | gain | | |

- **The four prompt chunks of a load took** 569.6 and 565.2 s with the preset, 556.2 and 557.9 with `dub256`, 546.2
  and 548.0 with `dub512` and 349.8 and 349.7 without the draft. The draft adds 217.7 s to a load with the preset and
  197.4 s with `dub512`, the means of each arm's two loads.
- **The micro-batch takes the same amount off every prompt token at every depth.** At 512 that is 0.40, 0.37, 0.38
  and 0.45 ms a prompt token at the four depths; at 256 it is 0.24, 0.20, 0.21 and 0.19 ms, about half.
- **The part that grows with depth does not change.** From an empty cache to 32,188 cached tokens the draft's cost
  rises by 3.60 ms a prompt token at 128, 3.65 at 256 and 3.55 at 512. A larger micro-batch means fewer passes over
  the draft layer's cache, so that rise is not in reading the cache once per micro-batch. It is the same for every
  prompt token whatever the batching, as the draft layer's attention is: every prompt token against every cached
  cell. That is a reading of these numbers, not a measurement; the second half of Q1215 profiles one prime.
- **At 8,192 the class is negligible because of one load:** `draft1` read its first chunk at 122.5 tok/s and `draft2`
  at 129.2, below both loads of `dub512` (132.8 and 131.9). The second pair's preset loads read the same chunk at
  127.7 and 127.2.
- **The preset repeats the second pair:** its four chunks took 569.6 and 565.2 s against 567.6 and 563.9 there; the
  line without the draft 349.8 and 349.7 against 350.0 and 350.4; at 49,152 the preset read 70.0 and 70.1 tok/s
  against 69.9 and 70.3.
- At depth 0 the 64-token tail reads at 32.8 and 32.8 tok/s with the preset, 32.8 and 31.8 with `dub256`, 32.9 and
  32.6 with `dub512`, and 33.5 and 33.8 without the draft.

### The host's memory

From the RAM watchdog's samples, one a second, over each whole load
([draft_prompt_cost/draft_ubatch_memory.tsv](draft_prompt_cost/draft_ubatch_memory.tsv), per request in
[draft_prompt_cost/host_memory.tsv](draft_prompt_cost/host_memory.tsv)). The watchdog's usable memory is MemAvailable
minus CmaFree; with the rig's kernel option `kho=off` the lowest of the two was the same in every load here.

| Load | MemAvailable after loading, MiB | Lowest MemAvailable | Most swap in use | Swap moved during the primes | Most written to swap in 10 s | Highest pressure, some / full |
|---|---:|---:|---:|---:|---:|---|
| `draft1` | 884 | 71 | 512 | +281 | 275 | 19.50 / 17.67 % |
| `draft2` | 930 | 81 | 1,012 | +754 | 208 | 6.17 / 4.48 % |
| `dub256_1` | 905 | 80 | 1,131 | +882 | 228 | 15.18 / 13.40 % |
| `dub256_2` | 904 | 79 | 1,147 | +969 | 206 | 11.15 / 9.77 % |
| `dub512_1` | 871 | 263 | 1,064 | +905 | 353 | 5.80 / 3.71 % |
| `dub512_2` | 873 | 238 | 1,097 | +880 | 284 | 10.29 / 8.28 % |
| `nodraft1` | 984 | 268 | 44 | 0 | 0 | 5.54 / 5.33 % |
| `nodraft2` | 1,025 | 257 | 46 | 0 | 0 | 0.36 / 0.29 % |

- **With 512 the host is not lower at its lowest point, and the watchdog stayed far from its limits.** Its stop
  points are MemAvailable under 60 MiB, usable memory under 20 MiB, or more than 600 MiB written to swap within
  10 s. With `dub512` the lowest MemAvailable was 263 and 238 MiB, as without the draft (268 and 257), against 71 and
  81 with the preset and 80 and 79 with `dub256`.
- **With 512 the host swaps earlier.** During the first prime at 8,192 it wrote 558 and 606 MiB to swap (the preset
  255 and 62, `dub256` 125 and 135), at most 353 and 284 MiB within 10 s (the preset 275 and 208). It was at its
  lowest during that prime, and from the prime at 16,384 on its MemAvailable stayed above 500 MiB. Over the four
  primes of a load the swap moved is alike: 905 and 880 MiB at 512, 281 and 754 at 128, 882 and 969 at 256.
- **The preset's lowest points:** in `draft1` 147 MiB during the prime and the first decode request at 8,192, 84
  during those at 16,384 and 71 during the prime at 49,152; in `draft2` 84 and 81 MiB during the prime and the first
  decode request at 8,192.
- **After loading** the host had 871 and 873 MiB available with 512 against 884 and 930 with the preset.
- **On the cards** the larger micro-batch takes 48 MiB more on card 8 and 2 more on card 4 at 256, and 146 and 4 more
  at 512: card 8 holds 4,711 MiB of its 6,144 after loading with `dub512` against 4,565 with the preset. Every other
  card holds the same.
- This run's decode requests write 16 tokens, so the first request's swapping that the second pair saw over 256
  tokens is only partly in it. A whole agent session on the 4 GB host is the agent check's (registry row Q1233).

### Generation, a check

- Every request generated its 16 tokens. At depth 0 all eight loads wrote the same reply, and every load of the three
  draft arms accepted 14 of its 46 drafted tokens over the two passes. The step on the repeated request at depth 0
  took 113.71 and 112.97 ms with the preset, 112.97 and 113.00 with `dub256` and 113.11 and 113.32 with `dub512`.
- From 8,192 tokens on, two loads of one arm wrote different replies, as in the earlier runs, and a 16-token request
  takes 6 to 9 steps. The step tables of [draft_prompt_cost/table.md](draft_prompt_cost/table.md) are not read for a
  class at that count. Each load's speed, acceptance and reply hash are in
  [draft_prompt_cost/draft_ubatch_generation.tsv](draft_prompt_cost/draft_ubatch_generation.tsv).

### What it decides

- **`--spec-draft-n-ubatch 512` is a gain in prompt reading, at no cost in the host's lowest memory and 146 MiB more
  on card 8.** It goes into the stacked line. It goes into the preset after the agent check on the rig, because
  release settings are chosen on agentic speed and only a whole agent session shows the fit on the 4 GB host
  (registry row Q1233). As 512 is llama.cpp's default, the change to the preset is to drop
  `--spec-draft-n-ubatch 128`.
- `dub256` is a gain too (+1.8 % over the primes) and takes away half as much as 512.
- **What is left for code:** of the 217.7 s the preset's draft adds to a load's 50,124 prompt tokens, 512 takes away
  20.3 s, and 197.4 s (90.7 %) are left. That is the ceiling for the KV-only catch-up (Q1105) and for the parts the
  profile in Q1215 names.

**Limits.**
- Two loads an arm, one day, one rig, the second pair's prompts.
- Decode requests of 16 tokens: generation here is a check, not a speed.
- The host's memory is one sample a second; the run does not show what was swapped.
- No quality is measured here. The draft's micro-batch changes at most which tokens the draft proposes; the main model
  checks every one.

## The sixth run: prompt micro-batches through the sparse kernel (Q1214)

In the first pair the sparse switch did not change prompt reading, and prompt reading slowed with depth in both arms.
The sparse kernel takes a call of at most 8 query tokens (`GGML_CUDA_FATTN_SPARSE_PASCAL_NQ`, default 8), and in mode
1 only when the cache holds at least 1.5 × 2,051 cells per query token (`_MIN_RATIO`): 393,792 cells for a
128-token micro-batch, beyond this line's context. So every prompt micro-batch of 512 tokens went to the dense
kernel, which computes each query token against every cell. This run read the prompts in micro-batches of 128
tokens through the sparse kernel: mode 2 (`GGML_CUDA_FATTN_SPARSE_PASCAL=2`) drops the ratio rule and sends every
call within the kernel's limits to it, `_NQ=128` lets 128 query tokens in, and `-ub 128` makes the micro-batches that
size (`sp128`). A second arm, `-ub 128` with the switch at 1 as in the control (`ub128`), takes the micro-batch alone.
The control `ub512` is the preset with the switch at 1, the line of the fourth run's `draft` arm. Depths 0, 16,384,
32,768 and 49,152, 16 tokens a decode request, two loads an arm after a warm-up load.

**With its prompt micro-batches through the sparse kernel the line reads prompts 46.7 % faster over a load's three
prompt chunks (131.1 and 131.3 tok/s against the preset's 89.5 and 89.4; 376.5 and 375.8 s against 559.9 and
560.6 s), and its prompt speed no longer falls with depth: 128.2 and 128.3 tok/s on an empty cache, 133.8 and 133.9
on 16,188 cached tokens and 131.3 and 131.9 on 32,572, against 123.8 and 123.7, 90.2 and 90.3, and 70.2 and 70.0 with
the preset. At 49,152 tokens it reads prompts faster with the draft than the line without the draft did in the fourth
run (119.5 and 119.6 tok/s on the same cache). `-ub 128` with the dense kernel is a gain at depth (+9.1 % and
+15.1 %) and a loss on an empty cache (−4.3 %). Mode 2 sends the decode steps to the kernel too, and at depth 0 the
repeated request's step takes 282.35 and 282.53 ms against 113.12 and 113.14: it is not a setting to serve, and the
prompt path needs a rule of its own in the switch.**

### Prompt reading: tok/s (ms a prompt token)

Per load, the tokens its prompt request read over their time. Each arm is classed against `ub512` by the rule of the
first pair. Beside: the fourth run's loads that read the same chunk on the same cached tokens, not classed. Gap
closed: the share of the gap between `ub512` and the fourth run's line without the draft, in ms a prompt token, that
an arm closes ([prompt_depth/prompt_depth.md](prompt_depth/prompt_depth.md),
[prompt_depth/prompt_depth_prompt.tsv](prompt_depth/prompt_depth_prompt.tsv)).

| Depth (tokens) | Read, on cached | Arm | Load 1 | Load 2 | Spread, tok/s | Against `ub512` | Class | Gap closed |
|---:|---|---|---|---|---:|---:|---|---:|
| 0 | 64 on 0 | ub512 | 32.7 (30.57) | 28.0 (35.70) | 4.7 | | | |
| | | ub128 | 32.4 (30.84) | 32.7 (30.62) | 0.2 | +7.2 % | negligible | |
| | | sp128 | 32.6 (30.69) | 32.1 (31.14) | 0.5 | +6.6 % | negligible | |
| | | the fourth run's `draft` (beside) | 32.8 (30.52) | 32.8 (30.45) | 0.1 | +8.0 % | | |
| | | the fourth run's `nodraft` (beside) | 33.5 (29.82) | 33.8 (29.57) | 0.3 | +10.9 % | | |
| 16,384 | 16,320 on 0 | ub512 | 123.8 (8.07) | 123.7 (8.08) | 0.1 | | | |
| | | ub128 | 118.2 (8.46) | 118.6 (8.43) | 0.3 | −4.3 % | loss | |
| | | sp128 | 128.2 (7.80) | 128.3 (7.79) | 0.1 | +3.6 % | gain | |
| 32,768 | 16,900 on 15,804 | ub512 | 90.2 (11.09) | 90.3 (11.08) | 0.1 | | | |
| | 16,516 on 16,188 | ub128 | 98.5 (10.15) | 98.5 (10.15) | 0.0 | +9.1 % | gain | |
| | 16,516 on 16,188 | sp128 | 133.8 (7.47) | 133.9 (7.47) | 0.0 | +48.3 % | gain | |
| 49,152 | 16,900 on 32,188 | ub512 | 70.2 (14.25) | 70.0 (14.29) | 0.2 | | | |
| | 16,516 on 32,572 | ub128 | 80.5 (12.42) | 80.7 (12.38) | 0.2 | +15.1 % | gain | 31.6 % |
| | 16,516 on 32,572 | sp128 | 131.3 (7.62) | 131.9 (7.58) | 0.6 | +87.8 % | gain | 112.9 % |
| | | the fourth run's `draft` (beside) | 70.0 (14.29) | 70.1 (14.26) | 0.1 | −0.0 % | | |
| | | the fourth run's `nodraft` (beside) | 119.5 (8.37) | 119.6 (8.36) | 0.1 | +70.6 % | | |
| all three primes | 50,120 tokens (`ub512`), 49,352 (`ub128`, `sp128`) | ub512 | 89.5 (559.9 s) | 89.4 (560.6 s) | 0.1 | | | |
| | | ub128 | 96.6 (510.8 s) | 96.8 (509.9 s) | 0.2 | +8.1 % | gain | |
| | | sp128 | 131.1 (376.5 s) | 131.3 (375.8 s) | 0.2 | +46.7 % | gain | |

- **The micro-batch moves the chunk a little.** The server restores the checkpoint it took 4 or `-ub` + 4 tokens
  before the end of the previous prime, so with `-ub 128` the primes at 32,768 and 49,152 read 16,516 tokens on
  16,188 and 32,572 cached, and with `-ub 512` 16,900 on 15,804 and 32,188: 384 tokens fewer, starting 384 tokens
  deeper. The tok/s compare the same kind of chunk per token. At 16,384 every arm read the same 16,320 tokens on an
  empty cache, since the depth-0 tail is not a prefix of the deeper prompts.
- **The sparse path makes prompt reading flat with depth.** `sp128_1` reads at 7.80, 7.47 and 7.62 ms a prompt token
  at the three depths, `ub512_1` at 8.07, 11.09 and 14.25. From the means of each arm's two loads, `sp128` takes
  0.28 ms off a prompt token on the chunk from an empty cache, 3.61 on 15,804 to 16,188 cached tokens and 6.67 on
  32,188 to 32,572.
- **Both models' attention.** The kernel also takes the MTP draft's attention
  ([2026-11-10_step_parts_depth](../2026-11-10_step_parts_depth/README.md)), so the micro-batches of 128 that the
  draft reads (`--spec-draft-n-ubatch 128` in every arm) go through it too. The fourth run found that the draft's
  prompt cost rises by 3.60 ms a prompt token from an empty cache to 32,188 cached tokens, and the line without the
  draft read 5.65 and 5.63 ms a prompt token on an empty cache and 8.37 and 8.36 there. With `sp128` the line with
  the draft reads 7.62 and 7.58 ms a prompt token on 32,572 cached tokens, 0.76 ms under the line without the draft on
  the dense path (the means): the rise of both is gone. Which part each model's attention had is not split here (registry row Q1249 runs `sp128` without the
  draft).
- **On an empty cache the gain is small:** +3.6 % at 16,384, where the cache grows from 0 to 16,320 cells within the
  chunk and the dense kernel has few cells to read in its first micro-batches. `-ub 128` alone is a loss there
  (−4.3 %); a micro-batch of 128 reads the weights four times as often per prompt token as one of 512, which would
  explain it (a reading, not measured).
- **The control repeats the fourth run** at 49,152 (70.2 and 70.0 tok/s against 70.0 and 70.1 on the same chunk and
  cache). The fourth run's chunks at 16,384 and 32,768 were others (8,708 on 7,612 and 16,388 on 16,316), so they
  are not beside.
- **Which kernel took a call is read on the times.** The server log does not say, and the kernel hands a call beyond
  its limits to the dense kernel without a message. `sp128`'s times differ from `ub128`'s at every depth past 0
  (+8.3 %, +35.9 % and +63.2 % in tok/s), so its calls went to the sparse kernel.

### The host's memory and the cards

From the RAM watchdog's samples, one a second, over each whole load
([prompt_depth/prompt_depth_memory.tsv](prompt_depth/prompt_depth_memory.tsv), per request in
[prompt_depth/host_memory.tsv](prompt_depth/host_memory.tsv)):

| Load | MemAvailable after loading, MiB | Lowest MemAvailable | Most swap in use | Swap moved during the primes | Most written to swap in 10 s | Highest pressure, some / full |
|---|---:|---:|---:|---:|---:|---|
| `ub512_1` | 896 | 220 | 1,061 | +929 | 242 | 10.05 / 8.96 % |
| `ub512_2` | 911 | 622 | 1,036 | +833 | 319 | 12.70 / 11.09 % |
| `ub128_1` | 996 | 67 | 282 | +232 | 129 | 3.42 / 2.62 % |
| `ub128_2` | 1,016 | 246 | 1,075 | +835 | 262 | 6.49 / 4.97 % |
| `sp128_1` | 1,013 | 69 | 635 | +590 | 240 | 9.53 / 9.31 % |
| `sp128_2` | 970 | 92 | 331 | +116 | 156 | 6.69 / 5.86 % |

- **The watchdog never acted** (its stop points: MemAvailable under 60 MiB, usable memory under 20 MiB, more than
  600 MiB written to swap within 10 s). The lowest points were 67 MiB in `ub128_1` (the prime and the first decode
  request at 49,152) and 69 MiB in `sp128_1` (the prime and the first decode request at 32,768).
- **The lowest point moves with when the host swaps, not with the arm alone.** The preset's two loads here swapped
  833 and 929 MiB during the primes, most of it in the first, and stayed at 220 MiB and above; the same line's loads
  in the fourth run fell to 71 and 81 MiB. With `-ub 128` the host has about 100 MiB more available after loading
  (996, 1,016, 1,013 and 970 MiB against 896 and 911), and three of the four loads swapped less during the primes.
- **On the cards** `-ub 128` holds 200 MiB less on each card after loading and 248 less on card 0 (4,811 against
  5,059 MiB; 42,171 against 44,019 MiB over the nine cards); `sp128` holds 2 MiB more than `ub128` on seven of the
  cards ([prompt_depth/prompt_depth.md](prompt_depth/prompt_depth.md)).

### Generation, a check

- Every request generated its 16 tokens. Over all depths generation is negligible in both arms: 11.73 and 10.01 tok/s
  (`ub128`) and 11.43 and 10.35 (`sp128`) against 11.18 and 10.81
  ([prompt_depth/table.md](prompt_depth/table.md)).
- **Mode 2 is a loss in decode at depth 0:** the repeated request's step takes 282.35 and 282.53 ms with `sp128`,
  113.11 and 113.19 with `ub128`, and 113.12 and 113.14 with the preset. Mode 2 sends a verify pass over a cache of
  64 cells to the sparse kernel, where mode 1's threshold keeps it on the dense kernel (registry row Q1212 times that
  threshold).
- **At depth 0 `sp128` wrote another greedy reply** (`95afbd…`, in both loads) than the preset and `ub128` (`9da71b…`,
  in all four loads, as in the fourth run). `-ub 128` alone did not change it there, so the other reply comes from
  the kernel's arithmetic on the tail or on the decode steps. From 16,384 tokens on, loads of one arm wrote different
  replies, as in the earlier runs. No quality is measured here (registry row Q1248).
- The step tables of [prompt_depth/table.md](prompt_depth/table.md) are not read for a class: a 16-token request
  takes 6 to 9 steps, and the first request after each prime moved with the host's swapping.

### What it decides

- **The rule in the row: a gain from the sparse prompt path → Q643's union gather for prompt micro-batches.** The
  gain is there: +46.7 % over the primes, +87.8 % on 32,572 cached tokens. Each query token of a micro-batch now reads
  its own list of about 2,051 cells. If the neighbouring query tokens of one micro-batch share much of their lists
  (Q643's premise for verify passes, not measured for prompts), one gather over the union of the lists reads each cell
  once and lets 512-token micro-batches through without a smaller batch.
- **Without code first:** micro-batches of 256 and 512 tokens through the kernel (`_NQ=256` and `_NQ=512`). The
  kernel takes up to 1,024 columns of query token × KV head a call, and the model's attention layers have 2 KV heads
  for 24 query heads, so 512 query tokens fit. The kernel's cases at those sizes come first, off the rig (registry
  row Q1245).
- **Mode 2 is not for serving:** a prompt rule in the switch (calls of more than 8 query tokens to the kernel from a
  depth on, mode 1's rule for verify passes; Q1246), then the agent check on the rig (Q1247), whether the kernel's
  prompt path changes the model's output (Q1248), the split between the two models (Q1249), and a 131,072-token
  line at `-ub 128` (Q1250).

**Limits.**
- Two loads an arm, one day, one rig, the second pair's prompts.
- Decode requests of 16 tokens: generation here is a check, not a speed.
- The chunks of the `-ub 128` arms are 384 tokens shorter and start 384 tokens deeper than the control's at 32,768
  and 49,152.
- Which kernel took a call is read on the times.
- No quality is measured.

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
- **The third run's line:** the second pair's two lines with `-c 131072 -ub 256` (`CTX=131072 UB=256`); nothing
  else changed ([ctx131k/loads.tsv](ctx131k/loads.tsv)). The prompts are cut from both arms of the release check's
  recorded requests (`TAG='^(t1swift|t1base)/'`, 24 sessions), since one arm's 12 sessions hold too few tokens for
  120,000.
- **The third run's steps:** the projection of four lines (131,072 and 65,536 tokens, with and without the draft,
  seconds a line), then `load_check`: one load of each line at depth 0 (`WARM=0 REPS=1 DEPTHS=0`), with the cards'
  memory read after loading and the watchdog on. Then the deep lane at 49,152, 65,536 and 120,000 tokens: a warm-up
  load of `draft`, then `nodraft1` and `draft1` (`LOADS="warm:draft nodraft1:nodraft draft1:draft"`), the line
  without the draft first, so that its numbers were in even if the draft line ran out of host memory. Depth 49,152
  is there to pair with the first two runs; its two decode requests cost about 40 s a load, since the prime to 65,536
  reads its tokens anyway. Each load was checked quiet before it (no llama process, 0 MiB on every card). Loading
  took 225 s with the draft and 208 s without; `nodraft1` took 26.9 minutes, `draft1` 36.9, the run 68 minutes.
- **The third run's tables:** the driver's own ([ctx131k/table.md](ctx131k/table.md); its control is `nodraft`, the
  first timed arm); [run/draft_pair.py](run/draft_pair.py), [run/host_memory.py](run/host_memory.py) and
  [run/card_fit.py](run/card_fit.py), the last of which puts the projection, the memory after loading and the most
  each card held during a load into one table. They ran on another machine and gave the rig's own tables again,
  byte for byte.
- **The fourth run's line:** the second pair's `draft` line (the preset with `GGML_CUDA_FATTN_SPARSE_PASCAL=1`) with
  `--spec-draft-n-ubatch 256` or `512` in place of the preset's `128` (`ARM_ARGS_<arm>`), and its `nodraft` line;
  nothing else changed ([draft_prompt_cost/loads.tsv](draft_prompt_cost/loads.tsv)). The second pair's depths and
  prompts, token for token; `N_PREDICT=16`, so each decode request writes 16 tokens.
- **The fourth run's loads:** one untimed warm-up load of `draft`, then `draft1`, `dub256_1`, `nodraft1`, `dub512_1`,
  `draft2`, `dub256_2`, `nodraft2`, `dub512_2` (`LOADS`), each after the quiet check. `dub512` came last in each
  round so that, had the watchdog stopped it, the round's other loads would have been in. Loading took 224 and 224 s
  for `draft`, 226 and 224 for `dub256`, 224 and 225 for `dub512`, 210 and 208 for `nodraft`; a whole load 14.2 and
  14.1 minutes, 14.1 and 14.0, 13.8 and 13.8, and 10.0 and 10.0; the run 108 minutes. The watchdog did not act.
- **The fourth run's tables:** the driver's own ([draft_prompt_cost/table.md](draft_prompt_cost/table.md), control
  `draft`), rebuilt on another machine byte for byte; [run/draft_ubatch.py](run/draft_ubatch.py) for the prompt,
  memory and generation tables of this page, and [run/host_memory.py](run/host_memory.py) per request. On the
  second pair's folder `draft_ubatch.py` gives that pair's prompt and host-memory numbers again.
- **The sixth run's line:** the preset with one variable set per arm (`ARM_ENV_<arm>`, no `ENV_ADD`, so that each
  line holds the switch once): `GGML_CUDA_FATTN_SPARSE_PASCAL=1` for `ub512` and `ub128`,
  `GGML_CUDA_FATTN_SPARSE_PASCAL=2 GGML_CUDA_FATTN_SPARSE_PASCAL_NQ=128` for `sp128`; `-ub 128` for `ub128` and
  `sp128` (`ARM_UB_<arm>`); nothing else changed ([prompt_depth/loads.tsv](prompt_depth/loads.tsv)). `ub512` is the
  fourth run's `draft` line. The prompts are the earlier runs' at depths 0, 16,384, 32,768 and 49,152;
  `N_PREDICT=16`.
- **The sixth run's loads:** one untimed warm-up load of `ub512`, then `ub512_1`, `ub128_1`, `sp128_1`, `ub512_2`,
  `ub128_2`, `sp128_2` (`LOADS`), each after the quiet check (no llama process, 0 MiB on every card). `sp128` came
  last in each round, since it runs the kernel on calls it had not served before. Loading took 224 to 226 s; a whole
  load 13.9 and 13.9 minutes (`ub512`), 13.0 and 13.1 (`ub128`), 10.8 and 10.8 (`sp128`); the run 80 minutes. The
  watchdog did not act.
- **The sixth run's tables:** the driver's own ([prompt_depth/table.md](prompt_depth/table.md), control `ub512`),
  rebuilt on another machine byte for byte; [run/prompt_depth.py](run/prompt_depth.py) for the prompt, memory and
  generation tables of this page, with the fourth run's `draft` and `nodraft` loads beside, and
  [run/host_memory.py](run/host_memory.py) per request. On the fourth run's folder `prompt_depth.py` gives that run's
  prompt classes, its share taken away (as the gap closed) and its host memory again.

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
| [ctx131k/table.md](ctx131k/table.md), [classes.tsv](ctx131k/classes.tsv), [requests.csv](ctx131k/requests.csv), [loads.tsv](ctx131k/loads.tsv) | the third run's driver tables, requests and loads (`warm`, `nodraft1`, `draft1` at 49,152, 65,536 and 120,000 tokens) |
| [ctx131k/draft_pair.md](ctx131k/draft_pair.md), [_classes.tsv](ctx131k/draft_pair_classes.tsv), [_breakeven.tsv](ctx131k/draft_pair_breakeven.tsv), [_prompt.tsv](ctx131k/draft_pair_prompt.tsv), [_turn.tsv](ctx131k/draft_pair_turn.tsv), [_loads.tsv](ctx131k/draft_pair_loads.tsv) | the third run read part by part, as for the second pair, with an estimate at the sampler's yield at 49,152 |
| [ctx131k/host_memory.tsv](ctx131k/host_memory.tsv) | the third run's host memory per load and request |
| [ctx131k/card_fit.tsv](ctx131k/card_fit.tsv) | MiB per card: the projection of four lines, and after loading and at the highest for every load of the third run, its load check and the second pair |
| [ctx131k/load_check/](ctx131k/load_check/table.md) (`table.md`, `classes.tsv`, `requests.csv`, `loads.tsv`, `host_memory.tsv`) | the load check: one load of each 131,072-token line at depth 0 |
| [run/card_fit.py](run/card_fit.py) | the card table's script (Python 3.9 or newer, standard library only) |
| [draft_prompt_cost/table.md](draft_prompt_cost/table.md), [classes.tsv](draft_prompt_cost/classes.tsv), [requests.csv](draft_prompt_cost/requests.csv), [loads.tsv](draft_prompt_cost/loads.tsv) | the fourth run's driver tables and classes (control `draft`), requests and loads (`warm`, then two loads each of `draft`, `dub256`, `nodraft` and `dub512`) |
| [draft_prompt_cost/draft_ubatch.md](draft_prompt_cost/draft_ubatch.md), [_prompt.tsv](draft_prompt_cost/draft_ubatch_prompt.tsv), [_memory.tsv](draft_prompt_cost/draft_ubatch_memory.tsv), [_generation.tsv](draft_prompt_cost/draft_ubatch_generation.tsv) | the fourth run read on prompt reading (every load by value, the class against `draft`, what the draft adds per prompt token, the share taken away), on the host's memory per load and each card's after loading, and on generation as a check |
| [draft_prompt_cost/host_memory.tsv](draft_prompt_cost/host_memory.tsv) | the fourth run's host memory per load and request |
| [run/draft_ubatch.py](run/draft_ubatch.py) | the fourth run's table script (Python 3.9 or newer, standard library only) |
| [prompt_depth/table.md](prompt_depth/table.md), [classes.tsv](prompt_depth/classes.tsv), [requests.csv](prompt_depth/requests.csv), [loads.tsv](prompt_depth/loads.tsv) | the sixth run's driver tables and classes (control `ub512`), requests and loads (`warm`, then two loads each of `ub512`, `ub128` and `sp128`) |
| [prompt_depth/prompt_depth.md](prompt_depth/prompt_depth.md), [_prompt.tsv](prompt_depth/prompt_depth_prompt.tsv), [_memory.tsv](prompt_depth/prompt_depth_memory.tsv), [_generation.tsv](prompt_depth/prompt_depth_generation.tsv) | the sixth run read on prompt reading (every load by value, the class against `ub512`, the fourth run's loads beside, the gap closed), on the host's memory per load and each card's after loading, and on generation as a check (tok/s and ms a step on both passes) |
| [prompt_depth/host_memory.tsv](prompt_depth/host_memory.tsv) | the sixth run's host memory per load and request |
| [run/prompt_depth.py](run/prompt_depth.py) | the sixth run's table script (Python 3.9 or newer, standard library only) |

Not published: the prompts' token ids, the replies' first characters and the server logs (recorded text and paths),
the raw telemetry (the cards every 10 s, the watchdog's samples, the quiet and fit checks, which the text above
summarizes) and the fit probe's raw log (paths; its numbers are in `ctx131k/card_fit.tsv`).

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

The third run. First the projection per card (`BUILDS` is the rig's folder of builds, `BUILD` the release build's
folder in it; `fit_probe.sh` adds `-ngl 99 -fa on -np 1 -fit on --ctx-checkpoints 4 --cache-ram 0`):

```bash
BUILD=<the release build's folder name>
E="GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768 GGML_CUDA_MMVQ_PASCAL_KCOL_TABLE=0 LLAMA_HOP_CAST=q8_0 GGML_CUDA_HOP_STAGED=1 LLAMA_DRAFT_SHORTLIST=32768,8192,2 GGML_CUDA_KEEPALIVE=decode GGML_CUDA_FATTN_SPARSE_PASCAL=1"
P="-bs -b 2048 -ts 6,6,6,6,5,6,5,6,3 -lm dio --no-host"
DR="--spec-type draft-mtp --spec-draft-n-max 3 -otd token_embd=CUDA4 --spec-draft-n-ubatch 128 -md <models>/mtp-Qwen3.8-Flash-Next-draft.gguf"
RIG=rig MODEL=<models>/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf BUILDS=<builds> OUT=fit_probe.md KEEP=400 \
CONFIGS="draft131k|$BUILD|131072|$P -ub 256 $DR|$E;nodraft131k|$BUILD|131072|$P -ub 256|$E;draft65k|$BUILD|65536|$P -ub 512 $DR|$E;nodraft65k|$BUILD|65536|$P -ub 512|$E" \
bash scripts/fit_probe.sh
```

Then the load check and the deep lane, each with the `wait` and `collect` commands above (`DRY=1` first):

```bash
RIG=rig BIN=<build>/bin/llama-server \
MODEL=<models>/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf \
MODEL_DRAFT=<models>/mtp-Qwen3.8-Flash-Next-draft.gguf \
TEXT_LOCAL=<recorded>/proxy.jsonl.bodies.gz TAG='^(t1swift|t1base)/' WD_ENV=MIN_AVAIL_MB=60 \
CTX=131072 UB=256 ENV_ADD=GGML_CUDA_FATTN_SPARSE_PASCAL=1 ARMS="draft nodraft" ARM_DRAFT_nodraft=off \
WARM=0 REPS=1 DEPTHS=0 \
bash bench/deep_lane/remote.sh start
```

```bash
RIG=rig BIN=<build>/bin/llama-server \
MODEL=<models>/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf \
MODEL_DRAFT=<models>/mtp-Qwen3.8-Flash-Next-draft.gguf \
TEXT_LOCAL=<recorded>/proxy.jsonl.bodies.gz TAG='^(t1swift|t1base)/' WD_ENV=MIN_AVAIL_MB=60 \
CTX=131072 UB=256 ENV_ADD=GGML_CUDA_FATTN_SPARSE_PASCAL=1 ARMS="draft nodraft" ARM_DRAFT_nodraft=off \
DEPTHS="49152 65536 120000" LOADS="warm:draft nodraft1:nodraft draft1:draft" \
bash bench/deep_lane/remote.sh start
```

Its tables (the yield is the sampler pass's tokens a step at 49,152 in
[2026-11-10_draft_acceptance_by_depth/pooled.tsv](../2026-11-10_draft_acceptance_by_depth/pooled.tsv)):

```bash
python3 -I bench/deep_lane/deep_table.py <deep lane folder> --out table.md --tsv classes.tsv
python3 -I docs/data/2026-11-10_deep_lane/run/draft_pair.py <deep lane folder> \
  --yield "the sampler off the rig, at 49,152:49152=2.876,65536=2.876,120000=2.876" --out draft_pair.md --tsv draft_pair
python3 -I docs/data/2026-11-10_deep_lane/run/host_memory.py <deep lane folder> > host_memory.tsv
python3 -I docs/data/2026-11-10_deep_lane/run/card_fit.py --probe fit_probe.md <load check folder> <deep lane folder> \
  <second pair's folder> > card_fit.tsv
```

The fourth run, with the `wait` and `collect` commands above (`DRY=1` first; its four server lines differ only in
`--spec-draft-n-ubatch` and, for `nodraft`, in the draft's flags):

```bash
RIG=rig BIN=<build>/bin/llama-server \
MODEL=<models>/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf \
MODEL_DRAFT=<models>/mtp-Qwen3.8-Flash-Next-draft.gguf \
TEXT_LOCAL=<recorded>/proxy.jsonl.bodies.gz TAG='^t1swift/' WD_ENV=MIN_AVAIL_MB=60 \
ENV_ADD=GGML_CUDA_FATTN_SPARSE_PASCAL=1 N_PREDICT=16 \
ARMS="draft dub256 dub512 nodraft" ARM_ARGS_dub256="--spec-draft-n-ubatch 256" \
ARM_ARGS_dub512="--spec-draft-n-ubatch 512" ARM_DRAFT_nodraft=off \
LOADS="warm:draft draft1:draft dub256_1:dub256 nodraft1:nodraft dub512_1:dub512 draft2:draft dub256_2:dub256 nodraft2:nodraft dub512_2:dub512" \
bash bench/deep_lane/remote.sh start
```

Its tables:

```bash
python3 -I bench/deep_lane/deep_table.py <results folder> --out table.md --tsv classes.tsv
python3 -I docs/data/2026-11-10_deep_lane/run/draft_ubatch.py <results folder> --out draft_ubatch.md --tsv draft_ubatch
python3 -I docs/data/2026-11-10_deep_lane/run/host_memory.py <results folder> > host_memory.tsv
```

The sixth run, with the `wait` and `collect` commands above (`DRY=1` first; its three server lines differ only in
`-ub` and the switch's variables):

```bash
RIG=rig BIN=<build>/bin/llama-server \
MODEL=<models>/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf \
MODEL_DRAFT=<models>/mtp-Qwen3.8-Flash-Next-draft.gguf \
TEXT_LOCAL=<recorded>/proxy.jsonl.bodies.gz TAG='^t1swift/' WD_ENV=MIN_AVAIL_MB=60 \
DEPTHS="0 16384 32768 49152" N_PREDICT=16 ARMS="ub512 ub128 sp128" \
ARM_ENV_ub512=GGML_CUDA_FATTN_SPARSE_PASCAL=1 \
ARM_ENV_ub128=GGML_CUDA_FATTN_SPARSE_PASCAL=1 ARM_UB_ub128=128 \
ARM_ENV_sp128="GGML_CUDA_FATTN_SPARSE_PASCAL=2 GGML_CUDA_FATTN_SPARSE_PASCAL_NQ=128" ARM_UB_sp128=128 \
LOADS="warm:ub512 ub512_1:ub512 ub128_1:ub128 sp128_1:sp128 ub512_2:ub512 ub128_2:ub128 sp128_2:sp128" \
bash bench/deep_lane/remote.sh start
```

Its tables (`<fourth run>` is the fourth run's results folder, for its loads beside):

```bash
python3 -I bench/deep_lane/deep_table.py <results folder> --out table.md --tsv classes.tsv
python3 -I docs/data/2026-11-10_deep_lane/run/prompt_depth.py <results folder> --control ub512 \
  --beside "<fourth run>:draft:draft (the fourth run)" --beside "<fourth run>:nodraft:nodraft (the fourth run)" \
  --floor "nodraft (the fourth run)" --out prompt_depth.md --tsv prompt_depth
python3 -I docs/data/2026-11-10_deep_lane/run/host_memory.py <results folder> > host_memory.tsv
```

## Provenance and licence

The recorded requests are this project's release check of 2026-11-05
([docs/data/2026-11-05_release_flashnext](../2026-11-05_release_flashnext/README.md)): the agent's sessions on 12
bug-fix tasks (Exercism exercises, MIT). The files here hold token counts, times, the server's counters and hashes;
no prompt or reply text. The scripts are under this repository's licence.
