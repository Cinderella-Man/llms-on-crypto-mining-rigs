# The deep lane on the rig: the sparse Pascal attention off and on, at 0 to 49,152 tokens of context (Q1143, Q230; 2026-11-10)

What this is: the first timing of the sparse Pascal attention (`GGML_CUDA_FATTN_SPARSE_PASCAL=1`, off by default) on
a real model. The line is the Miner Llama v3 release preset
([release/flash-next-v3/settings.env](../../../release/flash-next-v3/settings.env): Qwen3.8-Flash-Next, Swift Q2_0, the
MTP draft of three tokens, a 65,536-token context) on nine P106-090 cards, build `pascal-p106-v3-b12076-ladder5`
`1551f5fe1`. [bench/deep_lane](../../../bench/deep_lane/README.md) holds the context at 0, 8,192, 16,384, 32,768 and
49,152 tokens and generates 256 tokens after each, with the switch off (`dense`) and on (`sparse`), two server loads
an arm. Registry rows Q1143 (the lane) and Q230 (the kernel) in
[quants/experiments.md](../../../quants/experiments.md).

Why: on a P106 llama.cpp's attention reads the whole cache for every token, so generation slows as an agent's
context grows, and real coding attempts write 40,000 tokens and more. The model's attention layers keep about 2,051
cells per query token; the sparse kernel reads only those. It had passed its cases natively on a P106
([2026-11-10_replay_checks_rig2](../2026-11-10_replay_checks_rig2/README.md)) and written the dense kernels' text on
the whole model on newer cards ([2026-11-09_sparse_attn_whole_model](../2026-11-09_sparse_attn_whole_model/README.md)),
and had never been timed.

These are single-request diagnostics at a known depth. They do not replace the agent check.

## Result

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

## Files

| File | What |
|---|---|
| [dense_sparse/table.md](dense_sparse/table.md) | the driver's tables: generation per depth, arm, load and pass with ms a token, the step on pass 1 and pass 2, prompt reading, the draft's counters per load, the checks |
| [dense_sparse/classes.tsv](dense_sparse/classes.tsv) | every class as a table: metric (`generation`, `generation_all`, `step_pass1`, `step_pass2`, `prompt`), depth, each load's value, spread, difference, percent, class |
| [dense_sparse/requests.csv](dense_sparse/requests.csv) | one line per request of the five loads: prompt tokens sent, reused and read with their time, tokens generated with their time, the draft's counters, hashes of the prompt and the reply |
| [dense_sparse/loads.tsv](dense_sparse/loads.tsv) | one line per load: arm, start, seconds to load, the server line as it ran |

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

## Provenance and licence

The recorded requests are this project's release check of 2026-11-05
([docs/data/2026-11-05_release_flashnext](../2026-11-05_release_flashnext/README.md)): the agent's sessions on 12
bug-fix tasks (Exercism exercises, MIT). The files here hold token counts, times, the server's counters and hashes;
no prompt or reply text. The scripts are under this repository's licence.
