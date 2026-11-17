# The first request at a new depth without the keep-alive's stop (R191; Q1562; 2026-11-17)

What this measured: on Miner Llama v4.0 with Flash-Next's released preset, a first request at 4,096 tokens of context
generates about 291 ms slower than the same request repeated. R192
([2026-11-17_r192_first_request_graphs](../2026-11-17_r192_first_request_graphs/README.md)) found the time in three
passes that run kernel by kernel in both requests but take about twice as long in the first, where the keep-alive
switch `GGML_CUDA_KEEPALIVE_EAGER=1` (C811) had stopped each card's clock-holding block before them. This run repeats
R192's pair with one change, `GGML_CUDA_KEEPALIVE_EAGER=0` (the switch counts, stops nothing), to see whether the stop
is the whole cost and whether, without it, a kernel's first launch waits about 1 s for a running block (FN85). Nine
P106-090 cards, UkisAI's Swift-1.5 GSQ-RCO Q2_0 file with its MTP draft, the v4 preset otherwise unchanged. Registry
row Q1562 in [quants/experiments.md](../../../quants/experiments.md).

One load against R192's one load; every value below is from these two unless a row says otherwise.

## Result

**The stop is the whole cost of those passes, but it is also still needed in one place.** Without it, the first
request's kernel-by-kernel passes run as fast as the repeat's: the first verify pass 100.67 ms of GPU time on the nine
cards against the repeat's 102.94 (R192, with the stop: 219.92 against 118.12). But the MTP draft's first evaluation
after the prompt, on card 8, then waited 1,003.87 ms for the card's running block (R192: 6.64 ms), so this first
request generated 1,053.35 ms slower than its repeat, against R192's 290.97. So the stop must stay where a kernel is
new to the card and go where only the cache's size is new. That is C866, with a key that still tells the draft's case
apart.

### The passes, with and without the stop

How each pass ran on cards 0 to 8 (E kernel by kernel, C captured, R replayed) and its GPU time summed over the nine
cards, ms ([compare.md](compare.md), [compare.tsv](compare.tsv); this run's own pass table: [passes.md](passes.md)):

| Pass | R192 (stop) first | repeat | first − repeat | R191 (no stop) first | repeat | first − repeat |
|---|---:|---:|---:|---:|---:|---:|
| 0: the prompt's last 4 tokens (EEEEEEEEE) | 235.65 | 118.65 | +117.00 | 102.29 | 100.57 | +1.71 |
| 1: the first verify pass (EEEEEEEEE) | 219.92 | 118.12 | +101.80 | 100.67 | 102.94 | −2.27 |
| 2: the second verify pass (CCCCCCCCC) | 102.09 | 103.57 | −1.48 | 98.78 | 94.07 | +4.71 |
| the reply's last pass (EEEEEEEEE) | 188.19 | not written | | 70.26 | not written | |
| replayed passes, median a pass (RRRRRRRRR) | 94.17 | 94.27 | | 94.12 | 94.12 | |

- **Every pass ran as in R192:** the same three passes kernel by kernel on all nine cards in both requests, the second
  verify pass captured, every other pass replayed; the same 77 steps, the same 229 drafted and 178 accepted tokens and
  the same reply as R192.
- **Without the stop the first request's kernel-by-kernel passes cost what the repeat's cost,** card by card too: the
  first verify pass 12.51, 12.46, 11.59, 12.70, 9.53, 11.23, 10.63, 11.05 and 8.96 ms on cards 0 to 8, against
  26.15, 28.26, 25.89, 27.25, 20.66, 27.08, 23.51, 24.02 and 17.11 with the stop ([compare.md](compare.md)). Against
  R192's first request: the prompt's tail −133.36 ms, the first verify pass −119.25, the last pass −117.93 (370.54 ms of
  GPU time over the nine cards).
- **A kernel-by-kernel pass without the stop is about as fast as a captured one:** the first verify pass 100.67 ms,
  the captured second 98.78, a replay 94.12.
- **The repeat's kernel-by-kernel passes are also faster here** (100.57 and 102.94 against R192's 118.65 and 118.12),
  although R192's repeat had no stops either. One load each; not traced.

### The stops and the waits

The keep-alive's counts per request, from the line the server prints after each reply ([tasks.md](tasks.md)):

| Request | R192 (=1): kernel by kernel | at new shapes | block stops | R191 (=0): kernel by kernel | acted on | block stops | blocks ended by their 1 s timeout (R192; R191) |
|---|---:|---:|---:|---:|---:|---:|---|
| the load's warm-up request, depth 0 | 72 | 64 | 2 (and 54 outside a graph) | 72 | 0 | 0 | 0; 36 |
| the prime, 4,032 tokens | 9 | 9 | 9 (and 28 outside a graph) | 9 | 0 | 0 | 46; 62 |
| the first request | 31 | 30 | 30 | 31 | 0 | 0 | 4; 15 |
| the repeat | 28 | 0 | 0 | 27 | 0 | 0 | 9; 9 |

At 0 the switch counts the evaluations that run kernel by kernel and does nothing else: it does not track new shapes,
and it stops no block, before an evaluation or before work outside a graph. The server logs
"GGML_CUDA_KEEPALIVE_EAGER=0: counts only" at its start.

**One launch waited about 1 s in the first request, and others in the warm-up request.** Each evaluation of this run
paired with the same one in R192 (same piece of the profile, card and order; [compare.md](compare.md)):

| Where | Card | R192 ms | R191 ms | Increase |
|---|---:|---:|---:|---:|
| first request: the draft's first evaluation, after the prompt's tail (kernel by kernel, its graph at the new depth) | 8 | 6.64 | 1,003.87 | +997.23 |
| warm-up request (depth 0): card 8's third evaluation, kernel by kernel | 8 | 10.34 | 1,009.84 | +999.50 |
| warm-up request: the second evaluation, kernel by kernel | 5 | 31.80 | 247.80 | +216.00 |
| warm-up request: the same | 6 | 31.59 | 236.99 | +205.39 |
| warm-up request: the same | 7 | 30.99 | 266.72 | +235.73 |

- **No other evaluation grew by more than 200 ms;** in the prime and the repeat none grew by more than 26.70 and 0.60.
- **The first request's wait is the draft's.** In the step trace it is step 0's `d.sync` (the host's wait for the
  draft's first pass): +997.11 ms ([first-k1.md](first-k1.md)). It ended when the block's 1,000 ms timeout ran out
  (FN85's mechanism: under lazy module loading a kernel's first launch on a card waits until the card's running kernels
  end). With the stop (R192) this evaluation was one of the 30 at a new shape, and took 6.64 ms. Est. the kernel is
  the draft's first decode attention on the sparse path: at depth 0 the cache is smaller than the sparse attention's
  kept cells, so the draft's earlier passes in this server ran the dense kernels
  (`GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO`, `ggml/src/ggml-cuda/fattn-sparse-pascal.cu` line 34). The main model's
  passes at 4,096 waited for nothing: est. their kernels were first launched while no block ran (the prime's last
  pass after its captures, the prompt's tail after the 4 s pause).
- **The warm-up request at depth 0** (the server's first request of the load) took 1,650.60 ms longer to read its
  64-token prompt and 1,657.83 ms longer to generate its 32 tokens: the waits above, and others outside the profiled
  evaluations (36 blocks ended by their timeout, none in R192, where the switch also stopped the block before 689 new
  sizes outside a graph).

### Generation and prompt times

| | R192 (=1) first | repeat | first − repeat | R191 (=0) first | repeat | first − repeat |
|---|---:|---:|---:|---:|---:|---:|
| generation, ms (256 tokens) | 8,947.48 | 8,656.51 | +290.97 | 9,685.70 | 8,632.35 | +1,053.35 |
| step 0 (the draft's first pass and the first verify pass), ms | 255.10 | 145.39 | +109.72 | 1,132.03 | 130.27 | +1,001.76 |
| the last step, ms | 210.52 | 92.01 | +118.51 | 92.58 | 91.74 | +0.84 |
| prompt, ms | 1,616.85 | 149.31 | | 1,730.72 | 132.13 | |

- **Without the one wait, the first request is est. 51.66 ms slower than its repeat** (1,053.42 over the 77 paired
  steps less step 0's 1,001.76; steps 3 on +45.48 in all, median 0.07 a step, of which steps 8 and 9, where the
  draft's graph changes as in R192, take +12.5 and +11.8): against R192's +290.97.
- **The first request's prompt took 1,730.72 ms,** 113.87 more than R192's, though its 4-token tail took 133.36 ms less
  GPU time. The part before the tail took longer (the server's "ubatch taper" line 1,670.7 ms after the request's
  start, against 1,405.3 in R192); not traced. It is longer than all 13 first requests at 4,096 in R121's loads with the
  stop (the longest 1,701.58 ms, load `w1h2`).

### What it points to: C866, with a key that keeps the draft's stop

- **C866 (Q1562)** keeps `GGML_CUDA_KEEPALIVE_EAGER=1` and changes what counts as a new shape, so that a kernel-by-kernel
  pass at a new depth keeps its block. The key is `ggml_cuda_keepalive_shapes` in `ggml/src/ggml-cuda/ggml-cuda.cu`
  (v4.0, `eb82916d3`), lines 7686 to 7725: it mixes every node's sizes and strides (lines 7699 to 7704) and every
  source's (lines 7716 to 7721), so the cache's length (the K and V views, the mask, every node sized by the cache)
  makes each new depth new on every card. The change: leave the cache's length out and mix instead what it chooses,
  such as whether the sparse attention takes the call. The set the key is checked against,
  `ggml/src/ggml-cuda/keepalive-policy.h` line 331 (`d.shapes.insert(shape)` in `eager_begin`), stays as it is; so does
  the stop before work outside a graph at a new size.
- **What R191 says about C866:** the main passes would run as here (est. the first request's generation about 52 ms
  slower than its repeat instead of 291, and the prompt's tail about 2 ms instead of 117), and the draft's evaluation
  at the new depth must still find a new key, or it waits about 1 s, once per server and newly used kernel. Its check:
  a pair at 4,096 and another at 8,192 on C866's build, with this run's tables; no evaluation may grow by more than
  200 ms against R192.
- **C867 (Q1561)**, capturing these passes, would avoid both the stop's cost and the wait (a capture already stops the
  block first, and a capture at the new size is not slow: 98.78 ms here, 102.09 in R192), but a kernel-by-kernel pass
  without the stop already costs about what a capture costs (100.67 against 98.78), so est. it adds little over C866 for
  much more code.
- **New, C868 (Q1567):** the server's start runs one evaluation of every graph width at a depth past the sparse
  attention's threshold (main model and draft), so that every kernel an agent turn can use is loaded before the first
  turn; then C866's key could leave the cache's length out entirely.

## What ran

- **The engine:** Miner Llama v4.0 (`pascal-p106-v4` at the tag `miner-llama-v4.0`, `eb82916d3`), the release tarball's
  `llama-server` (md5 `e7afd26db01f4c671549ef12c504ada3`, checked on the rig before the run): R192's.
- **The line:** R192's ([run/r191.sh](run/r191.sh)): the deep lane at one depth, 4,096 tokens: the prime (4,032 tokens
  on an empty cache), then the first request (the whole 4,096-token prompt: it reads the last 64 tokens on the live
  cache) and its repeat (it restores a checkpoint four tokens before the end and reads 4), each 256 tokens at
  temperature 0 with the MTP draft, four idle seconds before every request.
- **Loads:** `warm` (the preset, depth 0, untimed, R192's), then `k1`, the one timed load, with R192's logging
  (`GGML_CUDA_GRAPH_DEBUG=1` with `-lv 5`, the graph-mode profile `GGML_CUDA_PROFILE` with
  `GGML_CUDA_PROFILE_MODE=graph`, R121's host profile and step trace) and `GGML_CUDA_KEEPALIVE_EAGER=0` after the
  preset's `=1` (env keeps the last value; the server's own line confirms 0). The started server lines differ from
  R192's in that one word ([started_lines.diff](started_lines.diff), with the run folder's name made equal); the warm
  loads' lines are identical.
- **When:** 2026-11-17, `warm` 13:57:39 to 14:02:08, `k1` 14:02:08 to 14:07:10 (the server up after 225 s each).
  Nothing else ran on the rig, and the only contact during the loads was the lane's own `wait` call (a test for its
  end marker every 20 s). Every card held 0 MiB before and after each load; the host's MemAvailable stayed at 375 MiB
  or more, swap 18 to 33 MiB, no stop by the RAM watchdog ([loads.tsv](loads.tsv), [requests.csv](requests.csv)).

## How it was read

- R192's scripts, unchanged: [graph_passes.py](../2026-11-17_r192_first_request_graphs/run/graph_passes.py) (passes
  from the graph-mode profile) and [server_graph_log.py](../2026-11-17_r192_first_request_graphs/run/server_graph_log.py)
  (debug lines and keep-alive counts per request); R121's
  [first_request.py](../2026-11-17_r121_flash_step_map/run/first_request.py) with R86's
  [align_t0.py](../2026-11-11_r86_step_trace/run/align_t0.py) (steps and the host profile).
- [run/compare.py](run/compare.py), new: this run against R192's published tables and profile: the passes not replayed
  on every card, the first verify pass and the prompt's tail card by card, the keep-alive's counts, the request times,
  and each evaluation paired in order with R192's (piece, card, order), with every pair that grew by more than 200 ms.
  A launch that waits for a running block is inside its evaluation's event pair, so it shows there.
- [run/tables.sh](run/tables.sh) runs them all. The tables ran on the operator's machine after the run, with
  `python3 -I`.

## Commands

From the repository's root, on the machine that drives the rig (it runs only ssh and tar):

```bash
RIG=<rig alias> WORK=<run folder on the rig> BIN=<v4.0 llama-server on the rig> \
  MODEL=<Swift Q2_0 shard 1 on the rig> MODEL_DRAFT=<the MTP draft on the rig> \
  TEXT_LOCAL=<the release check's proxy.jsonl.bodies.gz> \
  bash docs/data/2026-11-17_r191_keepalive_stop/run/r191.sh            # STEP=start; DRY=1 prints the lines
STEP=wait RIG=<rig alias> WORK=<run folder> bash docs/data/2026-11-17_r191_keepalive_stop/run/r191.sh
STEP=collect OUT=<folder here> RIG=<rig alias> WORK=<run folder> \
  bash docs/data/2026-11-17_r191_keepalive_stop/run/r191.sh
```

The tables (on any machine with Python 3):

```bash
S=<this repository> OUT=<the collected results> RES=<tables folder> \
  bash docs/data/2026-11-17_r191_keepalive_stop/run/tables.sh
```

## Files

| File | What |
|---|---|
| [compare.md](compare.md), [compare.tsv](compare.tsv) | this run against R192: passes, card by card, keep-alive counts, request times, each evaluation paired with R192's ([run/compare.py](run/compare.py)) |
| [passes.md](passes.md), [passes.tsv](passes.tsv) | every main pass of both requests: how it ran on each card, its GPU time; the draft's kernel-by-kernel and captured evaluations |
| [tasks.md](tasks.md), [tasks.tsv](tasks.tsv) | per request: the backend's debug lines by kind, llama.cpp's graph reuses, the keep-alive's counts |
| [first-k1.md](first-k1.md), [steps-k1.tsv](steps-k1.tsv), [pairs-k1.tsv](pairs-k1.tsv), [align-k1.md](align-k1.md) | R121's comparison of the two requests step by step, from the step trace and the host profile |
| [gp-k1.csv.gz](gp-k1.csv.gz) | the graph-mode profile as written: one line per evaluation and card (`how`, `graph`, `eval_us`) |
| [requests.csv](requests.csv) | every request of both loads: prompt and generation times, drafted and accepted tokens, the reply's md5 |
| [loads.tsv](loads.tsv) | each load's environment and server line as it ran (paths as `<OUT>`, `<BIN>`) |
| [started_lines.diff](started_lines.diff) | R192's timed server line against this run's, one word a line |
| [run/](run/) | `r191.sh` (the run), `tables.sh`, `compare.py` |

## Provenance and licence

Code: Miner Llama (MIT, like llama.cpp), v4.0. Model: Swift-1.5 GSQ-RCO Qwen3.8-Flash-Next Q2_0
([ukisai](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF), Swift Open License 1.0 and
qwen-community-1.0) with its MTP draft file; the prompts are the deep lane's recorded agent text
([bench/deep_lane](../../../bench/deep_lane/README.md)), R192's and R121's. The tables hold times, counts, graph
decisions and md5s, no text. Measured on a Manli M-P106L9-N6G (nine P106-090).
