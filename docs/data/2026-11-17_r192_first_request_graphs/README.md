# Which passes of a first request run without a CUDA graph (R192; Q1561; 2026-11-17)

What this measured: on Miner Llama v4.0 with Flash-Next's released preset, a first request at 4,096 tokens of context
generates about 310 ms slower than the same request repeated, and R121 found that time in two passes, the first verify
pass after the prompt and the reply's last pass
([2026-11-17_r121_flash_step_map](../2026-11-17_r121_flash_step_map/README.md), "The first request at a depth").
This run logs, for every pass of both requests on all nine cards, whether the CUDA backend ran its graph kernel by
kernel, captured it or replayed it, and names the code that decides it. Nine P106-090 cards, UkisAI's Swift-1.5
GSQ-RCO Q2_0 file with its MTP draft, [release/flash-next-v4/settings.env](../../../release/flash-next-v4/settings.env)
unchanged, plus logging only. Registry row Q1561 in [quants/experiments.md](../../../quants/experiments.md).

One load; every value below is from it unless a row says otherwise.

## Result

**Both requests run the same passes without a CUDA graph, so a missing graph is not what makes the first request
slow.** In both, the prompt's last 4 tokens, the first verify pass and the reply's last pass run kernel by kernel on
all nine cards; the second verify pass is captured and every later pass replayed. In the repeat those kernel-by-kernel
passes cost about 24 ms more than a replay over the nine cards; in the first request they take about twice as long.
The difference is that in the first request their shapes are new to every card (a new cache size), and before each
such evaluation the keep-alive switch `GGML_CUDA_KEEPALIVE_EAGER=1` (C811) ended the card's clock-holding block: 30
times in the first request, never in the repeat.

### Each pass, the first request against its repeat

How each main pass ran on cards 0 to 8 (E kernel by kernel, C captured, R replayed) and its GPU time summed over the
nine cards, ms ([passes.md](passes.md), [passes.tsv](passes.tsv)):

| Pass | First request | GPU ms | Repeat | GPU ms | Difference |
|---|---|---:|---|---:|---:|
| 0: the prompt's last 4 tokens | EEEEEEEEE | 235.65 | EEEEEEEEE | 118.65 | +117.00 |
| 1: the first verify pass after the prompt | EEEEEEEEE | 219.92 | EEEEEEEEE | 118.12 | +101.80 |
| 2: the second verify pass | CCCCCCCCC | 102.09 | CCCCCCCCC | 103.57 | −1.48 |
| 3 to the second-to-last | RRRRRRRRR in every pass | | RRRRRRRRR in every pass | | |
| the reply's last pass (one token, no draft) | EEEEEEEEE | 188.19 | EEEEEEEEE (counted, not timed) | | |

- **Pass 0 reads the prompt's last 4 tokens in both requests.** The first request reads its 64 new tokens as 60 and
  then 4, since `--ctx-checkpoints-incall` takes the checkpoint four tokens before the end inside the call; the
  repeat restores that checkpoint and reads the same 4. The profiler never writes the 60-token micro-batch (its
  graph, the prompt micro-batches', is not evaluated again in the load); est. it was captured, as all eight of the
  prime's micro-batches were (`GGML_CUDA_GRAPHS_FORCE`).
  Pass 0 is prompt time, so R121, which compared generation time, did not see it: **the first request also spends
  +117.00 ms more GPU time before its first token.**
- **The repeat's last pass ran kernel by kernel too**, by count: the profiler writes an evaluation when its graph runs
  again, so the load's very last pass has no line, but the keep-alive counts 28 kernel-by-kernel evaluations in the
  repeat: the 18 written (passes 0 and 1), the last pass's nine and the draft's last one.
- **Every pass from the third verify pass to the second-to-last is a replay on all nine cards in both requests**, and
  the steps match: the median step from step 3 on is 112.61 ms in the first request and 112.07 in the repeat
  ([first-g1.md](first-g1.md)).

The two verify passes after the prompt and the last pass, card by card (GPU ms, how in parentheses;
[passes.md](passes.md)):

| Pass | Request | Card 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|---|
| first verify | first | 26.15 (E) | 28.26 (E) | 25.89 (E) | 27.25 (E) | 20.66 (E) | 27.08 (E) | 23.51 (E) | 24.02 (E) | 17.11 (E) |
| first verify | repeat | 12.41 (E) | 16.02 (E) | 13.15 (E) | 16.28 (E) | 11.74 (E) | 13.46 (E) | 12.69 (E) | 13.31 (E) | 9.05 (E) |
| second verify | first | 16.86 (C) | 11.90 (C) | 11.24 (C) | 12.17 (C) | 9.31 (C) | 10.83 (C) | 10.40 (C) | 10.69 (C) | 8.68 (C) |
| second verify | repeat | 12.52 (C) | 13.70 (C) | 11.32 (C) | 13.98 (C) | 9.98 (C) | 11.56 (C) | 10.95 (C) | 11.44 (C) | 8.12 (C) |
| last | first | 21.67 (E) | 24.82 (E) | 21.45 (E) | 22.70 (E) | 17.63 (E) | 23.71 (E) | 20.88 (E) | 21.88 (E) | 13.43 (E) |

The first request's first verify pass is the one R121 measured (219.9 ms on the nine cards). A pass captured at the
new cache size is not slow: the first request's second verify pass takes 102.09 ms against the repeat's 103.57.

### What differs: new shapes, and the keep-alive's stops

Per request, from the server log ([tasks.md](tasks.md); the keep-alive's counts are printed after each reply, here
as the difference to the reply before):

| Request | Evaluations kernel by kernel | of them at shapes new to their card | a running block stopped before them | blocks ended by their timeout | Debug: graph reused | warm-up complete (captured) | warm-up reset | executable update failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| first | 31 | 30 | 30 | 4 | 901 | 12 | 3 | 15 |
| repeat | 28 | 0 | 0 | 9 | 902 | 9 | 9 | 0 |

- **The kernel-by-kernel evaluations are the same in number:** 27 main ones in each (three passes on nine cards) and
  the draft's 4 and 1. In the first request all but one are at a shape new to their card, and before each the switch
  stopped the card's block and held its next start until that evaluation's work was done. In the repeat the shapes
  were known, so the blocks kept running through the same passes.
- **R121's loads show the same counts at 4,096 tokens** (its server logs, every width-4 load: `w4u1`, `w4h1`, `w4h2`,
  `w4g1`, `w4g2`): 31 kernel by kernel, 30 at new shapes and 30 stops in the first request; 28 (27 in `w4h1`), 0 and
  0 in the repeat.
- **The debug lines name the decisions.** In the first request the main graph's nine "warm-up complete" lines come at
  its second verify pass (the first unchanged call), and its kernel-by-kernel passes log nothing: its warm-up had not
  completed since the prime. In the repeat the 4-token prompt pass resets the completed warm-up on all nine cards
  ("warm-up reset" 9), and the same pattern follows. The last pass's one-token graph is a graph of its own that runs
  once a reply and never completes a warm-up. The first request's other 3 resets and 3 of its 12 captures are the
  draft's; its 15 failed executable updates (each then instantiated anew) are 13 while it read its prompt and 2 at the
  draft's captures, none in the main passes of the generation.
- **The draft's graph changes twice more in the first request** (a kernel-by-kernel evaluation, then a capture, after
  main passes 8 and 10): steps 8, 9 and 10 take 128.26, 124.51 and 114.33 ms against the repeat's 112.10, 112.64 and
  110.02.

### Why (the code, v4.0 at `eb82916d3`)

- **Kernel by kernel, captured or replayed** is decided in `ggml_backend_cuda_graph_compute`
  (`ggml/src/ggml-cuda/ggml-cuda.cu`):
  - line 7780: `GGML_CUDA_GRAPHS_FORCE` captures at every change only the graphs that `ggml_cuda_graph_force_eligible`
    takes (`ggml/src/ggml-cuda/graph-force.h` line 25: more than 8 tokens, i.e. prompt micro-batches and the draft's
    catch-up). Decode and verify graphs keep ggml's warm-up, because their lazily made state (the persistent MoE
    kernel's mapped host words, the attention kernels' counters) must be made outside a capture (`graph-force.h`
    lines 20 to 22);
  - lines 7790 to 7802, the warm-up: an evaluation whose properties changed runs kernel by kernel, and the next
    unchanged one is captured;
  - lines 7803 to 7815, after a completed warm-up: a change resets it, and that evaluation runs kernel by kernel;
  - line 7871 names the outcome ("eager", "capture", "replay") for the profiler.

  A new cache size changes a decode or verify graph's properties (its cache views), so at every new depth the first
  pass of each width runs kernel by kernel, in a first request and in its repeat alike.
- **Why only the first request's are slow** (exact for the counts, est. for the mechanism): before a kernel-by-kernel
  evaluation, lines 7839 to 7866 hash its shapes (`ggml_cuda_keepalive_shapes`, line 7686: every node's op, type,
  sizes and strides) and call `ggml_cuda_keepalive_eager_begin`. At a shape new to the card
  (`ggml/src/ggml-cuda/keepalive-policy.h` line 331, in `eager_begin`) the card's block is ended and its restart held
  until the evaluation's work is done (C811: so that a kernel's first launch under lazy module loading does not wait
  about 1 s for the block). The cache size is in the hash, so every new depth is a new shape on every card, though
  the kernels are the same. Those passes then run with no block on their card; a P106 lowers its clock when it idles
  between kernels, which the block exists to prevent (Q1021), and kernels launched one by one leave it such gaps. That
  fits a pass about twice as long on every card with every launch group slower (R121's group profile). This run cannot
  tell the block's absence from other first-time costs of a new shape; R191 (Q1562) can, with the switch's counting
  mode.

### What C867 would change

- **C867 (Q1561)** takes decode and verify graphs into the forced path (or C779's patch), with their lazily made state
  made outside the capture: the first evaluation after a change is captured, so these passes never run kernel by
  kernel and never meet the keep-alive's stop. Here a capture at the new cache size cost what it cost in the repeat
  (102.09 against 103.57 ms). Est. GPU time saved per reply at 4,096 tokens: about 118 ms at the first verify pass
  (219.92 to about 102), about 138 at the last pass (188.19 to est. about 50, a one-token pass) and about 134 at the
  prompt's tail (235.65 to about 102), and about 16 ms at each of the repeat's two kernel-by-kernel passes. Not
  estimated: the host's cost of capturing the one-token graph once a reply.
- **A smaller change, new (C866, Q1562):** the keep-alive's shape key without the cache size, so that a
  kernel-by-kernel pass at a new depth keeps its card's block, as the repeat's do. If the block is the cost, those
  passes would cost what the repeat's cost (about 24 ms over a replay on the nine cards), with no capture code; the
  risk is a kernel that a new size picks for the first time (FN85's wait of about 1 s). Its check is R191.

### What it costs an agent

- **Generation:** in this load (all the logging on) the first request generates 290.97 ms slower than its repeat
  (8,947.48 against 8,656.51 ms; the same 229 drafted and 178 accepted tokens, 77 steps, the same reply): the step of
  the first verify pass +109.72, the second −0.59, the third −4.67, the last step +118.51 and the 73 steps between
  +68.01 (of which steps 8 to 10, the draft's changes, +16.16, +11.86 and +4.31). R121 measured 310.85 ms with no
  profiler, and 306.31 and 327.19 with the host profile.
- **Before the first token:** the 4-token prompt tail's +117.00 ms of GPU time on the nine cards.
- Every agent turn is a first request at a new depth, so est. both are paid once a turn.

## What ran

- **The engine:** Miner Llama v4.0 (`pascal-p106-v4` at the tag `miner-llama-v4.0`, `eb82916d3`), the release
  tarball's `llama-server` (md5 `e7afd26db01f4c671549ef12c504ada3`, checked on the rig before the run).
- **The line:** R121's deep lane ([run/r192.sh](run/r192.sh)), the v4 preset exactly, at one depth, 4,096 tokens:
  the prime (4,032 tokens on an empty cache), then the first request (the whole 4,096-token prompt: it reads the last
  64 tokens on the live cache) and its repeat (it restores a checkpoint four tokens before the end and reads 4), each
  256 tokens at temperature 0 with the MTP draft, four idle seconds before every request. No sampled request.
- **Loads:** `warm` (the preset, depth 0, untimed, as in R121), then `g1`, the one timed load, with only these added:
  - `GGML_CUDA_GRAPH_DEBUG=1`: the backend's graph decisions, logged at debug level, so the server line also gets
    `-lv 5` (at the default level 3 the server drops them);
  - `GGML_CUDA_PROFILE` with `GGML_CUDA_PROFILE_MODE=graph`: one line per evaluation and card with how it ran
    (`eager`, `capture`, `replay`), its graph's number and its GPU time, from an event pair around the whole
    evaluation; nothing synchronized. This is the per-evaluation record: the debug lines alone cannot tell an
    evaluation run kernel by kernel during a warm-up from a replay, since neither logs a line;
  - `GGML_CUDA_PROFILE_HOST` and `LLAMA_STEP_TRACE`: R121's host profile and step trace, for R121's own comparison of
    the two requests.
- **When:** 2026-11-17, `warm` 13:30:10 to 13:34:39, `g1` 13:34:39 to 13:39:37 (the server up after 225 s each);
  nothing else ran on the rig, and the only contact during the loads was the lane's own `wait` call (a test for its
  end marker every 20 s). Every card held 0 MiB before and
  after each load; the host's MemAvailable stayed at 369 MiB or more and 18 MiB of swap was in use, with no stop by
  the RAM watchdog ([loads.tsv](loads.tsv), [requests.csv](requests.csv)).
- **The logging's own cost:** this load's prime read 3,972 tokens in 29,278.01 ms (R121's unprofiled load: 4,032 in
  20,826.16 ms), and the first request generated 290.97 ms slower than its repeat against R121's 310.85 with no
  profiler; the replies are identical between the two requests.

## How it was read

- [run/graph_passes.py](run/graph_passes.py) cuts the graph-mode profile per request at the pauses and lines up the
  main passes: the j-th evaluation of cards 0 to 7 and the next one of card 8 (the MTP draft runs on card 8 only;
  card 8's other evaluations are the draft's). Pass 0 reads the prompt's tail, pass 1 is the first verify pass after
  it, the request's last pass verifies the reply's last tokens.
- [run/server_graph_log.py](run/server_graph_log.py) counts the debug lines per request (a slot task in the server
  log) and the keep-alive's counts the server prints after every reply while `GGML_CUDA_KEEPALIVE_EAGER` is set
  (evaluations run kernel by kernel, those at shapes new to their card, and the card's clock-holding block stopped
  before them).
- [run/tables.sh](run/tables.sh) runs both, and R121's
  [first_request.py](../2026-11-17_r121_flash_step_map/run/first_request.py) on the step trace and host profile.
- The tables ran on the rig's CPU after the run (`nice -n 19`), as in R121.

## Commands

From the repository's root, on the machine that drives the rig (it runs only ssh and tar):

```bash
RIG=<rig alias> WORK=<run folder on the rig> BIN=<v4.0 llama-server on the rig> \
  MODEL=<Swift Q2_0 shard 1 on the rig> MODEL_DRAFT=<the MTP draft on the rig> \
  TEXT_LOCAL=<the release check's proxy.jsonl.bodies.gz> \
  bash docs/data/2026-11-17_r192_first_request_graphs/run/r192.sh            # STEP=start; DRY=1 prints the lines
STEP=wait RIG=<rig alias> WORK=<run folder> bash docs/data/2026-11-17_r192_first_request_graphs/run/r192.sh
STEP=collect OUT=<folder here> RIG=<rig alias> WORK=<run folder> \
  bash docs/data/2026-11-17_r192_first_request_graphs/run/r192.sh
```

The tables (on any machine with Python 3; here the rig after the run):

```bash
S=<this repository> OUT=<the collected results> RES=<tables folder> \
  bash docs/data/2026-11-17_r192_first_request_graphs/run/tables.sh
```

## Files

| File | What |
|---|---|
| [passes.md](passes.md), [passes.tsv](passes.tsv) | every main pass of both requests: how it ran on each card, its GPU time; the draft's kernel-by-kernel and captured evaluations ([run/graph_passes.py](run/graph_passes.py)) |
| [tasks.md](tasks.md), [tasks.tsv](tasks.tsv) | per request: the backend's debug lines by kind, llama.cpp's graph reuses, the keep-alive's counts ([run/server_graph_log.py](run/server_graph_log.py)) |
| [first-g1.md](first-g1.md), [steps-g1.tsv](steps-g1.tsv), [pairs-g1.tsv](pairs-g1.tsv), [align-g1.md](align-g1.md) | R121's comparison of the two requests step by step, from the step trace and the host profile |
| [gp-g1.csv.gz](gp-g1.csv.gz) | the graph-mode profile as written: one line per evaluation and card (`how`, `graph`, `eval_us`) |
| [requests.csv](requests.csv) | every request of both loads: prompt and generation times, drafted and accepted tokens, the reply's md5 |
| [loads.tsv](loads.tsv) | each load's environment and server line as it ran (paths as `<OUT>`, `<MODELS>`, `<BIN>`) |
| [run/](run/) | `r192.sh` (the run), `tables.sh`, `graph_passes.py`, `server_graph_log.py` |

## Provenance and licence

Code: Miner Llama (MIT, like llama.cpp), v4.0. Model: Swift-1.5 GSQ-RCO Qwen3.8-Flash-Next Q2_0
([ukisai](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF), Swift Open License 1.0 and
qwen-community-1.0) with its MTP draft file; the prompts are the deep lane's recorded agent text
([bench/deep_lane](../../../bench/deep_lane/README.md)), the same as R121's. The tables hold times, counts, graph
decisions and md5s, no text. Measured on a Manli M-P106L9-N6G (nine P106-090).
