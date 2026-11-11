## The deep lane: eager, graphs, eagerb

Model file `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf`, llama-server version: 0.9.0-dev (build 12581, commit c408dabcb3).

16 tokens per request at temperature 0, each request twice per load (pass 1 is the first request at a new depth, pass 2 the same request again). Timed loads, in their order: eager1, graphs1, eager2.

| Arm | Loads | Draft | Context | Its own variables and flags |
|---|---|---|---:|---|
| eager (control) | eager1 | on | 65536 | `GGML_CUDA_FATTN_SPARSE_PASCAL=1 GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=1.0 GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=5 LLAMA_SPEC_DUMP=$WORK/lane/out/dump-eager.jsonl`, `--spec-draft-sampling probabilistic --spec-draft-p-min 0 --ctx-checkpoints-vram 4 --ubatch-taper on --spec-draft-n-ubatch 512` |
| graphs | graphs1 | on | 65536 | `GGML_CUDA_FATTN_SPARSE_PASCAL=1 GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=1.0 GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=5 GGML_CUDA_GRAPHS_FORCE=1 LLAMA_SPEC_DUMP=$WORK/lane/out/dump-graphs.jsonl`, `--spec-draft-sampling probabilistic --spec-draft-p-min 0 --ctx-checkpoints-vram 4 --ubatch-taper on --spec-draft-n-ubatch 512` |
| eagerb | eager2 | on | 65536 | `GGML_CUDA_FATTN_SPARSE_PASCAL=1 GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=1.0 GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=5 LLAMA_SPEC_DUMP=$WORK/lane/out/dump-eagerb.jsonl`, `--spec-draft-sampling probabilistic --spec-draft-p-min 0 --ctx-checkpoints-vram 4 --ubatch-taper on --spec-draft-n-ubatch 512` |

### Generation, tok/s

Per load: pass 1, pass 2 → the load, with the load's time per token. A load's speed is its two passes' tokens over their time. Spread: the difference between the arm's fastest and slowest load. Class against `eager`: gain when the mean is above the control's by more than the larger of the two spreads, loss when below by more than it, else negligible.

| Depth (tokens) | Arm | Per load | Spread | Against `eager` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | eager | eager1: 6.68, 13.84 → **9.01** (111.00 ms a token) | — | control | | |
|  | graphs | graphs1: 6.70, 13.83 → **9.03** (110.78 ms a token) | — | +0.02 tok/s | +0.2 % | no class (one load) |
|  | eagerb | eager2: 6.70, 13.84 → **9.03** (110.77 ms a token) | — | +0.02 tok/s | +0.2 % | no class (one load) |
| 32,768 | eager | eager1: 6.46, 11.05 → **8.15** (122.68 ms a token) | — | control | | |
|  | graphs | graphs1: 6.18, 11.23 → **7.98** (125.39 ms a token) | — | -0.18 tok/s | -2.2 % | no class (one load) |
|  | eagerb | eager2: 6.47, 11.25 → **8.21** (121.79 ms a token) | — | +0.06 tok/s | +0.7 % | no class (one load) |

### Generation over all depths, tok/s

Every depth's tokens over their time, per load.

| Depth (tokens) | Arm | Per load | Spread | Against `eager` | | Class |
|---:|---|---|---:|---:|---:|---|
| all | eager | eager1: **8.56** | — | control | | |
|  | graphs | graphs1: **8.47** | — | -0.09 tok/s | -1.1 % | no class (one load) |
|  | eagerb | eager2: **8.60** | — | +0.04 tok/s | +0.5 % | no class (one load) |

### The step on pass 1 (the first request at a new depth), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `eager`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `eager` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | eager | eager1: 266.21 (9 steps) | — | control | | |
|  | graphs | graphs1: 265.33 (9 steps) | — | -0.88 ms | -0.3 % | no class (one load) |
|  | eagerb | eager2: 265.41 (9 steps) | — | -0.79 ms | -0.3 % | no class (one load) |
| 32,768 | eager | eager1: 309.68 (8 steps) | — | control | | |
|  | graphs | graphs1: 323.39 (8 steps) | — | +13.72 ms | +4.4 % | no class (one load) |
|  | eagerb | eager2: 309.35 (8 steps) | — | -0.32 ms | -0.1 % | no class (one load) |

### The step on pass 2 (the same request again), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `eager`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `eager` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | eager | eager1: 128.47 (9 steps) | — | control | | |
|  | graphs | graphs1: 128.56 (9 steps) | — | +0.08 ms | +0.1 % | no class (one load) |
|  | eagerb | eager2: 128.45 (9 steps) | — | -0.03 ms | -0.0 % | no class (one load) |
| 32,768 | eager | eager1: 160.93 (9 steps) | — | control | | |
|  | graphs | graphs1: 158.35 (9 steps) | — | -2.58 ms | -1.6 % | no class (one load) |
|  | eagerb | eager2: 158.05 (9 steps) | — | -2.88 ms | -1.8 % | no class (one load) |

### Prompt reading, tok/s

Per load: the tokens read at that depth over their time (a prime request reads a chunk on top of the cache; without primes each decode request reads its prompt), with the tokens read and the tokens reused. Requests that read under 32 tokens are left out.

| Depth (tokens) | Arm | Per load | Spread | Against `eager` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | eager | eager1: 48.1 (64 read on 0 cached) | — | control | | |
|  | graphs | graphs1: 48.3 (64 read on 0 cached) | — | +0.2 tok/s | +0.4 % | no class (one load) |
|  | eagerb | eager2: 49.4 (64 read on 0 cached) | — | +1.2 tok/s | +2.6 % | no class (one load) |
| 32,768 | eager | eager1: 113.7 (32704 read on 0 cached) | — | control | | |
|  | graphs | graphs1: 145.2 (32704 read on 0 cached) | — | +31.5 tok/s | +27.7 % | no class (one load) |
|  | eagerb | eager2: 113.8 (32704 read on 0 cached) | — | +0.1 tok/s | +0.1 % | no class (one load) |

### The draft and the step

Per load, pass 1: draft tokens accepted of drafted (the share), tokens a step (generated tokens over steps, a step being one pass of the main model: generated minus accepted), and the time of a step. Pass 2 is listed where its counters differ.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | eager | eager1: 7 of 23 (0.304), 1.78 a step, 266.2 ms a step |
|  | graphs | graphs1: 7 of 23 (0.304), 1.78 a step, 265.3 ms a step |
|  | eagerb | eager2: 7 of 23 (0.304), 1.78 a step, 265.4 ms a step |
| 32,768 | eager | eager1: 8 of 18 (0.444), 2.00 a step, 309.7 ms a step (pass 2: 7 of 18 (0.389), 1.78 a step, 160.9 ms a step) |
|  | graphs | graphs1: 8 of 18 (0.444), 2.00 a step, 323.4 ms a step (pass 2: 7 of 18 (0.389), 1.78 a step, 158.4 ms a step) |
|  | eagerb | eager2: 8 of 18 (0.444), 2.00 a step, 309.4 ms a step (pass 2: 7 of 18 (0.389), 1.78 a step, 158.1 ms a step) |

### Checks

- Times: every load started on a machine checked quiet (valid=1).
- Prompts: 64 ids at depth 0, 32768 ids at depth 32,768; the same ids on every load and pass.
- Tokens out: 16 on every timed request.
- Greedy reply at depth 0: the same on every pass and load of every arm.
- Greedy reply at depth 32,768: the same on every pass and load of every arm.
- Primes (read on cached, the cache's source): 32,768: 32704 on 0 (none).
