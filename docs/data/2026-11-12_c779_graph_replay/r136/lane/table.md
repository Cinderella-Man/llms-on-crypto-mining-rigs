## The deep lane: force, patch

Model file `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf`, llama-server version: 0.9.0-dev (build 12682, commit 771cafa97).

16 tokens per request at temperature 0, each request twice per load (pass 1 is the first request at a new depth, pass 2 the same request again). Timed loads, in their order: force1, patch1.

| Arm | Loads | Draft | Context | Its own variables and flags |
|---|---|---|---:|---|
| force (control) | force1 | on | 65536 | `GGML_CUDA_FATTN_SPARSE_PASCAL=1 GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=1.0 GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=5 GGML_CUDA_GRAPHS_FORCE=1 LLAMA_SPEC_DUMP=$WORK/lane/out/dump-force.jsonl`, `--spec-draft-sampling probabilistic --spec-draft-p-min 0 --ctx-checkpoints-vram 4 --ubatch-taper on --spec-draft-n-ubatch 512` |
| patch | patch1 | on | 65536 | `GGML_CUDA_FATTN_SPARSE_PASCAL=1 GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=1.0 GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=5 GGML_CUDA_GRAPHS_FORCE=1 GGML_CUDA_GRAPHS_PATCH=1 LLAMA_SPEC_DUMP=$WORK/lane/out/dump-patch.jsonl`, `--spec-draft-sampling probabilistic --spec-draft-p-min 0 --ctx-checkpoints-vram 4 --ubatch-taper on --spec-draft-n-ubatch 512` |

### Generation, tok/s

Per load: pass 1, pass 2 → the load, with the load's time per token. A load's speed is its two passes' tokens over their time. Spread: the difference between the arm's fastest and slowest load. Class against `force`: gain when the mean is above the control's by more than the larger of the two spreads, loss when below by more than it, else negligible.

| Depth (tokens) | Arm | Per load | Spread | Against `force` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | force | force1: 6.66, 13.73 → **8.97** (111.47 ms a token) | — | control | | |
|  | patch | patch1: 6.67, 13.75 → **8.98** (111.34 ms a token) | — | +0.01 tok/s | +0.1 % | no class (one load) |
| 32,768 | force | force1: 6.13, 11.19 → **7.92** (126.23 ms a token) | — | control | | |
|  | patch | patch1: 6.44, 11.11 → **8.16** (122.59 ms a token) | — | +0.24 tok/s | +3.0 % | no class (one load) |

### Generation over all depths, tok/s

Every depth's tokens over their time, per load.

| Depth (tokens) | Arm | Per load | Spread | Against `force` | | Class |
|---:|---|---|---:|---:|---:|---|
| all | force | force1: **8.41** | — | control | | |
|  | patch | patch1: **8.55** | — | +0.14 tok/s | +1.6 % | no class (one load) |

### The step on pass 1 (the first request at a new depth), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `force`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `force` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | force | force1: 266.86 (9 steps) | — | control | | |
|  | patch | patch1: 266.55 (9 steps) | — | -0.32 ms | -0.1 % | no class (one load) |
| 32,768 | force | force1: 326.13 (8 steps) | — | control | | |
|  | patch | patch1: 310.34 (8 steps) | — | -15.79 ms | -4.8 % | no class (one load) |

### The step on pass 2 (the same request again), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `force`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `force` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | force | force1: 129.49 (9 steps) | — | control | | |
|  | patch | patch1: 129.33 (9 steps) | — | -0.16 ms | -0.1 % | no class (one load) |
| 32,768 | force | force1: 158.94 (9 steps) | — | control | | |
|  | patch | patch1: 160.02 (9 steps) | — | +1.09 ms | +0.7 % | no class (one load) |

### Prompt reading, tok/s

Per load: the tokens read at that depth over their time (a prime request reads a chunk on top of the cache; without primes each decode request reads its prompt), with the tokens read and the tokens reused. Requests that read under 32 tokens are left out.

| Depth (tokens) | Arm | Per load | Spread | Against `force` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | force | force1: 47.7 (64 read on 0 cached) | — | control | | |
|  | patch | patch1: 49.3 (64 read on 0 cached) | — | +1.7 tok/s | +3.5 % | no class (one load) |
| 32,768 | force | force1: 147.4 (32704 read on 0 cached) | — | control | | |
|  | patch | patch1: 146.1 (32704 read on 0 cached) | — | -1.3 tok/s | -0.9 % | no class (one load) |

### The draft and the step

Per load, pass 1: draft tokens accepted of drafted (the share), tokens a step (generated tokens over steps, a step being one pass of the main model: generated minus accepted), and the time of a step. Pass 2 is listed where its counters differ.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | force | force1: 7 of 23 (0.304), 1.78 a step, 266.9 ms a step |
|  | patch | patch1: 7 of 23 (0.304), 1.78 a step, 266.5 ms a step |
| 32,768 | force | force1: 8 of 18 (0.444), 2.00 a step, 326.1 ms a step (pass 2: 7 of 18 (0.389), 1.78 a step, 158.9 ms a step) |
|  | patch | patch1: 8 of 18 (0.444), 2.00 a step, 310.3 ms a step (pass 2: 7 of 18 (0.389), 1.78 a step, 160.0 ms a step) |

### Checks

- Times: every load started on a machine checked quiet (valid=1).
- Prompts: 64 ids at depth 0, 32768 ids at depth 32,768; the same ids on every load and pass.
- Tokens out: 16 on every timed request.
- Greedy reply at depth 0: the same on every pass and load of every arm.
- Greedy reply at depth 32,768: the same on every pass and load of every arm.
- Primes (read on cached, the cache's source): 32,768: 32704 on 0 (none).
- Graph reallocations of the target, requests with one per load: force1: 0 of 6 requests; patch1: 0 of 6 requests.
