## The deep lane: v3, v4

Model file `flint-near-q8-p065.gguf`, llama-server version: 1.0.0-dev (build 12832, commit eb312b7bce).

256 tokens per request at temperature 0, each request twice per load (pass 1 is the first request at a new depth, pass 2 the same request again). Timed loads, in their order: v31, v41, v32, v42.

| Arm | Loads | Draft | Context | Its own variables and flags |
|---|---|---|---:|---|
| v3 (control) | v31, v32 | off | 262144 | — |
| v4 | v41, v42 | off | 262144 | `LLAMA_KV_SEQ_RM_BOUND=1 LLAMA_SERVER_YIELD=2 LLAMA_SAMPLING_TOP_K_ROWS=1 GGML_CUDA_GRAPHS_FORCE=1 LLAMA_SCHED_PER_SIZE_RESERVE=2`, `--ctx-checkpoints-vram 4 --ubatch-taper short=600:64` |

### Generation, tok/s

Per load: pass 1, pass 2 → the load, with the load's time per token. A load's speed is its two passes' tokens over their time. Spread: the difference between the arm's fastest and slowest load. Class against `v3`: gain when the mean is above the control's by more than the larger of the two spreads, loss when below by more than it, else negligible.

| Depth (tokens) | Arm | Per load | Spread | Against `v3` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | v3 | v31: 47.10, 54.86 → **50.68** (19.73 ms a token); v32: 43.05, 54.90 → **48.26** (20.72 ms a token) | 2.42 | control | | |
|  | v4 | v41: 46.91, 55.28 → **50.75** (19.70 ms a token); v42: 46.22, 55.27 → **50.34** (19.86 ms a token) | 0.41 | +1.08 tok/s | +2.2 % | negligible |
| 4,096 | v3 | v31: 73.97, 79.55 → **76.66** (13.04 ms a token); v32: 74.03, 79.41 → **76.63** (13.05 ms a token) | 0.04 | control | | |
|  | v4 | v41: 75.09, 80.44 → **77.67** (12.87 ms a token); v42: 74.99, 80.43 → **77.61** (12.88 ms a token) | 0.06 | +1.00 tok/s | +1.3 % | gain |
| 8,192 | v3 | v31: 46.28, 50.81 → **48.44** (20.65 ms a token); v32: 46.55, 50.80 → **48.58** (20.58 ms a token) | 0.14 | control | | |
|  | v4 | v41: 47.08, 51.33 → **49.11** (20.36 ms a token); v42: 47.71, 51.28 → **49.43** (20.23 ms a token) | 0.32 | +0.76 tok/s | +1.6 % | gain |
| 16,384 | v3 | v31: 64.17, 69.82 → **66.88** (14.95 ms a token); v32: 64.31, 69.83 → **66.96** (14.94 ms a token) | 0.08 | control | | |
|  | v4 | v41: 65.00, 70.44 → **67.61** (14.79 ms a token); v42: 65.06, 70.37 → **67.61** (14.79 ms a token) | 0.00 | +0.69 tok/s | +1.0 % | gain |
| 32,768 | v3 | v31: 25.79, 26.78 → **26.28** (38.06 ms a token); v32: 25.79, 26.79 → **26.28** (38.05 ms a token) | 0.01 | control | | |
|  | v4 | v41: 26.05, 27.02 → **26.52** (37.70 ms a token); v42: 26.03, 27.00 → **26.51** (37.73 ms a token) | 0.02 | +0.24 tok/s | +0.9 % | gain |
| 49,152 | v3 | v31: 23.02, 23.86 → **23.43** (42.68 ms a token); v32: 23.01, 23.87 → **23.43** (42.68 ms a token) | 0.00 | control | | |
|  | v4 | v41: 23.15, 24.02 → **23.57** (42.42 ms a token); v42: 23.15, 24.02 → **23.58** (42.41 ms a token) | 0.00 | +0.14 tok/s | +0.6 % | gain |

### Generation over all depths, tok/s

Every depth's tokens over their time, per load.

| Depth (tokens) | Arm | Per load | Spread | Against `v3` | | Class |
|---:|---|---|---:|---:|---:|---|
| all | v3 | v31: **40.24**; v32: **40.00** | 0.24 | control | | |
|  | v4 | v41: **40.58**; v42: **40.57** | 0.02 | +0.46 tok/s | +1.1 % | gain |

### Prompt reading, tok/s

Per load: the tokens read at that depth over their time (a prime request reads a chunk on top of the cache; without primes each decode request reads its prompt), with the tokens read and the tokens reused. Requests that read under 32 tokens are left out.

| Depth (tokens) | Arm | Per load | Spread | Against `v3` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | v3 | v31: 86.4 (64 read on 0 cached); v32: 83.8 (64 read on 0 cached) | 2.7 | control | | |
|  | v4 | v41: 91.1 (64 read on 0 cached); v42: 88.3 (64 read on 0 cached) | 2.8 | +4.6 tok/s | +5.4 % | gain |
| 4,096 | v3 | v31: 479.7 (4032 read on 0 cached); v32: 479.9 (4032 read on 0 cached) | 0.2 | control | | |
|  | v4 | v41: 549.3 (4032 read on 0 cached); v42: 566.8 (4032 read on 0 cached) | 17.5 | +78.3 tok/s | +16.3 % | gain |
| 8,192 | v3 | v31: 450.0 (4100 read on 4028 cached); v32: 432.1 (4100 read on 4028 cached) | 17.8 | control | | |
|  | v4 | v41: 524.7 (4100 read on 4028 cached); v42: 540.5 (4100 read on 4028 cached) | 15.8 | +91.6 tok/s | +20.8 % | gain |
| 16,384 | v3 | v31: 477.2 (12292 read on 4028 cached); v32: 475.7 (12292 read on 4028 cached) | 1.5 | control | | |
|  | v4 | v41: 520.4 (12292 read on 4028 cached); v42: 539.0 (12292 read on 4028 cached) | 18.5 | +53.3 tok/s | +11.2 % | gain |
| 32,768 | v3 | v31: 327.6 (16388 read on 16316 cached); v32: 326.0 (16388 read on 16316 cached) | 1.6 | control | | |
|  | v4 | v41: 338.6 (16388 read on 16316 cached); v42: 342.2 (16388 read on 16316 cached) | 3.6 | +13.6 tok/s | +4.2 % | gain |
| 49,152 | v3 | v31: 213.3 (16388 read on 32700 cached); v32: 214.1 (16388 read on 32700 cached) | 0.8 | control | | |
|  | v4 | v41: 221.2 (16388 read on 32700 cached); v42: 222.0 (16388 read on 32700 cached) | 0.8 | +7.9 tok/s | +3.7 % | gain |

### The draft and the step

Per load, pass 1: draft tokens accepted of drafted (the share), tokens a step (generated tokens over steps, a step being one pass of the main model: generated minus accepted), and the time of a step. Pass 2 is listed where its counters differ.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | v3 | v31: no draft, 1.00 a step, 21.2 ms a step; v32: no draft, 1.00 a step, 23.2 ms a step |
|  | v4 | v41: no draft, 1.00 a step, 21.3 ms a step; v42: no draft, 1.00 a step, 21.6 ms a step |
| 4,096 | v3 | v31: no draft, 1.00 a step, 13.5 ms a step; v32: no draft, 1.00 a step, 13.5 ms a step |
|  | v4 | v41: no draft, 1.00 a step, 13.3 ms a step; v42: no draft, 1.00 a step, 13.3 ms a step |
| 8,192 | v3 | v31: no draft, 1.00 a step, 21.6 ms a step; v32: no draft, 1.00 a step, 21.5 ms a step |
|  | v4 | v41: no draft, 1.00 a step, 21.2 ms a step; v42: no draft, 1.00 a step, 21.0 ms a step |
| 16,384 | v3 | v31: no draft, 1.00 a step, 15.6 ms a step; v32: no draft, 1.00 a step, 15.6 ms a step |
|  | v4 | v41: no draft, 1.00 a step, 15.4 ms a step; v42: no draft, 1.00 a step, 15.4 ms a step |
| 32,768 | v3 | v31: no draft, 1.00 a step, 38.8 ms a step; v32: no draft, 1.00 a step, 38.8 ms a step |
|  | v4 | v41: no draft, 1.00 a step, 38.4 ms a step; v42: no draft, 1.00 a step, 38.4 ms a step |
| 49,152 | v3 | v31: no draft, 1.00 a step, 43.4 ms a step; v32: no draft, 1.00 a step, 43.5 ms a step |
|  | v4 | v41: no draft, 1.00 a step, 43.2 ms a step; v42: no draft, 1.00 a step, 43.2 ms a step |

### The sampler pass

One request per depth and load with the server line's own sampler and a fixed seed: tok/s, the draft counters and the step.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | v3 | v31: 59.91 tok/s, no draft, 1.00 a step, 16.7 ms a step; v32: 55.76 tok/s, no draft, 1.00 a step, 17.9 ms a step |
|  | v4 | v41: 58.63 tok/s, no draft, 1.00 a step, 17.1 ms a step; v42: 60.39 tok/s, no draft, 1.00 a step, 16.6 ms a step |
| 4,096 | v3 | v31: 65.56 tok/s, no draft, 1.00 a step, 15.3 ms a step; v32: 65.52 tok/s, no draft, 1.00 a step, 15.3 ms a step |
|  | v4 | v41: 66.36 tok/s, no draft, 1.00 a step, 15.1 ms a step; v42: 66.38 tok/s, no draft, 1.00 a step, 15.1 ms a step |
| 8,192 | v3 | v31: 42.31 tok/s, no draft, 1.00 a step, 23.6 ms a step; v32: 42.81 tok/s, no draft, 1.00 a step, 23.4 ms a step |
|  | v4 | v41: 43.26 tok/s, no draft, 1.00 a step, 23.1 ms a step; v42: 43.25 tok/s, no draft, 1.00 a step, 23.1 ms a step |
| 16,384 | v3 | v31: 35.51 tok/s, no draft, 1.00 a step, 28.2 ms a step; v32: 35.52 tok/s, no draft, 1.00 a step, 28.2 ms a step |
|  | v4 | v41: 35.94 tok/s, no draft, 1.00 a step, 27.8 ms a step; v42: 35.91 tok/s, no draft, 1.00 a step, 27.8 ms a step |
| 32,768 | v3 | v31: 26.62 tok/s, no draft, 1.00 a step, 37.6 ms a step; v32: 26.62 tok/s, no draft, 1.00 a step, 37.6 ms a step |
|  | v4 | v41: 26.87 tok/s, no draft, 1.00 a step, 37.2 ms a step; v42: 26.86 tok/s, no draft, 1.00 a step, 37.2 ms a step |
| 49,152 | v3 | v31: 22.54 tok/s, no draft, 1.00 a step, 44.4 ms a step; v32: 22.50 tok/s, no draft, 1.00 a step, 44.4 ms a step |
|  | v4 | v41: 22.71 tok/s, no draft, 1.00 a step, 44.0 ms a step; v42: 22.70 tok/s, no draft, 1.00 a step, 44.0 ms a step |

### Checks

- Times: every load started on a machine checked quiet (valid=1).
- Prompts: 64 ids at depth 0, 4096 ids at depth 4,096, 8192 ids at depth 8,192, 16384 ids at depth 16,384, 32768 ids at depth 32,768, 49152 ids at depth 49,152; the same ids on every load and pass.
- Tokens out: 256 on every timed request.
- Greedy reply at depth 0: the same on every pass and load of every arm.
- Greedy reply at depth 4,096: the same on every pass and load of every arm.
- Greedy reply at depth 8,192: the same on every pass and load of every arm.
- Greedy reply at depth 16,384: the same on every pass and load of every arm.
- Greedy reply at depth 32,768: the same on every pass and load of every arm.
- Greedy reply at depth 49,152: the same on every pass and load of every arm.
- Primes (read on cached, the cache's source): 4,096: 4032 on 0 (none); 8,192: 4100 on 4028 (checkpoint); 16,384: 12292 on 4028 (checkpoint); 32,768: 16388 on 16316 (checkpoint); 49,152: 16388 on 32700 (checkpoint).
- Graph reallocations of the target, requests with one per load: v31: 0 of 24 requests; v41: 0 of 24 requests; v32: 0 of 24 requests; v42: 0 of 24 requests.
