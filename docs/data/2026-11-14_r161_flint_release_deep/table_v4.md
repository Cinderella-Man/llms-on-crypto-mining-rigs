## The deep lane: v3, v4, v4s

Model file `flint-near-q8-p065.gguf`, llama-server version: 1.0.0-dev (build 12832, commit eb312b7bce).

256 tokens per request at temperature 0, each request twice per load (pass 1 is the first request at a new depth, pass 2 the same request again). Timed loads, in their order: v31, v41, s1, v32, v42, s2.

| Arm | Loads | Draft | Context | Its own variables and flags |
|---|---|---|---:|---|
| v3 | v31, v32 | off | 262144 | — |
| v4 (control) | v41, v42 | off | 262144 | `LLAMA_KV_SEQ_RM_BOUND=1 LLAMA_SERVER_YIELD=2 LLAMA_SAMPLING_TOP_K_ROWS=1` |
| v4s | s1, s2 | off | 262144 | `LLAMA_KV_SEQ_RM_BOUND=1 LLAMA_SERVER_YIELD=2 LLAMA_SAMPLING_TOP_K_ROWS=1 GGML_CUDA_KEEPALIVE=decode GGML_CUDA_KEEPALIVE_EAGER=1 GGML_CUDA_HOP_STAGED=1 GGML_CUDA_GRAPHS_FORCE=1`, `--ctx-checkpoints-vram 4 --ctx-checkpoints-incall --ubatch-taper short=600:64` |

### Generation, tok/s

Per load: pass 1, pass 2 → the load, with the load's time per token. A load's speed is its two passes' tokens over their time. Spread: the difference between the arm's fastest and slowest load. Class against `v4`: gain when the mean is above the control's by more than the larger of the two spreads, loss when below by more than it, else negligible.

| Depth (tokens) | Arm | Per load | Spread | Against `v4` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | v3 | v31: 46.02, 54.59 → **49.94** (20.02 ms a token); v32: 46.14, 54.46 → **49.96** (20.02 ms a token) | 0.02 | -0.33 tok/s | -0.7 % | negligible |
|  | v4 | v41: 45.23, 55.34 → **49.78** (20.09 ms a token); v42: 46.87, 55.42 → **50.79** (19.69 ms a token) | 1.01 | control | | |
|  | v4s | s1: 47.61, 55.78 → **51.37** (19.47 ms a token); s2: 49.14, 55.74 → **52.23** (19.15 ms a token) | 0.86 | +1.52 tok/s | +3.0 % | gain |
| 4,096 | v3 | v31: 73.94, 79.26 → **76.51** (13.07 ms a token); v32: 73.99, 79.27 → **76.54** (13.06 ms a token) | 0.04 | -0.80 tok/s | -1.0 % | loss |
|  | v4 | v41: 74.66, 80.16 → **77.31** (12.93 ms a token); v42: 74.68, 80.17 → **77.33** (12.93 ms a token) | 0.01 | control | | |
|  | v4s | s1: 73.32, 79.79 → **76.42** (13.09 ms a token); s2: 73.25, 79.82 → **76.40** (13.09 ms a token) | 0.02 | -0.91 tok/s | -1.2 % | loss |
| 8,192 | v3 | v31: 46.30, 50.63 → **48.37** (20.68 ms a token); v32: 46.31, 50.62 → **48.37** (20.68 ms a token) | 0.00 | -0.33 tok/s | -0.7 % | loss |
|  | v4 | v41: 46.63, 51.12 → **48.77** (20.51 ms a token); v42: 46.44, 51.02 → **48.62** (20.57 ms a token) | 0.15 | control | | |
|  | v4s | s1: 45.84, 50.59 → **48.10** (20.79 ms a token); s2: 46.01, 50.56 → **48.18** (20.76 ms a token) | 0.08 | -0.56 tok/s | -1.1 % | loss |
| 16,384 | v3 | v31: 63.90, 69.46 → **66.56** (15.02 ms a token); v32: 64.14, 69.56 → **66.74** (14.98 ms a token) | 0.18 | -0.40 tok/s | -0.6 % | loss |
|  | v4 | v41: 64.11, 69.96 → **66.91** (14.95 ms a token); v42: 64.62, 69.99 → **67.20** (14.88 ms a token) | 0.29 | control | | |
|  | v4s | s1: 63.34, 69.11 → **66.10** (15.13 ms a token); s2: 63.36, 69.15 → **66.13** (15.12 ms a token) | 0.03 | -0.94 tok/s | -1.4 % | loss |
| 32,768 | v3 | v31: 25.70, 26.67 → **26.18** (38.20 ms a token); v32: 25.67, 26.66 → **26.16** (38.23 ms a token) | 0.02 | -0.19 tok/s | -0.7 % | loss |
|  | v4 | v41: 25.85, 26.87 → **26.35** (37.95 ms a token); v42: 25.87, 26.87 → **26.36** (37.94 ms a token) | 0.01 | control | | |
|  | v4s | s1: 25.31, 26.32 → **25.81** (38.75 ms a token); s2: 25.35, 26.33 → **25.83** (38.72 ms a token) | 0.02 | -0.54 tok/s | -2.0 % | loss |
| 49,152 | v3 | v31: 22.87, 23.72 → **23.29** (42.94 ms a token); v32: 22.90, 23.75 → **23.31** (42.89 ms a token) | 0.03 | -0.14 tok/s | -0.6 % | loss |
|  | v4 | v41: 23.03, 23.87 → **23.44** (42.65 ms a token); v42: 23.02, 23.87 → **23.44** (42.67 ms a token) | 0.01 | control | | |
|  | v4s | s1: 22.28, 23.15 → **22.71** (44.04 ms a token); s2: 22.33, 23.17 → **22.74** (43.97 ms a token) | 0.04 | -0.72 tok/s | -3.1 % | loss |

### Generation over all depths, tok/s

Every depth's tokens over their time, per load.

| Depth (tokens) | Arm | Per load | Spread | Against `v4` | | Class |
|---:|---|---|---:|---:|---:|---|
| all | v3 | v31: **40.02**; v32: **40.04** | 0.02 | -0.28 tok/s | -0.7 % | loss |
|  | v4 | v41: **40.25**; v42: **40.36** | 0.11 | control | | |
|  | v4s | s1: **39.67**; s2: **39.79** | 0.12 | -0.58 tok/s | -1.4 % | loss |

### Prompt reading, tok/s

Per load: the tokens read at that depth over their time (a prime request reads a chunk on top of the cache; without primes each decode request reads its prompt), with the tokens read and the tokens reused. Requests that read under 32 tokens are left out.

| Depth (tokens) | Arm | Per load | Spread | Against `v4` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | v3 | v31: 85.1 (64 read on 0 cached); v32: 85.2 (64 read on 0 cached) | 0.1 | +0.9 tok/s | +1.1 % | gain |
|  | v4 | v41: 84.5 (64 read on 0 cached); v42: 84.0 (64 read on 0 cached) | 0.5 | control | | |
|  | v4s | s1: 111.1 (64 read on 0 cached); s2: 110.2 (64 read on 0 cached) | 0.9 | +26.3 tok/s | +31.3 % | gain |
| 4,096 | v3 | v31: 477.5 (4032 read on 0 cached); v32: 478.2 (4032 read on 0 cached) | 0.8 | -31.7 tok/s | -6.2 % | loss |
|  | v4 | v41: 503.3 (4032 read on 0 cached); v42: 515.7 (4032 read on 0 cached) | 12.3 | control | | |
|  | v4s | s1: 539.5 (4032 read on 0 cached); s2: 542.0 (4032 read on 0 cached) | 2.5 | +31.3 tok/s | +6.1 % | gain |
| 8,192 | v3 | v31: 442.6 (4100 read on 4028 cached); v32: 446.7 (4100 read on 4028 cached) | 4.0 | -15.3 tok/s | -3.3 % | negligible |
|  | v4 | v41: 443.6 (4100 read on 4028 cached); v42: 476.3 (4100 read on 4028 cached) | 32.7 | control | | |
|  | v4s | s1: 457.4 (4100 read on 4028 cached); s2: 511.4 (4100 read on 4028 cached) | 54.0 | +24.5 tok/s | +5.3 % | negligible |
| 16,384 | v3 | v31: 472.6 (12292 read on 4028 cached); v32: 475.4 (12292 read on 4028 cached) | 2.8 | -15.7 tok/s | -3.2 % | loss |
|  | v4 | v41: 488.5 (12292 read on 4028 cached); v42: 490.8 (12292 read on 4028 cached) | 2.3 | control | | |
|  | v4s | s1: 462.1 (12292 read on 4028 cached); s2: 466.7 (12292 read on 4028 cached) | 4.5 | -25.3 tok/s | -5.2 % | loss |
| 32,768 | v3 | v31: 321.9 (16388 read on 16316 cached); v32: 322.3 (16388 read on 16316 cached) | 0.4 | -10.4 tok/s | -3.1 % | loss |
|  | v4 | v41: 332.7 (16388 read on 16316 cached); v42: 332.2 (16388 read on 16316 cached) | 0.5 | control | | |
|  | v4s | s1: 263.8 (16388 read on 16316 cached); s2: 264.0 (16388 read on 16316 cached) | 0.2 | -68.5 tok/s | -20.6 % | loss |
| 49,152 | v3 | v31: 212.7 (16388 read on 32700 cached); v32: 213.6 (16388 read on 32700 cached) | 0.9 | -5.5 tok/s | -2.5 % | loss |
|  | v4 | v41: 218.0 (16388 read on 32700 cached); v42: 219.2 (16388 read on 32700 cached) | 1.2 | control | | |
|  | v4s | s1: 180.8 (16388 read on 32700 cached); s2: 180.7 (16388 read on 32700 cached) | 0.1 | -37.8 tok/s | -17.3 % | loss |

### The draft and the step

Per load, pass 1: draft tokens accepted of drafted (the share), tokens a step (generated tokens over steps, a step being one pass of the main model: generated minus accepted), and the time of a step. Pass 2 is listed where its counters differ.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | v3 | v31: no draft, 1.00 a step, 21.7 ms a step; v32: no draft, 1.00 a step, 21.7 ms a step |
|  | v4 | v41: no draft, 1.00 a step, 22.1 ms a step; v42: no draft, 1.00 a step, 21.3 ms a step |
|  | v4s | s1: no draft, 1.00 a step, 21.0 ms a step; s2: no draft, 1.00 a step, 20.4 ms a step |
| 4,096 | v3 | v31: no draft, 1.00 a step, 13.5 ms a step; v32: no draft, 1.00 a step, 13.5 ms a step |
|  | v4 | v41: no draft, 1.00 a step, 13.4 ms a step; v42: no draft, 1.00 a step, 13.4 ms a step |
|  | v4s | s1: no draft, 1.00 a step, 13.6 ms a step; s2: no draft, 1.00 a step, 13.7 ms a step |
| 8,192 | v3 | v31: no draft, 1.00 a step, 21.6 ms a step; v32: no draft, 1.00 a step, 21.6 ms a step |
|  | v4 | v41: no draft, 1.00 a step, 21.4 ms a step; v42: no draft, 1.00 a step, 21.5 ms a step |
|  | v4s | s1: no draft, 1.00 a step, 21.8 ms a step; s2: no draft, 1.00 a step, 21.7 ms a step |
| 16,384 | v3 | v31: no draft, 1.00 a step, 15.6 ms a step; v32: no draft, 1.00 a step, 15.6 ms a step |
|  | v4 | v41: no draft, 1.00 a step, 15.6 ms a step; v42: no draft, 1.00 a step, 15.5 ms a step |
|  | v4s | s1: no draft, 1.00 a step, 15.8 ms a step; s2: no draft, 1.00 a step, 15.8 ms a step |
| 32,768 | v3 | v31: no draft, 1.00 a step, 38.9 ms a step; v32: no draft, 1.00 a step, 39.0 ms a step |
|  | v4 | v41: no draft, 1.00 a step, 38.7 ms a step; v42: no draft, 1.00 a step, 38.7 ms a step |
|  | v4s | s1: no draft, 1.00 a step, 39.5 ms a step; s2: no draft, 1.00 a step, 39.5 ms a step |
| 49,152 | v3 | v31: no draft, 1.00 a step, 43.7 ms a step; v32: no draft, 1.00 a step, 43.7 ms a step |
|  | v4 | v41: no draft, 1.00 a step, 43.4 ms a step; v42: no draft, 1.00 a step, 43.4 ms a step |
|  | v4s | s1: no draft, 1.00 a step, 44.9 ms a step; s2: no draft, 1.00 a step, 44.8 ms a step |

### The sampler pass

One request per depth and load with the server line's own sampler and a fixed seed: tok/s, the draft counters and the step.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | v3 | v31: 59.67 tok/s, no draft, 1.00 a step, 16.8 ms a step; v32: 62.04 tok/s, no draft, 1.00 a step, 16.1 ms a step |
|  | v4 | v41: 59.37 tok/s, no draft, 1.00 a step, 16.8 ms a step; v42: 59.38 tok/s, no draft, 1.00 a step, 16.8 ms a step |
|  | v4s | s1: 57.06 tok/s, no draft, 1.00 a step, 17.5 ms a step; s2: 61.46 tok/s, no draft, 1.00 a step, 16.3 ms a step |
| 4,096 | v3 | v31: 65.35 tok/s, no draft, 1.00 a step, 15.3 ms a step; v32: 65.40 tok/s, no draft, 1.00 a step, 15.3 ms a step |
|  | v4 | v41: 66.20 tok/s, no draft, 1.00 a step, 15.1 ms a step; v42: 66.15 tok/s, no draft, 1.00 a step, 15.1 ms a step |
|  | v4s | s1: 65.83 tok/s, no draft, 1.00 a step, 15.2 ms a step; s2: 65.78 tok/s, no draft, 1.00 a step, 15.2 ms a step |
| 8,192 | v3 | v31: 42.61 tok/s, no draft, 1.00 a step, 23.5 ms a step; v32: 42.64 tok/s, no draft, 1.00 a step, 23.5 ms a step |
|  | v4 | v41: 43.48 tok/s, no draft, 1.00 a step, 23.0 ms a step; v42: 43.09 tok/s, no draft, 1.00 a step, 23.2 ms a step |
|  | v4s | s1: 39.53 tok/s, no draft, 1.00 a step, 25.3 ms a step; s2: 42.63 tok/s, no draft, 1.00 a step, 23.5 ms a step |
| 16,384 | v3 | v31: 35.34 tok/s, no draft, 1.00 a step, 28.3 ms a step; v32: 35.33 tok/s, no draft, 1.00 a step, 28.3 ms a step |
|  | v4 | v41: 35.66 tok/s, no draft, 1.00 a step, 28.0 ms a step; v42: 35.72 tok/s, no draft, 1.00 a step, 28.0 ms a step |
|  | v4s | s1: 35.32 tok/s, no draft, 1.00 a step, 28.3 ms a step; s2: 35.34 tok/s, no draft, 1.00 a step, 28.3 ms a step |
| 32,768 | v3 | v31: 26.52 tok/s, no draft, 1.00 a step, 37.7 ms a step; v32: 26.50 tok/s, no draft, 1.00 a step, 37.7 ms a step |
|  | v4 | v41: 26.69 tok/s, no draft, 1.00 a step, 37.5 ms a step; v42: 26.71 tok/s, no draft, 1.00 a step, 37.4 ms a step |
|  | v4s | s1: 26.18 tok/s, no draft, 1.00 a step, 38.2 ms a step; s2: 26.21 tok/s, no draft, 1.00 a step, 38.2 ms a step |
| 49,152 | v3 | v31: 22.39 tok/s, no draft, 1.00 a step, 44.7 ms a step; v32: 22.40 tok/s, no draft, 1.00 a step, 44.6 ms a step |
|  | v4 | v41: 22.57 tok/s, no draft, 1.00 a step, 44.3 ms a step; v42: 22.56 tok/s, no draft, 1.00 a step, 44.3 ms a step |
|  | v4s | s1: 21.88 tok/s, no draft, 1.00 a step, 45.7 ms a step; s2: 21.90 tok/s, no draft, 1.00 a step, 45.7 ms a step |

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
- Graph reallocations of the target, requests with one per load: v31: 0 of 24 requests; v41: 0 of 24 requests; s1: 0 of 24 requests; v32: 0 of 24 requests; v42: 0 of 24 requests; s2: 0 of 24 requests.
