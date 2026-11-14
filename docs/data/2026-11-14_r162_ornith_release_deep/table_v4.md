## The deep lane: v3, v4, v4s

Model file `Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf`, llama-server version: 1.0.0-dev (build 12832, commit eb312b7bce).

256 tokens per request at temperature 0, each request twice per load (pass 1 is the first request at a new depth, pass 2 the same request again). Timed loads, in their order: v31, v41, s1, v32, v42, s2.

| Arm | Loads | Draft | Context | Its own variables and flags |
|---|---|---|---:|---|
| v3 | v31, v32 | off | 49152 | — |
| v4 (control) | v41, v42 | off | 49152 | `GGML_CUDA_KEEPALIVE_EAGER=1 LLAMA_SERVER_YIELD=2 LLAMA_SAMPLING_TOP_K_ROWS=1` |
| v4s | s1, s2 | off | 49152 | `GGML_CUDA_KEEPALIVE_EAGER=1 LLAMA_SERVER_YIELD=2 LLAMA_SAMPLING_TOP_K_ROWS=1 GGML_CUDA_HOP_STAGED=1 GGML_CUDA_GRAPHS_FORCE=1`, `--ctx-checkpoints-vram 4 --ctx-checkpoints-incall --ubatch-taper short=600:64 -b 4096` |

### Generation, tok/s

Per load: pass 1, pass 2 → the load, with the load's time per token. A load's speed is its two passes' tokens over their time. Spread: the difference between the arm's fastest and slowest load. Class against `v4`: gain when the mean is above the control's by more than the larger of the two spreads, loss when below by more than it, else negligible.

| Depth (tokens) | Arm | Per load | Spread | Against `v4` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | v3 | v31: 18.17, 18.51 → **18.34** (54.53 ms a token); v32: 18.16, 18.50 → **18.32** (54.57 ms a token) | 0.01 | -0.06 tok/s | -0.3 % | loss |
|  | v4 | v41: 18.18, 18.59 → **18.38** (54.40 ms a token); v42: 18.19, 18.60 → **18.39** (54.38 ms a token) | 0.01 | control | | |
|  | v4s | s1: 18.24, 18.66 → **18.45** (54.20 ms a token); s2: 18.23, 18.64 → **18.43** (54.25 ms a token) | 0.02 | +0.05 tok/s | +0.3 % | gain |
| 4,096 | v3 | v31: 16.79, 16.84 → **16.82** (59.47 ms a token); v32: 16.77, 16.82 → **16.80** (59.54 ms a token) | 0.02 | -0.06 tok/s | -0.4 % | loss |
|  | v4 | v41: 16.81, 16.92 → **16.87** (59.28 ms a token); v42: 16.81, 16.92 → **16.86** (59.30 ms a token) | 0.01 | control | | |
|  | v4s | s1: 16.88, 17.00 → **16.94** (59.02 ms a token); s2: 16.87, 16.99 → **16.93** (59.08 ms a token) | 0.02 | +0.07 tok/s | +0.4 % | gain |
| 8,192 | v3 | v31: 15.47, 15.45 → **15.46** (64.69 ms a token); v32: 15.45, 15.45 → **15.45** (64.73 ms a token) | 0.01 | -0.09 tok/s | -0.6 % | loss |
|  | v4 | v41: 15.51, 15.58 → **15.54** (64.34 ms a token); v42: 15.50, 15.57 → **15.54** (64.37 ms a token) | 0.01 | control | | |
|  | v4s | s1: 15.50, 15.58 → **15.54** (64.36 ms a token); s2: 15.53, 15.60 → **15.57** (64.24 ms a token) | 0.03 | +0.01 tok/s | +0.1 % | negligible |
| 16,384 | v3 | v31: 13.33, 13.34 → **13.34** (74.99 ms a token); v32: 13.33, 13.33 → **13.33** (75.03 ms a token) | 0.01 | -0.05 tok/s | -0.4 % | loss |
|  | v4 | v41: 13.37, 13.41 → **13.39** (74.69 ms a token); v42: 13.36, 13.40 → **13.38** (74.72 ms a token) | 0.01 | control | | |
|  | v4s | s1: 13.34, 13.39 → **13.37** (74.82 ms a token); s2: 13.35, 13.39 → **13.37** (74.79 ms a token) | 0.01 | -0.02 tok/s | -0.1 % | loss |
| 32,768 | v3 | v31: 10.42, 10.43 → **10.43** (95.91 ms a token); v32: 10.41, 10.43 → **10.42** (95.96 ms a token) | 0.00 | -0.02 tok/s | -0.2 % | loss |
|  | v4 | v41: 10.44, 10.47 → **10.46** (95.63 ms a token); v42: 10.42, 10.45 → **10.44** (95.79 ms a token) | 0.02 | control | | |
|  | v4s | s1: 10.43, 10.46 → **10.44** (95.75 ms a token); s2: 10.42, 10.44 → **10.43** (95.89 ms a token) | 0.02 | -0.01 tok/s | -0.1 % | negligible |
| 48,640 | v3 | v31: 8.61, 8.61 → **8.61** (116.11 ms a token); v32: 8.61, 8.61 → **8.61** (116.17 ms a token) | 0.00 | -0.01 tok/s | -0.1 % | loss |
|  | v4 | v41: 8.62, 8.63 → **8.63** (115.94 ms a token); v42: 8.61, 8.63 → **8.62** (116.02 ms a token) | 0.01 | control | | |
|  | v4s | s1: 8.64, 8.66 → **8.65** (115.63 ms a token); s2: 8.62, 8.64 → **8.63** (115.89 ms a token) | 0.02 | +0.02 tok/s | +0.2 % | negligible |

### Generation over all depths, tok/s

Every depth's tokens over their time, per load.

| Depth (tokens) | Arm | Per load | Spread | Against `v4` | | Class |
|---:|---|---|---:|---:|---:|---|
| all | v3 | v31: **12.88**; v32: **12.88** | 0.01 | -0.04 tok/s | -0.3 % | loss |
|  | v4 | v41: **12.92**; v42: **12.92** | 0.01 | control | | |
|  | v4s | s1: **12.94**; s2: **12.93** | 0.01 | +0.01 tok/s | +0.1 % | gain |

### Prompt reading, tok/s

Per load: the tokens read at that depth over their time (a prime request reads a chunk on top of the cache; without primes each decode request reads its prompt), with the tokens read and the tokens reused. Requests that read under 32 tokens are left out.

| Depth (tokens) | Arm | Per load | Spread | Against `v4` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | v3 | v31: 24.6 (64 read on 0 cached); v32: 24.3 (64 read on 0 cached) | 0.3 | -0.1 tok/s | -0.5 % | negligible |
|  | v4 | v41: 24.6 (64 read on 0 cached); v42: 24.6 (64 read on 0 cached) | 0.0 | control | | |
|  | v4s | s1: 40.6 (64 read on 0 cached); s2: 40.6 (64 read on 0 cached) | 0.0 | +16.0 tok/s | +65.1 % | gain |
| 4,096 | v3 | v31: 132.4 (4032 read on 0 cached); v32: 132.2 (4032 read on 0 cached) | 0.2 | -1.5 tok/s | -1.1 % | loss |
|  | v4 | v41: 133.8 (4032 read on 0 cached); v42: 133.9 (4032 read on 0 cached) | 0.1 | control | | |
|  | v4s | s1: 169.4 (4032 read on 0 cached); s2: 168.8 (4032 read on 0 cached) | 0.6 | +35.3 tok/s | +26.4 % | gain |
| 8,192 | v3 | v31: 144.8 (4612 read on 3516 cached); v32: 144.5 (4612 read on 3516 cached) | 0.3 | +1.2 tok/s | +0.9 % | gain |
|  | v4 | v41: 143.2 (4612 read on 3516 cached); v42: 143.6 (4612 read on 3516 cached) | 0.4 | control | | |
|  | v4s | s1: 182.3 (4612 read on 3516 cached); s2: 182.2 (4612 read on 3516 cached) | 0.1 | +38.8 tok/s | +27.1 % | gain |
| 16,384 | v3 | v31: 193.8 (12804 read on 3516 cached); v32: 193.4 (12804 read on 3516 cached) | 0.3 | +1.8 tok/s | +0.9 % | gain |
|  | v4 | v41: 191.9 (12804 read on 3516 cached); v42: 191.7 (12804 read on 3516 cached) | 0.1 | control | | |
|  | v4s | s1: 219.8 (12804 read on 3516 cached); s2: 219.4 (12804 read on 3516 cached) | 0.4 | +27.8 tok/s | +14.5 % | gain |
| 32,768 | v3 | v31: 155.5 (16900 read on 15804 cached); v32: 155.5 (16900 read on 15804 cached) | 0.0 | +3.3 tok/s | +2.1 % | gain |
|  | v4 | v41: 152.3 (16900 read on 15804 cached); v42: 152.3 (16900 read on 15804 cached) | 0.0 | control | | |
|  | v4s | s1: 168.8 (16900 read on 15804 cached); s2: 168.6 (16900 read on 15804 cached) | 0.2 | +16.4 tok/s | +10.8 % | gain |
| 48,640 | v3 | v31: 120.7 (16388 read on 32188 cached); v32: 120.5 (16388 read on 32188 cached) | 0.1 | +1.1 tok/s | +1.0 % | gain |
|  | v4 | v41: 119.7 (16388 read on 32188 cached); v42: 119.2 (16388 read on 32188 cached) | 0.4 | control | | |
|  | v4s | s1: 136.7 (16388 read on 32188 cached); s2: 136.4 (16388 read on 32188 cached) | 0.4 | +17.1 tok/s | +14.3 % | gain |

### The draft and the step

Per load, pass 1: draft tokens accepted of drafted (the share), tokens a step (generated tokens over steps, a step being one pass of the main model: generated minus accepted), and the time of a step. Pass 2 is listed where its counters differ.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | v3 | v31: no draft, 1.00 a step, 55.0 ms a step; v32: no draft, 1.00 a step, 55.1 ms a step |
|  | v4 | v41: no draft, 1.00 a step, 55.0 ms a step; v42: no draft, 1.00 a step, 55.0 ms a step |
|  | v4s | s1: no draft, 1.00 a step, 54.8 ms a step; s2: no draft, 1.00 a step, 54.8 ms a step |
| 4,096 | v3 | v31: no draft, 1.00 a step, 59.6 ms a step; v32: no draft, 1.00 a step, 59.6 ms a step |
|  | v4 | v41: no draft, 1.00 a step, 59.5 ms a step; v42: no draft, 1.00 a step, 59.5 ms a step |
|  | v4s | s1: no draft, 1.00 a step, 59.2 ms a step; s2: no draft, 1.00 a step, 59.3 ms a step |
| 8,192 | v3 | v31: no draft, 1.00 a step, 64.6 ms a step; v32: no draft, 1.00 a step, 64.7 ms a step |
|  | v4 | v41: no draft, 1.00 a step, 64.5 ms a step; v42: no draft, 1.00 a step, 64.5 ms a step |
|  | v4s | s1: no draft, 1.00 a step, 64.5 ms a step; s2: no draft, 1.00 a step, 64.4 ms a step |
| 16,384 | v3 | v31: no draft, 1.00 a step, 75.0 ms a step; v32: no draft, 1.00 a step, 75.0 ms a step |
|  | v4 | v41: no draft, 1.00 a step, 74.8 ms a step; v42: no draft, 1.00 a step, 74.8 ms a step |
|  | v4s | s1: no draft, 1.00 a step, 74.9 ms a step; s2: no draft, 1.00 a step, 74.9 ms a step |
| 32,768 | v3 | v31: no draft, 1.00 a step, 95.9 ms a step; v32: no draft, 1.00 a step, 96.0 ms a step |
|  | v4 | v41: no draft, 1.00 a step, 95.8 ms a step; v42: no draft, 1.00 a step, 95.9 ms a step |
|  | v4s | s1: no draft, 1.00 a step, 95.9 ms a step; s2: no draft, 1.00 a step, 96.0 ms a step |
| 48,640 | v3 | v31: no draft, 1.00 a step, 116.1 ms a step; v32: no draft, 1.00 a step, 116.2 ms a step |
|  | v4 | v41: no draft, 1.00 a step, 116.0 ms a step; v42: no draft, 1.00 a step, 116.1 ms a step |
|  | v4s | s1: no draft, 1.00 a step, 115.7 ms a step; s2: no draft, 1.00 a step, 116.0 ms a step |

### The sampler pass

One request per depth and load with the server line's own sampler and a fixed seed: tok/s, the draft counters and the step.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | v3 | v31: 18.11 tok/s, no draft, 1.00 a step, 55.2 ms a step; v32: 18.14 tok/s, no draft, 1.00 a step, 55.1 ms a step |
|  | v4 | v41: 18.22 tok/s, no draft, 1.00 a step, 54.9 ms a step; v42: 18.21 tok/s, no draft, 1.00 a step, 54.9 ms a step |
|  | v4s | s1: 18.21 tok/s, no draft, 1.00 a step, 54.9 ms a step; s2: 18.20 tok/s, no draft, 1.00 a step, 54.9 ms a step |
| 4,096 | v3 | v31: 16.85 tok/s, no draft, 1.00 a step, 59.3 ms a step; v32: 16.83 tok/s, no draft, 1.00 a step, 59.4 ms a step |
|  | v4 | v41: 16.93 tok/s, no draft, 1.00 a step, 59.1 ms a step; v42: 16.92 tok/s, no draft, 1.00 a step, 59.1 ms a step |
|  | v4s | s1: 16.94 tok/s, no draft, 1.00 a step, 59.0 ms a step; s2: 16.93 tok/s, no draft, 1.00 a step, 59.1 ms a step |
| 8,192 | v3 | v31: 15.48 tok/s, no draft, 1.00 a step, 64.6 ms a step; v32: 15.47 tok/s, no draft, 1.00 a step, 64.6 ms a step |
|  | v4 | v41: 15.55 tok/s, no draft, 1.00 a step, 64.3 ms a step; v42: 15.55 tok/s, no draft, 1.00 a step, 64.3 ms a step |
|  | v4s | s1: 15.56 tok/s, no draft, 1.00 a step, 64.3 ms a step; s2: 15.58 tok/s, no draft, 1.00 a step, 64.2 ms a step |
| 16,384 | v3 | v31: 13.32 tok/s, no draft, 1.00 a step, 75.1 ms a step; v32: 13.33 tok/s, no draft, 1.00 a step, 75.0 ms a step |
|  | v4 | v41: 13.36 tok/s, no draft, 1.00 a step, 74.9 ms a step; v42: 13.36 tok/s, no draft, 1.00 a step, 74.8 ms a step |
|  | v4s | s1: 13.36 tok/s, no draft, 1.00 a step, 74.9 ms a step; s2: 13.36 tok/s, no draft, 1.00 a step, 74.8 ms a step |
| 32,768 | v3 | v31: 10.43 tok/s, no draft, 1.00 a step, 95.8 ms a step; v32: 10.43 tok/s, no draft, 1.00 a step, 95.9 ms a step |
|  | v4 | v41: 10.45 tok/s, no draft, 1.00 a step, 95.7 ms a step; v42: 10.44 tok/s, no draft, 1.00 a step, 95.7 ms a step |
|  | v4s | s1: 10.44 tok/s, no draft, 1.00 a step, 95.8 ms a step; s2: 10.43 tok/s, no draft, 1.00 a step, 95.9 ms a step |
| 48,640 | v3 | v31: 8.60 tok/s, no draft, 1.00 a step, 116.3 ms a step; v32: 8.59 tok/s, no draft, 1.00 a step, 116.3 ms a step |
|  | v4 | v41: 8.61 tok/s, no draft, 1.00 a step, 116.2 ms a step; v42: 8.60 tok/s, no draft, 1.00 a step, 116.3 ms a step |
|  | v4s | s1: 8.64 tok/s, no draft, 1.00 a step, 115.7 ms a step; s2: 8.64 tok/s, no draft, 1.00 a step, 115.8 ms a step |

### Checks

- Times: every load started on a machine checked quiet (valid=1).
- Prompts: 64 ids at depth 0, 4096 ids at depth 4,096, 8192 ids at depth 8,192, 16384 ids at depth 16,384, 32768 ids at depth 32,768, 48640 ids at depth 48,640; the same ids on every load and pass.
- Tokens out: 256 on every timed request.
- Greedy reply at depth 0: the same on every pass and load of every arm.
- Greedy reply at depth 4,096: the same on every pass and load of every arm.
- Greedy reply at depth 8,192: **differs from `v4` in v4s.**
- Greedy reply at depth 16,384: **differs from `v4` in v4s.**
- Greedy reply at depth 32,768: **differs from `v4` in v4s.**
- Greedy reply at depth 48,640: the same on every pass and load of every arm.
- Primes (read on cached, the cache's source): 4,096: 4032 on 0 (none); 8,192: 4612 on 3516 (checkpoint); 16,384: 12804 on 3516 (checkpoint); 32,768: 16900 on 15804 (checkpoint); 48,640: 16388 on 32188 (checkpoint).
- Graph reallocations of the target, requests with one per load: v31: 0 of 24 requests; v41: 0 of 24 requests; s1: 0 of 24 requests; v32: 0 of 24 requests; v42: 0 of 24 requests; s2: 0 of 24 requests.
