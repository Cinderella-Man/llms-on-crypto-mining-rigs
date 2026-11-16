## The deep lane: v4, v4gf, v4hs

Model file `flint-near-q8-p065.gguf`, llama-server version: 1.0.0-dev (build 12832, commit eb312b7bce).

256 tokens per request at temperature 0, each request twice per load (pass 1 is the first request at a new depth, pass 2 the same request again). Timed loads, in their order: v41, gf1, hs1, v42, gf2, hs2.

| Arm | Loads | Draft | Context | Its own variables and flags |
|---|---|---|---:|---|
| v4 (control) | v41, v42 | off | 262144 | — |
| v4gf | gf1, gf2 | off | 262144 | `GGML_CUDA_GRAPHS_FORCE=1` |
| v4hs | hs1, hs2 | off | 262144 | `GGML_CUDA_HOP_STAGED=1` |

### Generation, tok/s

Per load: pass 1, pass 2 → the load, with the load's time per token. A load's speed is its two passes' tokens over their time. Spread: the difference between the arm's fastest and slowest load. Class against `v4`: gain when the mean is above the control's by more than the larger of the two spreads, loss when below by more than it, else negligible.

| Depth (tokens) | Arm | Per load | Spread | Against `v4` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | v4 | v41: 46.41, 55.50 → **50.55** (19.78 ms a token); v42: 43.74, 55.79 → **49.04** (20.39 ms a token) | 1.51 | control | | |
|  | v4gf | gf1: 46.93, 54.76 → **50.54** (19.79 ms a token); gf2: 46.71, 55.08 → **50.55** (19.78 ms a token) | 0.01 | +0.75 tok/s | +1.5 % | negligible |
|  | v4hs | hs1: 48.32, 57.02 → **52.31** (19.12 ms a token); hs2: 48.77, 56.91 → **52.52** (19.04 ms a token) | 0.22 | +2.62 tok/s | +5.3 % | gain |
| 4,096 | v4 | v41: 74.56, 80.33 → **77.34** (12.93 ms a token); v42: 74.67, 80.43 → **77.45** (12.91 ms a token) | 0.11 | control | | |
|  | v4gf | gf1: 74.66, 80.34 → **77.40** (12.92 ms a token); gf2: 74.77, 80.30 → **77.44** (12.91 ms a token) | 0.04 | +0.03 tok/s | +0.0 % | negligible |
|  | v4hs | hs1: 76.05, 81.44 → **78.65** (12.71 ms a token); hs2: 76.15, 81.52 → **78.74** (12.70 ms a token) | 0.09 | +1.31 tok/s | +1.7 % | gain |
| 8,192 | v4 | v41: 46.75, 51.22 → **48.88** (20.46 ms a token); v42: 47.81, 51.23 → **49.46** (20.22 ms a token) | 0.58 | control | | |
|  | v4gf | gf1: 47.44, 51.23 → **49.26** (20.30 ms a token); gf2: 47.73, 51.23 → **49.42** (20.23 ms a token) | 0.16 | +0.17 tok/s | +0.3 % | negligible |
|  | v4hs | hs1: 47.35, 51.78 → **49.47** (20.22 ms a token); hs2: 47.16, 51.76 → **49.35** (20.26 ms a token) | 0.11 | +0.24 tok/s | +0.5 % | negligible |
| 16,384 | v4 | v41: 64.59, 70.21 → **67.28** (14.86 ms a token); v42: 64.89, 70.27 → **67.48** (14.82 ms a token) | 0.19 | control | | |
|  | v4gf | gf1: 64.87, 70.30 → **67.48** (14.82 ms a token); gf2: 64.88, 70.30 → **67.48** (14.82 ms a token) | 0.00 | +0.10 tok/s | +0.2 % | negligible |
|  | v4hs | hs1: 65.37, 71.03 → **68.08** (14.69 ms a token); hs2: 65.63, 71.10 → **68.26** (14.65 ms a token) | 0.17 | +0.79 tok/s | +1.2 % | gain |
| 32,768 | v4 | v41: 25.89, 26.91 → **26.39** (37.89 ms a token); v42: 26.01, 27.00 → **26.50** (37.74 ms a token) | 0.10 | control | | |
|  | v4gf | gf1: 25.98, 26.96 → **26.46** (37.79 ms a token); gf2: 26.01, 27.00 → **26.50** (37.74 ms a token) | 0.03 | +0.04 tok/s | +0.1 % | negligible |
|  | v4hs | hs1: 26.16, 27.22 → **26.68** (37.48 ms a token); hs2: 26.17, 27.25 → **26.70** (37.45 ms a token) | 0.02 | +0.24 tok/s | +0.9 % | gain |
| 49,152 | v4 | v41: 23.12, 23.95 → **23.52** (42.51 ms a token); v42: 23.16, 24.00 → **23.57** (42.42 ms a token) | 0.05 | control | | |
|  | v4gf | gf1: 23.08, 23.97 → **23.51** (42.53 ms a token); gf2: 23.09, 23.99 → **23.53** (42.49 ms a token) | 0.02 | -0.03 tok/s | -0.1 % | negligible |
|  | v4hs | hs1: 23.30, 24.17 → **23.73** (42.15 ms a token); hs2: 23.31, 24.18 → **23.74** (42.13 ms a token) | 0.01 | +0.18 tok/s | +0.8 % | gain |

### Generation over all depths, tok/s

Every depth's tokens over their time, per load.

| Depth (tokens) | Arm | Per load | Spread | Against `v4` | | Class |
|---:|---|---|---:|---:|---:|---|
| all | v4 | v41: **40.42**; v42: **40.40** | 0.02 | control | | |
|  | v4gf | gf1: **40.50**; gf2: **40.55** | 0.04 | +0.11 tok/s | +0.3 % | gain |
|  | v4hs | hs1: **40.99**; hs2: **41.03** | 0.04 | +0.60 tok/s | +1.5 % | gain |

### Prompt reading, tok/s

Per load: the tokens read at that depth over their time (a prime request reads a chunk on top of the cache; without primes each decode request reads its prompt), with the tokens read and the tokens reused. Requests that read under 32 tokens are left out.

| Depth (tokens) | Arm | Per load | Spread | Against `v4` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | v4 | v41: 86.4 (64 read on 0 cached); v42: 87.1 (64 read on 0 cached) | 0.7 | control | | |
|  | v4gf | gf1: 85.5 (64 read on 0 cached); gf2: 86.4 (64 read on 0 cached) | 0.9 | -0.8 tok/s | -0.9 % | negligible |
|  | v4hs | hs1: 88.7 (64 read on 0 cached); hs2: 89.4 (64 read on 0 cached) | 0.7 | +2.3 tok/s | +2.6 % | gain |
| 4,096 | v4 | v41: 519.0 (4032 read on 0 cached); v42: 523.4 (4032 read on 0 cached) | 4.4 | control | | |
|  | v4gf | gf1: 541.9 (4032 read on 0 cached); gf2: 464.8 (4032 read on 0 cached) | 77.0 | -17.9 tok/s | -3.4 % | negligible |
|  | v4hs | hs1: 532.6 (4032 read on 0 cached); hs2: 447.6 (4032 read on 0 cached) | 84.9 | -31.1 tok/s | -6.0 % | negligible |
| 8,192 | v4 | v41: 478.2 (4100 read on 4028 cached); v42: 479.2 (4100 read on 4028 cached) | 1.0 | control | | |
|  | v4gf | gf1: 498.5 (4100 read on 4028 cached); gf2: 483.6 (4100 read on 4028 cached) | 15.0 | +12.3 tok/s | +2.6 % | negligible |
|  | v4hs | hs1: 446.1 (4100 read on 4028 cached); hs2: 437.4 (4100 read on 4028 cached) | 8.6 | -37.0 tok/s | -7.7 % | loss |
| 16,384 | v4 | v41: 493.8 (12292 read on 4028 cached); v42: 483.1 (12292 read on 4028 cached) | 10.7 | control | | |
|  | v4gf | gf1: 516.4 (12292 read on 4028 cached); gf2: 514.5 (12292 read on 4028 cached) | 1.9 | +27.1 tok/s | +5.5 % | gain |
|  | v4hs | hs1: 498.5 (12292 read on 4028 cached); hs2: 498.4 (12292 read on 4028 cached) | 0.1 | +10.0 tok/s | +2.1 % | negligible |
| 32,768 | v4 | v41: 338.6 (16388 read on 16316 cached); v42: 332.6 (16388 read on 16316 cached) | 6.1 | control | | |
|  | v4gf | gf1: 334.7 (16388 read on 16316 cached); gf2: 334.5 (16388 read on 16316 cached) | 0.3 | -1.0 tok/s | -0.3 % | negligible |
|  | v4hs | hs1: 335.1 (16388 read on 16316 cached); hs2: 334.5 (16388 read on 16316 cached) | 0.5 | -0.8 tok/s | -0.2 % | negligible |
| 49,152 | v4 | v41: 220.0 (16388 read on 32700 cached); v42: 218.2 (16388 read on 32700 cached) | 1.8 | control | | |
|  | v4gf | gf1: 218.8 (16388 read on 32700 cached); gf2: 217.9 (16388 read on 32700 cached) | 0.9 | -0.8 tok/s | -0.4 % | negligible |
|  | v4hs | hs1: 217.9 (16388 read on 32700 cached); hs2: 218.3 (16388 read on 32700 cached) | 0.3 | -1.1 tok/s | -0.5 % | negligible |

### The draft and the step

Per load, pass 1: draft tokens accepted of drafted (the share), tokens a step (generated tokens over steps, a step being one pass of the main model: generated minus accepted), and the time of a step. Pass 2 is listed where its counters differ.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | v4 | v41: no draft, 1.00 a step, 21.5 ms a step; v42: no draft, 1.00 a step, 22.9 ms a step |
|  | v4gf | gf1: no draft, 1.00 a step, 21.3 ms a step; gf2: no draft, 1.00 a step, 21.4 ms a step |
|  | v4hs | hs1: no draft, 1.00 a step, 20.7 ms a step; hs2: no draft, 1.00 a step, 20.5 ms a step |
| 4,096 | v4 | v41: no draft, 1.00 a step, 13.4 ms a step; v42: no draft, 1.00 a step, 13.4 ms a step |
|  | v4gf | gf1: no draft, 1.00 a step, 13.4 ms a step; gf2: no draft, 1.00 a step, 13.4 ms a step |
|  | v4hs | hs1: no draft, 1.00 a step, 13.1 ms a step; hs2: no draft, 1.00 a step, 13.1 ms a step |
| 8,192 | v4 | v41: no draft, 1.00 a step, 21.4 ms a step; v42: no draft, 1.00 a step, 20.9 ms a step |
|  | v4gf | gf1: no draft, 1.00 a step, 21.1 ms a step; gf2: no draft, 1.00 a step, 21.0 ms a step |
|  | v4hs | hs1: no draft, 1.00 a step, 21.1 ms a step; hs2: no draft, 1.00 a step, 21.2 ms a step |
| 16,384 | v4 | v41: no draft, 1.00 a step, 15.5 ms a step; v42: no draft, 1.00 a step, 15.4 ms a step |
|  | v4gf | gf1: no draft, 1.00 a step, 15.4 ms a step; gf2: no draft, 1.00 a step, 15.4 ms a step |
|  | v4hs | hs1: no draft, 1.00 a step, 15.3 ms a step; hs2: no draft, 1.00 a step, 15.2 ms a step |
| 32,768 | v4 | v41: no draft, 1.00 a step, 38.6 ms a step; v42: no draft, 1.00 a step, 38.4 ms a step |
|  | v4gf | gf1: no draft, 1.00 a step, 38.5 ms a step; gf2: no draft, 1.00 a step, 38.4 ms a step |
|  | v4hs | hs1: no draft, 1.00 a step, 38.2 ms a step; hs2: no draft, 1.00 a step, 38.2 ms a step |
| 49,152 | v4 | v41: no draft, 1.00 a step, 43.3 ms a step; v42: no draft, 1.00 a step, 43.2 ms a step |
|  | v4gf | gf1: no draft, 1.00 a step, 43.3 ms a step; gf2: no draft, 1.00 a step, 43.3 ms a step |
|  | v4hs | hs1: no draft, 1.00 a step, 42.9 ms a step; hs2: no draft, 1.00 a step, 42.9 ms a step |

### The sampler pass

One request per depth and load with the server line's own sampler and a fixed seed: tok/s, the draft counters and the step.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | v4 | v41: 61.40 tok/s, no draft, 1.00 a step, 16.3 ms a step; v42: 54.08 tok/s, no draft, 1.00 a step, 18.5 ms a step |
|  | v4gf | gf1: 61.28 tok/s, no draft, 1.00 a step, 16.3 ms a step; gf2: 58.32 tok/s, no draft, 1.00 a step, 17.1 ms a step |
|  | v4hs | hs1: 61.41 tok/s, no draft, 1.00 a step, 16.3 ms a step; hs2: 60.17 tok/s, no draft, 1.00 a step, 16.6 ms a step |
| 4,096 | v4 | v41: 66.32 tok/s, no draft, 1.00 a step, 15.1 ms a step; v42: 66.37 tok/s, no draft, 1.00 a step, 15.1 ms a step |
|  | v4gf | gf1: 66.28 tok/s, no draft, 1.00 a step, 15.1 ms a step; gf2: 66.36 tok/s, no draft, 1.00 a step, 15.1 ms a step |
|  | v4hs | hs1: 67.20 tok/s, no draft, 1.00 a step, 14.9 ms a step; hs2: 67.37 tok/s, no draft, 1.00 a step, 14.8 ms a step |
| 8,192 | v4 | v41: 43.22 tok/s, no draft, 1.00 a step, 23.1 ms a step; v42: 42.85 tok/s, no draft, 1.00 a step, 23.3 ms a step |
|  | v4gf | gf1: 42.43 tok/s, no draft, 1.00 a step, 23.6 ms a step; gf2: 43.68 tok/s, no draft, 1.00 a step, 22.9 ms a step |
|  | v4hs | hs1: 43.61 tok/s, no draft, 1.00 a step, 22.9 ms a step; hs2: 43.20 tok/s, no draft, 1.00 a step, 23.1 ms a step |
| 16,384 | v4 | v41: 35.77 tok/s, no draft, 1.00 a step, 28.0 ms a step; v42: 35.84 tok/s, no draft, 1.00 a step, 27.9 ms a step |
|  | v4gf | gf1: 35.82 tok/s, no draft, 1.00 a step, 27.9 ms a step; gf2: 35.89 tok/s, no draft, 1.00 a step, 27.9 ms a step |
|  | v4hs | hs1: 36.30 tok/s, no draft, 1.00 a step, 27.5 ms a step; hs2: 36.35 tok/s, no draft, 1.00 a step, 27.5 ms a step |
| 32,768 | v4 | v41: 26.76 tok/s, no draft, 1.00 a step, 37.4 ms a step; v42: 26.83 tok/s, no draft, 1.00 a step, 37.3 ms a step |
|  | v4gf | gf1: 26.83 tok/s, no draft, 1.00 a step, 37.3 ms a step; gf2: 26.85 tok/s, no draft, 1.00 a step, 37.2 ms a step |
|  | v4hs | hs1: 27.08 tok/s, no draft, 1.00 a step, 36.9 ms a step; hs2: 27.11 tok/s, no draft, 1.00 a step, 36.9 ms a step |
| 49,152 | v4 | v41: 22.64 tok/s, no draft, 1.00 a step, 44.2 ms a step; v42: 22.67 tok/s, no draft, 1.00 a step, 44.1 ms a step |
|  | v4gf | gf1: 22.67 tok/s, no draft, 1.00 a step, 44.1 ms a step; gf2: 22.67 tok/s, no draft, 1.00 a step, 44.1 ms a step |
|  | v4hs | hs1: 22.85 tok/s, no draft, 1.00 a step, 43.8 ms a step; hs2: 22.85 tok/s, no draft, 1.00 a step, 43.8 ms a step |

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
- Graph reallocations of the target, requests with one per load: v41: 0 of 24 requests; gf1: 0 of 24 requests; hs1: 0 of 24 requests; v42: 0 of 24 requests; gf2: 0 of 24 requests; hs2: 0 of 24 requests.
