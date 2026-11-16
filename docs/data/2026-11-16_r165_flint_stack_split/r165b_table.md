## The deep lane: v4, v4ka, v4ck, v4tp

Model file `flint-near-q8-p065.gguf`, llama-server version: 1.0.0-dev (build 12832, commit eb312b7bce).

256 tokens per request at temperature 0, each request twice per load (pass 1 is the first request at a new depth, pass 2 the same request again). Timed loads, in their order: v41, ka1, ck1, tp1, v42, ka2, ck2, tp2.

| Arm | Loads | Draft | Context | Its own variables and flags |
|---|---|---|---:|---|
| v4 (control) | v41, v42 | off | 262144 | — |
| v4ka | ka1, ka2 | off | 262144 | `GGML_CUDA_KEEPALIVE=decode GGML_CUDA_KEEPALIVE_EAGER=1` |
| v4ck | ck1, ck2 | off | 262144 | `--ctx-checkpoints-vram 4` |
| v4tp | tp1, tp2 | off | 262144 | `--ubatch-taper short=600:64` |

### Generation, tok/s

Per load: pass 1, pass 2 → the load, with the load's time per token. A load's speed is its two passes' tokens over their time. Spread: the difference between the arm's fastest and slowest load. Class against `v4`: gain when the mean is above the control's by more than the larger of the two spreads, loss when below by more than it, else negligible.

| Depth (tokens) | Arm | Per load | Spread | Against `v4` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | v4 | v41: 47.02, 55.31 → **50.83** (19.67 ms a token); v42: 39.80, 55.45 → **46.34** (21.58 ms a token) | 4.49 | control | | |
|  | v4ka | ka1: 49.32, 54.51 → **51.78** (19.31 ms a token); ka2: 49.00, 54.55 → **51.63** (19.37 ms a token) | 0.16 | +3.12 tok/s | +6.4 % | negligible |
|  | v4ck | ck1: 47.64, 55.69 → **51.35** (19.47 ms a token); ck2: 47.20, 55.67 → **51.09** (19.57 ms a token) | 0.27 | +2.63 tok/s | +5.4 % | negligible |
|  | v4tp | tp1: 43.01, 55.72 → **48.54** (20.60 ms a token); tp2: 45.12, 55.32 → **49.70** (20.12 ms a token) | 1.16 | +0.54 tok/s | +1.1 % | negligible |
| 4,096 | v4 | v41: 74.77, 80.48 → **77.52** (12.90 ms a token); v42: 75.07, 80.35 → **77.62** (12.88 ms a token) | 0.10 | control | | |
|  | v4ka | ka1: 71.83, 78.35 → **74.95** (13.34 ms a token); ka2: 71.51, 78.33 → **74.76** (13.38 ms a token) | 0.18 | -2.71 tok/s | -3.5 % | loss |
|  | v4ck | ck1: 75.02, 80.34 → **77.59** (12.89 ms a token); ck2: 75.05, 80.41 → **77.64** (12.88 ms a token) | 0.05 | +0.04 tok/s | +0.1 % | negligible |
|  | v4tp | tp1: 75.06, 80.36 → **77.62** (12.88 ms a token); tp2: 75.03, 80.36 → **77.60** (12.89 ms a token) | 0.02 | +0.04 tok/s | +0.1 % | negligible |
| 8,192 | v4 | v41: 46.77, 51.29 → **48.93** (20.44 ms a token); v42: 44.38, 51.23 → **47.56** (21.03 ms a token) | 1.37 | control | | |
|  | v4ka | ka1: 43.93, 49.73 → **46.65** (21.44 ms a token); ka2: 42.76, 49.71 → **45.98** (21.75 ms a token) | 0.67 | -1.93 tok/s | -4.0 % | loss |
|  | v4ck | ck1: 47.69, 51.26 → **49.41** (20.24 ms a token); ck2: 47.69, 51.23 → **49.40** (20.24 ms a token) | 0.02 | +1.16 tok/s | +2.4 % | negligible |
|  | v4tp | tp1: 46.95, 51.19 → **48.98** (20.42 ms a token); tp2: 46.93, 51.22 → **48.98** (20.42 ms a token) | 0.00 | +0.74 tok/s | +1.5 % | negligible |
| 16,384 | v4 | v41: 64.79, 70.28 → **67.43** (14.83 ms a token); v42: 64.67, 70.24 → **67.34** (14.85 ms a token) | 0.09 | control | | |
|  | v4ka | ka1: 62.07, 67.74 → **64.78** (15.44 ms a token); ka2: 61.84, 67.70 → **64.64** (15.47 ms a token) | 0.14 | -2.67 tok/s | -4.0 % | loss |
|  | v4ck | ck1: 64.89, 70.27 → **67.47** (14.82 ms a token); ck2: 64.84, 70.22 → **67.42** (14.83 ms a token) | 0.05 | +0.06 tok/s | +0.1 % | negligible |
|  | v4tp | tp1: 64.68, 70.24 → **67.35** (14.85 ms a token); tp2: 64.80, 70.18 → **67.38** (14.84 ms a token) | 0.04 | -0.02 tok/s | -0.0 % | negligible |
| 32,768 | v4 | v41: 26.04, 27.01 → **26.52** (37.71 ms a token); v42: 25.99, 27.00 → **26.48** (37.76 ms a token) | 0.04 | control | | |
|  | v4ka | ka1: 24.91, 25.85 → **25.37** (39.42 ms a token); ka2: 24.88, 25.83 → **25.34** (39.46 ms a token) | 0.02 | -1.14 tok/s | -4.3 % | loss |
|  | v4ck | ck1: 26.03, 27.02 → **26.52** (37.71 ms a token); ck2: 25.96, 26.99 → **26.47** (37.79 ms a token) | 0.05 | -0.01 tok/s | -0.0 % | negligible |
|  | v4tp | tp1: 26.03, 27.00 → **26.50** (37.73 ms a token); tp2: 25.97, 26.99 → **26.47** (37.78 ms a token) | 0.04 | -0.01 tok/s | -0.0 % | negligible |
| 49,152 | v4 | v41: 23.15, 24.00 → **23.57** (42.43 ms a token); v42: 23.15, 24.01 → **23.57** (42.42 ms a token) | 0.00 | control | | |
|  | v4ka | ka1: 21.99, 22.81 → **22.39** (44.66 ms a token); ka2: 22.01, 22.79 → **22.40** (44.65 ms a token) | 0.01 | -1.18 tok/s | -5.0 % | loss |
|  | v4ck | ck1: 23.16, 24.02 → **23.58** (42.40 ms a token); ck2: 23.17, 24.00 → **23.58** (42.41 ms a token) | 0.00 | +0.01 tok/s | +0.0 % | gain |
|  | v4tp | tp1: 23.11, 24.00 → **23.55** (42.47 ms a token); tp2: 23.15, 24.00 → **23.57** (42.43 ms a token) | 0.02 | -0.02 tok/s | -0.1 % | negligible |

### Generation over all depths, tok/s

Every depth's tokens over their time, per load.

| Depth (tokens) | Arm | Per load | Spread | Against `v4` | | Class |
|---:|---|---|---:|---:|---:|---|
| all | v4 | v41: **40.55**; v42: **39.86** | 0.69 | control | | |
|  | v4ka | ka1: **39.06**; ka2: **38.94** | 0.12 | -1.20 tok/s | -3.0 % | loss |
|  | v4ck | ck1: **40.67**; ck2: **40.62** | 0.05 | +0.44 tok/s | +1.1 % | negligible |
|  | v4tp | tp1: **40.28**; tp2: **40.41** | 0.13 | +0.14 tok/s | +0.4 % | negligible |

### Prompt reading, tok/s

Per load: the tokens read at that depth over their time (a prime request reads a chunk on top of the cache; without primes each decode request reads its prompt), with the tokens read and the tokens reused. Requests that read under 32 tokens are left out.

| Depth (tokens) | Arm | Per load | Spread | Against `v4` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | v4 | v41: 85.3 (64 read on 0 cached); v42: 73.2 (64 read on 0 cached) | 12.1 | control | | |
|  | v4ka | ka1: 96.1 (64 read on 0 cached); ka2: 105.7 (64 read on 0 cached) | 9.5 | +21.6 tok/s | +27.3 % | gain |
|  | v4ck | ck1: 90.2 (64 read on 0 cached); ck2: 92.1 (64 read on 0 cached) | 2.0 | +11.9 tok/s | +15.0 % | negligible |
|  | v4tp | tp1: 85.7 (64 read on 0 cached); tp2: 86.4 (64 read on 0 cached) | 0.7 | +6.8 tok/s | +8.6 % | negligible |
| 4,096 | v4 | v41: 519.7 (4032 read on 0 cached); v42: 520.2 (4032 read on 0 cached) | 0.5 | control | | |
|  | v4ka | ka1: 445.7 (4032 read on 0 cached); ka2: 449.7 (4032 read on 0 cached) | 3.9 | -72.3 tok/s | -13.9 % | loss |
|  | v4ck | ck1: 440.8 (4032 read on 0 cached); ck2: 544.1 (4032 read on 0 cached) | 103.3 | -27.6 tok/s | -5.3 % | negligible |
|  | v4tp | tp1: 502.2 (4032 read on 0 cached); tp2: 523.4 (4032 read on 0 cached) | 21.2 | -7.2 tok/s | -1.4 % | negligible |
| 8,192 | v4 | v41: 448.4 (4100 read on 4028 cached); v42: 425.7 (4100 read on 4028 cached) | 22.7 | control | | |
|  | v4ka | ka1: 384.4 (4100 read on 4028 cached); ka2: 404.8 (4100 read on 4028 cached) | 20.4 | -42.4 tok/s | -9.7 % | loss |
|  | v4ck | ck1: 474.9 (4100 read on 4028 cached); ck2: 516.3 (4100 read on 4028 cached) | 41.4 | +58.6 tok/s | +13.4 % | gain |
|  | v4tp | tp1: 473.4 (4100 read on 4028 cached); tp2: 478.9 (4100 read on 4028 cached) | 5.5 | +39.1 tok/s | +9.0 % | gain |
| 16,384 | v4 | v41: 488.4 (12292 read on 4028 cached); v42: 479.7 (12292 read on 4028 cached) | 8.7 | control | | |
|  | v4ka | ka1: 436.4 (12292 read on 4028 cached); ka2: 423.7 (12292 read on 4028 cached) | 12.7 | -54.0 tok/s | -11.2 % | loss |
|  | v4ck | ck1: 498.4 (12292 read on 4028 cached); ck2: 507.6 (12292 read on 4028 cached) | 9.2 | +19.0 tok/s | +3.9 % | gain |
|  | v4tp | tp1: 493.0 (12292 read on 4028 cached); tp2: 496.0 (12292 read on 4028 cached) | 3.1 | +10.5 tok/s | +2.2 % | gain |
| 32,768 | v4 | v41: 338.5 (16388 read on 16316 cached); v42: 337.0 (16388 read on 16316 cached) | 1.4 | control | | |
|  | v4ka | ka1: 270.0 (16388 read on 16316 cached); ka2: 269.4 (16388 read on 16316 cached) | 0.6 | -68.0 tok/s | -20.1 % | loss |
|  | v4ck | ck1: 341.0 (16388 read on 16316 cached); ck2: 340.4 (16388 read on 16316 cached) | 0.5 | +3.0 tok/s | +0.9 % | gain |
|  | v4tp | tp1: 336.4 (16388 read on 16316 cached); tp2: 337.4 (16388 read on 16316 cached) | 1.0 | -0.9 tok/s | -0.3 % | negligible |
| 49,152 | v4 | v41: 218.0 (16388 read on 32700 cached); v42: 218.1 (16388 read on 32700 cached) | 0.1 | control | | |
|  | v4ka | ka1: 181.7 (16388 read on 32700 cached); ka2: 181.3 (16388 read on 32700 cached) | 0.4 | -36.5 tok/s | -16.8 % | loss |
|  | v4ck | ck1: 219.6 (16388 read on 32700 cached); ck2: 222.5 (16388 read on 32700 cached) | 2.9 | +3.0 tok/s | +1.4 % | gain |
|  | v4tp | tp1: 217.4 (16388 read on 32700 cached); tp2: 220.4 (16388 read on 32700 cached) | 3.0 | +0.9 tok/s | +0.4 % | negligible |

### The draft and the step

Per load, pass 1: draft tokens accepted of drafted (the share), tokens a step (generated tokens over steps, a step being one pass of the main model: generated minus accepted), and the time of a step. Pass 2 is listed where its counters differ.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | v4 | v41: no draft, 1.00 a step, 21.3 ms a step; v42: no draft, 1.00 a step, 25.1 ms a step |
|  | v4ka | ka1: no draft, 1.00 a step, 20.3 ms a step; ka2: no draft, 1.00 a step, 20.4 ms a step |
|  | v4ck | ck1: no draft, 1.00 a step, 21.0 ms a step; ck2: no draft, 1.00 a step, 21.2 ms a step |
|  | v4tp | tp1: no draft, 1.00 a step, 23.3 ms a step; tp2: no draft, 1.00 a step, 22.2 ms a step |
| 4,096 | v4 | v41: no draft, 1.00 a step, 13.4 ms a step; v42: no draft, 1.00 a step, 13.3 ms a step |
|  | v4ka | ka1: no draft, 1.00 a step, 13.9 ms a step; ka2: no draft, 1.00 a step, 14.0 ms a step |
|  | v4ck | ck1: no draft, 1.00 a step, 13.3 ms a step; ck2: no draft, 1.00 a step, 13.3 ms a step |
|  | v4tp | tp1: no draft, 1.00 a step, 13.3 ms a step; tp2: no draft, 1.00 a step, 13.3 ms a step |
| 8,192 | v4 | v41: no draft, 1.00 a step, 21.4 ms a step; v42: no draft, 1.00 a step, 22.5 ms a step |
|  | v4ka | ka1: no draft, 1.00 a step, 22.8 ms a step; ka2: no draft, 1.00 a step, 23.4 ms a step |
|  | v4ck | ck1: no draft, 1.00 a step, 21.0 ms a step; ck2: no draft, 1.00 a step, 21.0 ms a step |
|  | v4tp | tp1: no draft, 1.00 a step, 21.3 ms a step; tp2: no draft, 1.00 a step, 21.3 ms a step |
| 16,384 | v4 | v41: no draft, 1.00 a step, 15.4 ms a step; v42: no draft, 1.00 a step, 15.5 ms a step |
|  | v4ka | ka1: no draft, 1.00 a step, 16.1 ms a step; ka2: no draft, 1.00 a step, 16.2 ms a step |
|  | v4ck | ck1: no draft, 1.00 a step, 15.4 ms a step; ck2: no draft, 1.00 a step, 15.4 ms a step |
|  | v4tp | tp1: no draft, 1.00 a step, 15.5 ms a step; tp2: no draft, 1.00 a step, 15.4 ms a step |
| 32,768 | v4 | v41: no draft, 1.00 a step, 38.4 ms a step; v42: no draft, 1.00 a step, 38.5 ms a step |
|  | v4ka | ka1: no draft, 1.00 a step, 40.1 ms a step; ka2: no draft, 1.00 a step, 40.2 ms a step |
|  | v4ck | ck1: no draft, 1.00 a step, 38.4 ms a step; ck2: no draft, 1.00 a step, 38.5 ms a step |
|  | v4tp | tp1: no draft, 1.00 a step, 38.4 ms a step; tp2: no draft, 1.00 a step, 38.5 ms a step |
| 49,152 | v4 | v41: no draft, 1.00 a step, 43.2 ms a step; v42: no draft, 1.00 a step, 43.2 ms a step |
|  | v4ka | ka1: no draft, 1.00 a step, 45.5 ms a step; ka2: no draft, 1.00 a step, 45.4 ms a step |
|  | v4ck | ck1: no draft, 1.00 a step, 43.2 ms a step; ck2: no draft, 1.00 a step, 43.2 ms a step |
|  | v4tp | tp1: no draft, 1.00 a step, 43.3 ms a step; tp2: no draft, 1.00 a step, 43.2 ms a step |

### The sampler pass

One request per depth and load with the server line's own sampler and a fixed seed: tok/s, the draft counters and the step.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | v4 | v41: 61.19 tok/s, no draft, 1.00 a step, 16.3 ms a step; v42: 56.28 tok/s, no draft, 1.00 a step, 17.8 ms a step |
|  | v4ka | ka1: 54.40 tok/s, no draft, 1.00 a step, 18.4 ms a step; ka2: 52.29 tok/s, no draft, 1.00 a step, 19.1 ms a step |
|  | v4ck | ck1: 54.70 tok/s, no draft, 1.00 a step, 18.3 ms a step; ck2: 60.40 tok/s, no draft, 1.00 a step, 16.6 ms a step |
|  | v4tp | tp1: 57.63 tok/s, no draft, 1.00 a step, 17.4 ms a step; tp2: 61.62 tok/s, no draft, 1.00 a step, 16.2 ms a step |
| 4,096 | v4 | v41: 66.41 tok/s, no draft, 1.00 a step, 15.1 ms a step; v42: 66.30 tok/s, no draft, 1.00 a step, 15.1 ms a step |
|  | v4ka | ka1: 64.51 tok/s, no draft, 1.00 a step, 15.5 ms a step; ka2: 64.51 tok/s, no draft, 1.00 a step, 15.5 ms a step |
|  | v4ck | ck1: 66.39 tok/s, no draft, 1.00 a step, 15.1 ms a step; ck2: 66.37 tok/s, no draft, 1.00 a step, 15.1 ms a step |
|  | v4tp | tp1: 66.30 tok/s, no draft, 1.00 a step, 15.1 ms a step; tp2: 66.25 tok/s, no draft, 1.00 a step, 15.1 ms a step |
| 8,192 | v4 | v41: 40.21 tok/s, no draft, 1.00 a step, 24.9 ms a step; v42: 43.00 tok/s, no draft, 1.00 a step, 23.3 ms a step |
|  | v4ka | ka1: 41.95 tok/s, no draft, 1.00 a step, 23.8 ms a step; ka2: 42.12 tok/s, no draft, 1.00 a step, 23.7 ms a step |
|  | v4ck | ck1: 43.10 tok/s, no draft, 1.00 a step, 23.2 ms a step; ck2: 43.26 tok/s, no draft, 1.00 a step, 23.1 ms a step |
|  | v4tp | tp1: 43.21 tok/s, no draft, 1.00 a step, 23.1 ms a step; tp2: 43.22 tok/s, no draft, 1.00 a step, 23.1 ms a step |
| 16,384 | v4 | v41: 35.85 tok/s, no draft, 1.00 a step, 27.9 ms a step; v42: 35.82 tok/s, no draft, 1.00 a step, 27.9 ms a step |
|  | v4ka | ka1: 34.62 tok/s, no draft, 1.00 a step, 28.9 ms a step; ka2: 34.60 tok/s, no draft, 1.00 a step, 28.9 ms a step |
|  | v4ck | ck1: 35.86 tok/s, no draft, 1.00 a step, 27.9 ms a step; ck2: 35.85 tok/s, no draft, 1.00 a step, 27.9 ms a step |
|  | v4tp | tp1: 35.84 tok/s, no draft, 1.00 a step, 27.9 ms a step; tp2: 35.86 tok/s, no draft, 1.00 a step, 27.9 ms a step |
| 32,768 | v4 | v41: 26.86 tok/s, no draft, 1.00 a step, 37.2 ms a step; v42: 26.85 tok/s, no draft, 1.00 a step, 37.2 ms a step |
|  | v4ka | ka1: 25.71 tok/s, no draft, 1.00 a step, 38.9 ms a step; ka2: 25.68 tok/s, no draft, 1.00 a step, 38.9 ms a step |
|  | v4ck | ck1: 26.85 tok/s, no draft, 1.00 a step, 37.2 ms a step; ck2: 26.83 tok/s, no draft, 1.00 a step, 37.3 ms a step |
|  | v4tp | tp1: 26.85 tok/s, no draft, 1.00 a step, 37.2 ms a step; tp2: 26.84 tok/s, no draft, 1.00 a step, 37.3 ms a step |
| 49,152 | v4 | v41: 22.68 tok/s, no draft, 1.00 a step, 44.1 ms a step; v42: 22.69 tok/s, no draft, 1.00 a step, 44.1 ms a step |
|  | v4ka | ka1: 21.54 tok/s, no draft, 1.00 a step, 46.4 ms a step; ka2: 21.52 tok/s, no draft, 1.00 a step, 46.5 ms a step |
|  | v4ck | ck1: 22.70 tok/s, no draft, 1.00 a step, 44.0 ms a step; ck2: 22.69 tok/s, no draft, 1.00 a step, 44.1 ms a step |
|  | v4tp | tp1: 22.67 tok/s, no draft, 1.00 a step, 44.1 ms a step; tp2: 22.66 tok/s, no draft, 1.00 a step, 44.1 ms a step |

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
- Graph reallocations of the target, requests with one per load: v41: 0 of 24 requests; ka1: 0 of 24 requests; ck1: 0 of 24 requests; tp1: 0 of 24 requests; v42: 0 of 24 requests; ka2: 0 of 24 requests; ck2: 0 of 24 requests; tp2: 0 of 24 requests.
