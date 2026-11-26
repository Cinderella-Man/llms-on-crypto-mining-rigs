## The deep lane: g0, g1, g2, u0, u1, u2

Model file `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-pmix-fast-gmix-q2-00001-of-00003.gguf`, llama-server version: 1.0.0-dev (build 13244, commit 883f21f2c5).

256 tokens per request at temperature 0, each request twice per load (pass 1 is the first request at a new depth, pass 2 the same request again). Timed loads, in their order: g0, g1, g2, u0, u1, u2, u0b.

| Arm | Loads | Draft | Context | Its own variables and flags |
|---|---|---|---:|---|
| g0 | g0 | on | 65536 | `GGML_CUDA_PROFILE=<OUT>/gp-g0.csv GGML_CUDA_PROFILE_MODE=group` |
| g1 | g1 | on | 65536 | `GGML_CUDA_PROFILE=<OUT>/gp-g1.csv GGML_CUDA_PROFILE_MODE=group GGML_CUDA_MTOK_IQ_FAST=1` |
| g2 | g2 | on | 65536 | `GGML_CUDA_PROFILE=<OUT>/gp-g2.csv GGML_CUDA_PROFILE_MODE=group GGML_CUDA_MTOK_IQ_FAST=2` |
| u0 (control) | u0, u0b | on | 65536 | — |
| u1 | u1 | on | 65536 | `GGML_CUDA_MTOK_IQ_FAST=1` |
| u2 | u2 | on | 65536 | `GGML_CUDA_MTOK_IQ_FAST=2` |

### Generation, tok/s

Per load: pass 1, pass 2 → the load, with the load's time per token. A load's speed is its two passes' tokens over their time. Spread: the difference between the arm's fastest and slowest load. Class against `u0`: gain when the mean is above the control's by more than the larger of the two spreads, loss when below by more than it, else negligible.

| Depth (tokens) | Arm | Per load | Spread | Against `u0` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | g0 | g0: 15.78, 17.24 → **16.48** (60.69 ms a token) | — | -5.14 tok/s | -23.8 % | no class (one load) |
|  | g1 | g1: 17.88, 18.49 → **18.18** (55.00 ms a token) | — | -3.44 tok/s | -15.9 % | no class (one load) |
|  | g2 | g2: 18.25, 18.84 → **18.54** (53.94 ms a token) | — | -3.08 tok/s | -14.2 % | no class (one load) |
|  | u0 | u0: 21.16, 22.11 → **21.63** (46.24 ms a token); u0b: 21.15, 22.10 → **21.62** (46.26 ms a token) | 0.01 | control | | |
|  | u1 | u1: 23.06, 24.15 → **23.59** (42.39 ms a token) | — | +1.97 tok/s | +9.1 % | no class (one load) |
|  | u2 | u2: 23.66, 24.71 → **24.17** (41.37 ms a token) | — | +2.55 tok/s | +11.8 % | no class (one load) |
| 8,192 | g0 | g0: 16.40, 16.84 → **16.62** (60.18 ms a token) | — | -3.43 tok/s | -17.1 % | no class (one load) |
|  | g1 | g1: 17.55, 17.19 → **17.37** (57.57 ms a token) | — | -2.67 tok/s | -13.3 % | no class (one load) |
|  | g2 | g2: 17.82, 18.11 → **17.97** (55.66 ms a token) | — | -2.08 tok/s | -10.4 % | no class (one load) |
|  | u0 | u0: 18.23, 20.97 → **19.50** (51.28 ms a token); u0b: 20.24, 20.95 → **20.59** (48.57 ms a token) | 1.09 | control | | |
|  | u1 | u1: 21.79, 22.73 → **22.25** (44.95 ms a token) | — | +2.20 tok/s | +11.0 % | no class (one load) |
|  | u2 | u2: 22.19, 23.15 → **22.66** (44.13 ms a token) | — | +2.62 tok/s | +13.0 % | no class (one load) |

### Generation over all depths, tok/s

Every depth's tokens over their time, per load.

| Depth (tokens) | Arm | Per load | Spread | Against `u0` | | Class |
|---:|---|---|---:|---:|---:|---|
| all | g0 | g0: **16.55** | — | -4.25 tok/s | -20.4 % | no class (one load) |
|  | g1 | g1: **17.77** | — | -3.03 tok/s | -14.6 % | no class (one load) |
|  | g2 | g2: **18.25** | — | -2.55 tok/s | -12.3 % | no class (one load) |
|  | u0 | u0: **20.51**; u0b: **21.09** | 0.58 | control | | |
|  | u1 | u1: **22.90** | — | +2.10 tok/s | +10.1 % | no class (one load) |
|  | u2 | u2: **23.39** | — | +2.59 tok/s | +12.5 % | no class (one load) |

### The step on pass 1 (the first request at a new depth), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `u0`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `u0` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | g0 | g0: 154.48 (105 steps) | — | +39.23 ms | +34.0 % | no class (one load) |
|  | g1 | g1: 136.34 (105 steps) | — | +21.09 ms | +18.3 % | no class (one load) |
|  | g2 | g2: 133.62 (105 steps) | — | +18.37 ms | +15.9 % | no class (one load) |
|  | u0 | u0: 115.22 (105 steps); u0b: 115.27 (105 steps) | 0.05 | control | | |
|  | u1 | u1: 105.74 (105 steps) | — | -9.51 ms | -8.2 % | no class (one load) |
|  | u2 | u2: 103.05 (105 steps) | — | -12.20 ms | -10.6 % | no class (one load) |
| 8,192 | g0 | g0: 160.90 (97 steps) | — | +23.28 ms | +16.9 % | no class (one load) |
|  | g1 | g1: 150.35 (97 steps) | — | +12.73 ms | +9.3 % | no class (one load) |
|  | g2 | g2: 148.08 (97 steps) | — | +10.47 ms | +7.6 % | no class (one load) |
|  | u0 | u0: 144.80 (97 steps); u0b: 130.42 (97 steps) | 14.38 | control | | |
|  | u1 | u1: 121.13 (97 steps) | — | -16.49 ms | -12.0 % | no class (one load) |
|  | u2 | u2: 118.93 (97 steps) | — | -18.68 ms | -13.6 % | no class (one load) |

### The step on pass 2 (the same request again), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `u0`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `u0` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | g0 | g0: 142.80 (104 steps) | — | +31.46 ms | +28.3 % | no class (one load) |
|  | g1 | g1: 133.11 (104 steps) | — | +21.77 ms | +19.6 % | no class (one load) |
|  | g2 | g2: 130.63 (104 steps) | — | +19.29 ms | +17.3 % | no class (one load) |
|  | u0 | u0: 111.31 (104 steps); u0b: 111.37 (104 steps) | 0.05 | control | | |
|  | u1 | u1: 101.91 (104 steps) | — | -9.43 ms | -8.5 % | no class (one load) |
|  | u2 | u2: 99.61 (104 steps) | — | -11.73 ms | -10.5 % | no class (one load) |
| 8,192 | g0 | g0: 156.77 (97 steps) | — | +30.85 ms | +24.5 % | no class (one load) |
|  | g1 | g1: 153.51 (97 steps) | — | +27.60 ms | +21.9 % | no class (one load) |
|  | g2 | g2: 145.73 (97 steps) | — | +19.82 ms | +15.7 % | no class (one load) |
|  | u0 | u0: 125.87 (97 steps); u0b: 125.95 (97 steps) | 0.08 | control | | |
|  | u1 | u1: 116.12 (97 steps) | — | -9.79 ms | -7.8 % | no class (one load) |
|  | u2 | u2: 114.01 (97 steps) | — | -11.90 ms | -9.5 % | no class (one load) |

### Prompt reading, tok/s

Per load: the tokens read at that depth over their time (a prime request reads a chunk on top of the cache; without primes each decode request reads its prompt), with the tokens read and the tokens reused. Requests that read under 32 tokens are left out.

| Depth (tokens) | Arm | Per load | Spread | Against `u0` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | g0 | g0: 45.7 (64 read on 0 cached) | — | -3.0 tok/s | -6.2 % | no class (one load) |
|  | g1 | g1: 45.7 (64 read on 0 cached) | — | -3.0 tok/s | -6.2 % | no class (one load) |
|  | g2 | g2: 46.4 (64 read on 0 cached) | — | -2.4 tok/s | -4.8 % | no class (one load) |
|  | u0 | u0: 48.8 (64 read on 0 cached); u0b: 48.7 (64 read on 0 cached) | 0.2 | control | | |
|  | u1 | u1: 49.7 (64 read on 0 cached) | — | +1.0 tok/s | +2.0 % | no class (one load) |
|  | u2 | u2: 48.8 (64 read on 0 cached) | — | +0.0 tok/s | +0.1 % | no class (one load) |
| 8,192 | g0 | g0: 75.7 (8128 read on 0 cached) | — | -70.5 tok/s | -48.2 % | no class (one load) |
|  | g1 | g1: 77.0 (8128 read on 0 cached) | — | -69.1 tok/s | -47.3 % | no class (one load) |
|  | g2 | g2: 76.2 (8128 read on 0 cached) | — | -69.9 tok/s | -47.8 % | no class (one load) |
|  | u0 | u0: 143.3 (8128 read on 0 cached); u0b: 148.9 (8128 read on 0 cached) | 5.6 | control | | |
|  | u1 | u1: 139.5 (8128 read on 0 cached) | — | -6.6 tok/s | -4.5 % | no class (one load) |
|  | u2 | u2: 146.7 (8128 read on 0 cached) | — | +0.6 tok/s | +0.4 % | no class (one load) |

### The draft and the step

Per load, pass 1: draft tokens accepted of drafted (the share), tokens a step (generated tokens over steps, a step being one pass of the main model: generated minus accepted), and the time of a step. Pass 2 is listed where its counters differ.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | g0 | g0: 151 of 312 (0.484), 2.44 a step, 154.5 ms a step (pass 2: 152 of 309 (0.492), 2.46 a step, 142.8 ms a step) |
|  | g1 | g1: 151 of 312 (0.484), 2.44 a step, 136.3 ms a step (pass 2: 152 of 309 (0.492), 2.46 a step, 133.1 ms a step) |
|  | g2 | g2: 151 of 312 (0.484), 2.44 a step, 133.6 ms a step (pass 2: 152 of 309 (0.492), 2.46 a step, 130.6 ms a step) |
|  | u0 | u0: 151 of 312 (0.484), 2.44 a step, 115.2 ms a step (pass 2: 152 of 309 (0.492), 2.46 a step, 111.3 ms a step); u0b: 151 of 312 (0.484), 2.44 a step, 115.3 ms a step (pass 2: 152 of 309 (0.492), 2.46 a step, 111.4 ms a step) |
|  | u1 | u1: 151 of 312 (0.484), 2.44 a step, 105.7 ms a step (pass 2: 152 of 309 (0.492), 2.46 a step, 101.9 ms a step) |
|  | u2 | u2: 151 of 312 (0.484), 2.44 a step, 103.0 ms a step (pass 2: 152 of 309 (0.492), 2.46 a step, 99.6 ms a step) |
| 8,192 | g0 | g0: 159 of 286 (0.556), 2.64 a step, 160.9 ms a step |
|  | g1 | g1: 159 of 286 (0.556), 2.64 a step, 150.3 ms a step |
|  | g2 | g2: 159 of 286 (0.556), 2.64 a step, 148.1 ms a step |
|  | u0 | u0: 159 of 286 (0.556), 2.64 a step, 144.8 ms a step; u0b: 159 of 286 (0.556), 2.64 a step, 130.4 ms a step |
|  | u1 | u1: 159 of 286 (0.556), 2.64 a step, 121.1 ms a step |
|  | u2 | u2: 159 of 286 (0.556), 2.64 a step, 118.9 ms a step |

### The sampler pass

One request per depth and load with the server line's own sampler and a fixed seed: tok/s, the draft counters and the step.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | g0 | g0: 13.39 tok/s, 137 of 354 (0.387), 2.15 a step, 160.6 ms a step |
|  | g1 | g1: 15.69 tok/s, 137 of 354 (0.387), 2.15 a step, 137.1 ms a step |
|  | g2 | g2: 15.94 tok/s, 137 of 354 (0.387), 2.15 a step, 135.0 ms a step |
|  | u0 | u0: 18.56 tok/s, 137 of 354 (0.387), 2.15 a step, 115.9 ms a step; u0b: 17.48 tok/s, 137 of 354 (0.387), 2.15 a step, 123.1 ms a step |
|  | u1 | u1: 20.22 tok/s, 137 of 354 (0.387), 2.15 a step, 106.4 ms a step |
|  | u2 | u2: 20.71 tok/s, 137 of 354 (0.387), 2.15 a step, 103.9 ms a step |
| 8,192 | g0 | g0: 17.09 tok/s, 170 of 252 (0.675), 2.98 a step, 174.2 ms a step |
|  | g1 | g1: 18.82 tok/s, 170 of 252 (0.675), 2.98 a step, 158.1 ms a step |
|  | g2 | g2: 18.36 tok/s, 170 of 252 (0.675), 2.98 a step, 162.1 ms a step |
|  | u0 | u0: 23.02 tok/s, 170 of 252 (0.675), 2.98 a step, 129.3 ms a step; u0b: 22.95 tok/s, 170 of 252 (0.675), 2.98 a step, 129.7 ms a step |
|  | u1 | u1: 24.78 tok/s, 170 of 252 (0.675), 2.98 a step, 120.1 ms a step |
|  | u2 | u2: 25.16 tok/s, 170 of 252 (0.675), 2.98 a step, 118.3 ms a step |

### Checks

- Times: every load started on a machine checked quiet (valid=1).
- Prompts: 64 ids at depth 0, 8192 ids at depth 8,192; the same ids on every load and pass.
- Tokens out: 256 on every timed request.
- Greedy reply at depth 0: the same on every pass and load of every arm.
- Greedy reply at depth 8,192: the same on every pass and load of every arm.
- Primes (read on cached, the cache's source): 8,192: 8128 on 0 (none).
- Graph reallocations of the target, requests with one per load: g0: 0 of 8 requests; g1: 0 of 8 requests; g2: 0 of 8 requests; u0: 0 of 8 requests; u1: 0 of 8 requests; u2: 0 of 8 requests; u0b: 0 of 8 requests.
