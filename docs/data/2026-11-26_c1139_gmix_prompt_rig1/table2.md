## The deep lane: gg, pg, gu, pu

Model file `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-pmix-fast-00001-of-00003.gguf`, llama-server version: 1.0.0-dev (build 13244, commit 883f21f2c5).

256 tokens per request at temperature 0, each request twice per load (pass 1 is the first request at a new depth, pass 2 the same request again). Timed loads, in their order: gg2, pg2, gu2, pu2.

| Arm | Loads | Draft | Context | Its own variables and flags |
|---|---|---|---:|---|
| gg | gg2 | on | 65536 | `GGML_CUDA_PROFILE=<OUT>/gp-gg.csv GGML_CUDA_PROFILE_MODE=group GGML_CUDA_MTOK_IQ_FAST=2`, `-m <MODELS>/swift_q2_0_gmix_q2/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-pmix-fast-gmix-q2-00001-of-00003.gguf` |
| pg | pg2 | on | 65536 | `GGML_CUDA_PROFILE=<OUT>/gp-pg.csv GGML_CUDA_PROFILE_MODE=group` |
| gu | gu2 | on | 65536 | `GGML_CUDA_MTOK_IQ_FAST=2`, `-m <MODELS>/swift_q2_0_gmix_q2/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-pmix-fast-gmix-q2-00001-of-00003.gguf` |
| pu (control) | pu2 | on | 65536 | — |

### Generation, tok/s

Per load: pass 1, pass 2 → the load, with the load's time per token. A load's speed is its two passes' tokens over their time. Spread: the difference between the arm's fastest and slowest load. Class against `pu`: gain when the mean is above the control's by more than the larger of the two spreads, loss when below by more than it, else negligible.

| Depth (tokens) | Arm | Per load | Spread | Against `pu` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | gg | gg2: 17.08, 18.76 → **17.88** (55.92 ms a token) | — | -11.61 tok/s | -39.4 % | no class (one load) |
|  | pg | pg2: 21.59, 23.41 → **22.47** (44.51 ms a token) | — | -7.02 tok/s | -23.8 % | no class (one load) |
|  | gu | gu2: 23.48, 24.58 → **24.02** (41.64 ms a token) | — | -5.47 tok/s | -18.6 % | no class (one load) |
|  | pu | pu2: 28.07, 31.07 → **29.49** (33.91 ms a token) | — | control | | |
| 8,192 | gg | gg2: 17.78, 16.72 → **17.23** (58.03 ms a token) | — | -9.67 tok/s | -35.9 % | no class (one load) |
|  | pg | pg2: 18.44, 21.72 → **19.94** (50.14 ms a token) | — | -6.95 tok/s | -25.9 % | no class (one load) |
|  | gu | gu2: 22.08, 23.00 → **22.53** (44.39 ms a token) | — | -4.37 tok/s | -16.2 % | no class (one load) |
|  | pu | pu2: 25.95, 27.92 → **26.90** (37.18 ms a token) | — | control | | |
| 8,704 | gg | gg2: 19.00, 19.16 → **19.08** (52.42 ms a token) | — | -8.46 tok/s | -30.7 % | no class (one load) |
|  | pg | pg2: 21.53, 21.43 → **21.48** (46.55 ms a token) | — | -6.06 tok/s | -22.0 % | no class (one load) |
|  | gu | gu2: 23.34, 24.58 → **23.95** (41.76 ms a token) | — | -3.59 tok/s | -13.0 % | no class (one load) |
|  | pu | pu2: 27.16, 27.93 → **27.54** (36.31 ms a token) | — | control | | |

### Generation over all depths, tok/s

Every depth's tokens over their time, per load.

| Depth (tokens) | Arm | Per load | Spread | Against `pu` | | Class |
|---:|---|---|---:|---:|---:|---|
| all | gg | gg2: **18.03** | — | -9.90 tok/s | -35.4 % | no class (one load) |
|  | pg | pg2: **21.25** | — | -6.69 tok/s | -23.9 % | no class (one load) |
|  | gu | gu2: **23.48** | — | -4.46 tok/s | -16.0 % | no class (one load) |
|  | pu | pu2: **27.93** | — | control | | |

### The step on pass 1 (the first request at a new depth), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `pu`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `pu` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | gg | gg2: 142.77 (105 steps) | — | +40.29 ms | +39.3 % | no class (one load) |
|  | pg | pg2: 133.21 (89 steps) | — | +30.73 ms | +30.0 % | no class (one load) |
|  | gu | gu2: 103.84 (105 steps) | — | +1.36 ms | +1.3 % | no class (one load) |
|  | pu | pu2: 102.48 (89 steps) | — | control | | |
| 8,192 | gg | gg2: 148.47 (97 steps) | — | +35.05 ms | +30.9 % | no class (one load) |
|  | pg | pg2: 159.60 (87 steps) | — | +46.19 ms | +40.7 % | no class (one load) |
|  | gu | gu2: 119.55 (97 steps) | — | +6.14 ms | +5.4 % | no class (one load) |
|  | pu | pu2: 113.41 (87 steps) | — | control | | |
| 8,704 | gg | gg2: 148.06 (91 steps) | — | +37.15 ms | +33.5 % | no class (one load) |
|  | pg | pg2: 139.89 (85 steps) | — | +28.98 ms | +26.1 % | no class (one load) |
|  | gu | gu2: 120.53 (91 steps) | — | +9.62 ms | +8.7 % | no class (one load) |
|  | pu | pu2: 110.91 (85 steps) | — | control | | |

### The step on pass 2 (the same request again), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `pu`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `pu` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | gg | gg2: 131.18 (104 steps) | — | +37.54 ms | +40.1 % | no class (one load) |
|  | pg | pg2: 124.24 (88 steps) | — | +30.60 ms | +32.7 % | no class (one load) |
|  | gu | gu2: 100.15 (104 steps) | — | +6.51 ms | +6.9 % | no class (one load) |
|  | pu | pu2: 93.64 (88 steps) | — | control | | |
| 8,192 | gg | gg2: 157.84 (97 steps) | — | +48.69 ms | +44.6 % | no class (one load) |
|  | pg | pg2: 140.34 (84 steps) | — | +31.19 ms | +28.6 % | no class (one load) |
|  | gu | gu2: 114.76 (97 steps) | — | +5.61 ms | +5.1 % | no class (one load) |
|  | pu | pu2: 109.15 (84 steps) | — | control | | |
| 8,704 | gg | gg2: 146.86 (91 steps) | — | +39.03 ms | +36.2 % | no class (one load) |
|  | pg | pg2: 140.52 (85 steps) | — | +32.69 ms | +30.3 % | no class (one load) |
|  | gu | gu2: 114.44 (91 steps) | — | +6.61 ms | +6.1 % | no class (one load) |
|  | pu | pu2: 107.83 (85 steps) | — | control | | |

### Prompt reading, tok/s

Per load: the tokens read at that depth over their time (a prime request reads a chunk on top of the cache; without primes each decode request reads its prompt), with the tokens read and the tokens reused. Requests that read under 32 tokens are left out.

| Depth (tokens) | Arm | Per load | Spread | Against `pu` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | gg | gg2: 46.2 (64 read on 0 cached) | — | -9.0 tok/s | -16.3 % | no class (one load) |
|  | pg | pg2: 38.2 (64 read on 0 cached) | — | -17.0 tok/s | -30.8 % | no class (one load) |
|  | gu | gu2: 49.1 (64 read on 0 cached) | — | -6.0 tok/s | -11.0 % | no class (one load) |
|  | pu | pu2: 55.1 (64 read on 0 cached) | — | control | | |
| 8,192 | gg | gg2: 76.8 (8128 read on 0 cached) | — | -145.7 tok/s | -65.5 % | no class (one load) |
|  | pg | pg2: 81.0 (8128 read on 0 cached) | — | -141.5 tok/s | -63.6 % | no class (one load) |
|  | gu | gu2: 144.5 (8128 read on 0 cached) | — | -78.0 tok/s | -35.1 % | no class (one load) |
|  | pu | pu2: 222.5 (8128 read on 0 cached) | — | control | | |
| 8,704 | gg | gg2: 38.6 (1028 read on 7612 cached) | — | -116.0 tok/s | -75.0 % | no class (one load) |
|  | pg | pg2: 41.3 (1028 read on 7612 cached) | — | -113.2 tok/s | -73.3 % | no class (one load) |
|  | gu | gu2: 94.3 (1028 read on 7612 cached) | — | -60.2 tok/s | -38.9 % | no class (one load) |
|  | pu | pu2: 154.5 (1028 read on 7612 cached) | — | control | | |

### The draft and the step

Per load, pass 1: draft tokens accepted of drafted (the share), tokens a step (generated tokens over steps, a step being one pass of the main model: generated minus accepted), and the time of a step. Pass 2 is listed where its counters differ.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | gg | gg2: 151 of 312 (0.484), 2.44 a step, 142.8 ms a step (pass 2: 152 of 309 (0.492), 2.46 a step, 131.2 ms a step) |
|  | pg | pg2: 167 of 261 (0.640), 2.88 a step, 133.2 ms a step (pass 2: 168 of 258 (0.651), 2.91 a step, 124.2 ms a step) |
|  | gu | gu2: 151 of 312 (0.484), 2.44 a step, 103.8 ms a step (pass 2: 152 of 309 (0.492), 2.46 a step, 100.2 ms a step) |
|  | pu | pu2: 167 of 261 (0.640), 2.88 a step, 102.5 ms a step (pass 2: 168 of 258 (0.651), 2.91 a step, 93.6 ms a step) |
| 8,192 | gg | gg2: 159 of 286 (0.556), 2.64 a step, 148.5 ms a step |
|  | pg | pg2: 169 of 257 (0.658), 2.94 a step, 159.6 ms a step (pass 2: 172 of 248 (0.694), 3.05 a step, 140.3 ms a step) |
|  | gu | gu2: 159 of 286 (0.556), 2.64 a step, 119.5 ms a step |
|  | pu | pu2: 169 of 257 (0.658), 2.94 a step, 113.4 ms a step (pass 2: 172 of 248 (0.694), 3.05 a step, 109.1 ms a step) |
| 8,704 | gg | gg2: 165 of 267 (0.618), 2.81 a step, 148.1 ms a step |
|  | pg | pg2: 171 of 249 (0.687), 3.01 a step, 139.9 ms a step |
|  | gu | gu2: 165 of 267 (0.618), 2.81 a step, 120.5 ms a step |
|  | pu | pu2: 171 of 249 (0.687), 3.01 a step, 110.9 ms a step |

### The sampler pass

One request per depth and load with the server line's own sampler and a fixed seed: tok/s, the draft counters and the step.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | gg | gg2: 15.88 tok/s, 137 of 354 (0.387), 2.15 a step, 135.4 ms a step |
|  | pg | pg2: 23.68 tok/s, 171 of 251 (0.681), 3.01 a step, 127.2 ms a step |
|  | gu | gu2: 20.54 tok/s, 137 of 354 (0.387), 2.15 a step, 104.7 ms a step |
|  | pu | pu2: 31.36 tok/s, 171 of 251 (0.681), 3.01 a step, 96.0 ms a step |
| 8,192 | gg | gg2: 18.52 tok/s, 170 of 252 (0.675), 2.98 a step, 160.8 ms a step |
|  | pg | pg2: 20.06 tok/s, 168 of 258 (0.651), 2.91 a step, 145.0 ms a step |
|  | gu | gu2: 25.16 tok/s, 170 of 252 (0.675), 2.98 a step, 118.3 ms a step |
|  | pu | pu2: 25.96 tok/s, 168 of 258 (0.651), 2.91 a step, 112.1 ms a step |
| 8,704 | gg | gg2: 20.55 tok/s, 172 of 247 (0.696), 3.05 a step, 148.3 ms a step |
|  | pg | pg2: 19.68 tok/s, 171 of 252 (0.679), 3.01 a step, 153.1 ms a step |
|  | gu | gu2: 26.08 tok/s, 172 of 247 (0.696), 3.05 a step, 116.9 ms a step |
|  | pu | pu2: 27.18 tok/s, 171 of 252 (0.679), 3.01 a step, 110.8 ms a step |

### Checks

- Times: every load started on a machine checked quiet (valid=1).
- Prompts: 64 ids at depth 0, 8192 ids at depth 8,192, 8704 ids at depth 8,704; the same ids on every load and pass.
- Tokens out: 256 on every timed request.
- Greedy reply at depth 0: **differs from `pu` in gg, gu.**
- Greedy reply at depth 8,192: **differs from `pu` in gg, gu.**
- Greedy reply at depth 8,704: **differs from `pu` in gg, gu.**
- Primes (read on cached, the cache's source): 8,192: 8128 on 0 (none); 8,704: 1028 on 7612 (checkpoint).
- Graph reallocations of the target, requests with one per load: gg2: 0 of 12 requests; pg2: 0 of 12 requests; gu2: 0 of 12 requests; pu2: 0 of 12 requests.
