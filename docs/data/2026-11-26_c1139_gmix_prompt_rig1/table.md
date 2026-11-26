## The deep lane: pg, gg, ph, gh, pu, gu

Model file `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-pmix-fast-00001-of-00003.gguf`, llama-server version: 1.0.0-dev (build 13244, commit 883f21f2c5).

256 tokens per request at temperature 0, each request twice per load (pass 1 is the first request at a new depth, pass 2 the same request again). Timed loads, in their order: pg, gg, ph, gh, pu1, gu1.

| Arm | Loads | Draft | Context | Its own variables and flags |
|---|---|---|---:|---|
| pg | pg | on | 65536 | `GGML_CUDA_PROFILE=<OUT>/gp-pg.csv GGML_CUDA_PROFILE_MODE=group` |
| gg | gg | on | 65536 | `GGML_CUDA_PROFILE=<OUT>/gp-gg.csv GGML_CUDA_PROFILE_MODE=group GGML_CUDA_MTOK_IQ_FAST=2`, `-m <MODELS>/swift_q2_0_gmix_q2/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-pmix-fast-gmix-q2-00001-of-00003.gguf` |
| ph | ph | on | 65536 | `GGML_CUDA_PROFILE_HOST=<OUT>/hp-ph.csv LLAMA_STEP_TRACE=<OUT>/st-ph.jsonl` |
| gh | gh | on | 65536 | `GGML_CUDA_PROFILE_HOST=<OUT>/hp-gh.csv LLAMA_STEP_TRACE=<OUT>/st-gh.jsonl GGML_CUDA_MTOK_IQ_FAST=2`, `-m <MODELS>/swift_q2_0_gmix_q2/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-pmix-fast-gmix-q2-00001-of-00003.gguf` |
| pu (control) | pu1 | on | 65536 | — |
| gu | gu1 | on | 65536 | `GGML_CUDA_MTOK_IQ_FAST=2`, `-m <MODELS>/swift_q2_0_gmix_q2/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-pmix-fast-gmix-q2-00001-of-00003.gguf` |

### Generation, tok/s

Per load: pass 1, pass 2 → the load, with the load's time per token. A load's speed is its two passes' tokens over their time. Spread: the difference between the arm's fastest and slowest load. Class against `pu`: gain when the mean is above the control's by more than the larger of the two spreads, loss when below by more than it, else negligible.

| Depth (tokens) | Arm | Per load | Spread | Against `pu` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | pg | pg: 21.88, 23.44 → **22.63** (44.19 ms a token) | — | -6.90 tok/s | -23.4 % | no class (one load) |
|  | gg | gg: 18.22, 18.74 → **18.47** (54.13 ms a token) | — | -11.06 tok/s | -37.4 % | no class (one load) |
|  | ph | ph: 27.83, 30.95 → **29.30** (34.12 ms a token) | — | -0.23 tok/s | -0.8 % | no class (one load) |
|  | gh | gh: 23.39, 24.51 → **23.94** (41.77 ms a token) | — | -5.59 tok/s | -18.9 % | no class (one load) |
|  | pu | pu1: 28.13, 31.07 → **29.53** (33.86 ms a token) | — | control | | |
|  | gu | gu1: 23.51, 24.64 → **24.06** (41.56 ms a token) | — | -5.47 tok/s | -18.5 % | no class (one load) |
| 8,192 | pg | pg: 20.70, 20.07 → **20.38** (49.07 ms a token) | — | -6.66 tok/s | -24.6 % | no class (one load) |
|  | gg | gg: 17.76, 17.03 → **17.39** (57.51 ms a token) | — | -9.65 tok/s | -35.7 % | no class (one load) |
|  | ph | ph: 25.93, 27.93 → **26.89** (37.19 ms a token) | — | -0.15 tok/s | -0.6 % | no class (one load) |
|  | gh | gh: 22.00, 22.97 → **22.48** (44.49 ms a token) | — | -4.57 tok/s | -16.9 % | no class (one load) |
|  | pu | pu1: 26.02, 28.15 → **27.04** (36.98 ms a token) | — | control | | |
|  | gu | gu1: 22.08, 23.14 → **22.60** (44.26 ms a token) | — | -4.45 tok/s | -16.4 % | no class (one load) |
| 9,216 | pg | pg: 21.33, 21.52 → **21.42** (46.68 ms a token) | — | -5.73 tok/s | -21.1 % | no class (one load) |
|  | gg | gg: 23.33, 23.93 → **23.63** (42.32 ms a token) | — | -3.52 tok/s | -13.0 % | no class (one load) |
|  | ph | ph: 26.60, 27.68 → **27.13** (36.86 ms a token) | — | -0.02 tok/s | -0.1 % | no class (one load) |
|  | gh | gh: 29.04, 30.19 → **29.60** (33.78 ms a token) | — | +2.45 tok/s | +9.0 % | no class (one load) |
|  | pu | pu1: 26.80, 27.51 → **27.15** (36.83 ms a token) | — | control | | |
|  | gu | gu1: 29.19, 30.35 → **29.76** (33.60 ms a token) | — | +2.61 tok/s | +9.6 % | no class (one load) |

### Generation over all depths, tok/s

Every depth's tokens over their time, per load.

| Depth (tokens) | Arm | Per load | Spread | Against `pu` | | Class |
|---:|---|---|---:|---:|---:|---|
| all | pg | pg: **21.44** | — | -6.42 tok/s | -23.1 % | no class (one load) |
|  | gg | gg: **19.48** | — | -8.38 tok/s | -30.1 % | no class (one load) |
|  | ph | ph: **27.73** | — | -0.13 tok/s | -0.5 % | no class (one load) |
|  | gh | gh: **24.99** | — | -2.87 tok/s | -10.3 % | no class (one load) |
|  | pu | pu1: **27.86** | — | control | | |
|  | gu | gu1: **25.12** | — | -2.74 tok/s | -9.8 % | no class (one load) |

### The step on pass 1 (the first request at a new depth), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `pu`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `pu` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | pg | pg: 131.47 (89 steps) | — | +29.23 ms | +28.6 % | no class (one load) |
|  | gg | gg: 133.85 (105 steps) | — | +31.61 ms | +30.9 % | no class (one load) |
|  | ph | ph: 103.37 (89 steps) | — | +1.13 ms | +1.1 % | no class (one load) |
|  | gh | gh: 104.23 (105 steps) | — | +2.00 ms | +2.0 % | no class (one load) |
|  | pu | pu1: 102.24 (89 steps) | — | control | | |
|  | gu | gu1: 103.71 (105 steps) | — | +1.47 ms | +1.4 % | no class (one load) |
| 8,192 | pg | pg: 142.14 (87 steps) | — | +29.04 ms | +25.7 % | no class (one load) |
|  | gg | gg: 148.57 (97 steps) | — | +35.47 ms | +31.4 % | no class (one load) |
|  | ph | ph: 113.48 (87 steps) | — | +0.38 ms | +0.3 % | no class (one load) |
|  | gh | gh: 119.94 (97 steps) | — | +6.84 ms | +6.1 % | no class (one load) |
|  | pu | pu1: 113.10 (87 steps) | — | control | | |
|  | gu | gu1: 119.53 (97 steps) | — | +6.43 ms | +5.7 % | no class (one load) |
| 9,216 | pg | pg: 139.58 (86 steps) | — | +28.53 ms | +25.7 % | no class (one load) |
|  | gg | gg: 148.30 (74 steps) | — | +37.25 ms | +33.5 % | no class (one load) |
|  | ph | ph: 111.90 (86 steps) | — | +0.85 ms | +0.8 % | no class (one load) |
|  | gh | gh: 119.12 (74 steps) | — | +8.07 ms | +7.3 % | no class (one load) |
|  | pu | pu1: 111.05 (86 steps) | — | control | | |
|  | gu | gu1: 118.50 (74 steps) | — | +7.44 ms | +6.7 % | no class (one load) |

### The step on pass 2 (the same request again), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `pu`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `pu` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | pg | pg: 124.13 (88 steps) | — | +30.51 ms | +32.6 % | no class (one load) |
|  | gg | gg: 131.36 (104 steps) | — | +37.74 ms | +40.3 % | no class (one load) |
|  | ph | ph: 94.00 (88 steps) | — | +0.38 ms | +0.4 % | no class (one load) |
|  | gh | gh: 100.42 (104 steps) | — | +6.80 ms | +7.3 % | no class (one load) |
|  | pu | pu1: 93.62 (88 steps) | — | control | | |
|  | gu | gu1: 99.90 (104 steps) | — | +6.29 ms | +6.7 % | no class (one load) |
| 8,192 | pg | pg: 151.88 (84 steps) | — | +43.62 ms | +40.3 % | no class (one load) |
|  | gg | gg: 154.98 (97 steps) | — | +46.72 ms | +43.2 % | no class (one load) |
|  | ph | ph: 109.13 (84 steps) | — | +0.87 ms | +0.8 % | no class (one load) |
|  | gh | gh: 114.90 (97 steps) | — | +6.63 ms | +6.1 % | no class (one load) |
|  | pu | pu1: 108.26 (84 steps) | — | control | | |
|  | gu | gu1: 114.06 (97 steps) | — | +5.80 ms | +5.4 % | no class (one load) |
| 9,216 | pg | pg: 138.33 (86 steps) | — | +30.12 ms | +27.8 % | no class (one load) |
|  | gg | gg: 144.54 (74 steps) | — | +36.32 ms | +33.6 % | no class (one load) |
|  | ph | ph: 107.55 (86 steps) | — | -0.67 ms | -0.6 % | no class (one load) |
|  | gh | gh: 114.60 (74 steps) | — | +6.38 ms | +5.9 % | no class (one load) |
|  | pu | pu1: 108.22 (86 steps) | — | control | | |
|  | gu | gu1: 114.00 (74 steps) | — | +5.78 ms | +5.3 % | no class (one load) |

### Prompt reading, tok/s

Per load: the tokens read at that depth over their time (a prime request reads a chunk on top of the cache; without primes each decode request reads its prompt), with the tokens read and the tokens reused. Requests that read under 32 tokens are left out.

| Depth (tokens) | Arm | Per load | Spread | Against `pu` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | pg | pg: 48.2 (64 read on 0 cached) | — | -6.7 tok/s | -12.2 % | no class (one load) |
|  | gg | gg: 45.9 (64 read on 0 cached) | — | -9.0 tok/s | -16.4 % | no class (one load) |
|  | ph | ph: 55.0 (64 read on 0 cached) | — | +0.1 tok/s | +0.2 % | no class (one load) |
|  | gh | gh: 48.7 (64 read on 0 cached) | — | -6.2 tok/s | -11.3 % | no class (one load) |
|  | pu | pu1: 54.9 (64 read on 0 cached) | — | control | | |
|  | gu | gu1: 48.2 (64 read on 0 cached) | — | -6.7 tok/s | -12.3 % | no class (one load) |
| 8,192 | pg | pg: 80.1 (8128 read on 0 cached) | — | -140.8 tok/s | -63.7 % | no class (one load) |
|  | gg | gg: 75.7 (8128 read on 0 cached) | — | -145.2 tok/s | -65.7 % | no class (one load) |
|  | ph | ph: 142.4 (8128 read on 0 cached) | — | -78.6 tok/s | -35.6 % | no class (one load) |
|  | gh | gh: 124.3 (8128 read on 0 cached) | — | -96.6 tok/s | -43.7 % | no class (one load) |
|  | pu | pu1: 221.0 (8128 read on 0 cached) | — | control | | |
|  | gu | gu1: 145.1 (8128 read on 0 cached) | — | -75.9 tok/s | -34.3 % | no class (one load) |
| 9,216 | pg | pg: 73.0 (1540 read on 7612 cached) | — | -98.1 tok/s | -57.3 % | no class (one load) |
|  | gg | gg: 66.9 (1540 read on 7612 cached) | — | -104.3 tok/s | -60.9 % | no class (one load) |
|  | ph | ph: 99.0 (1540 read on 7612 cached) | — | -72.2 tok/s | -42.2 % | no class (one load) |
|  | gh | gh: 94.3 (1540 read on 7612 cached) | — | -76.9 tok/s | -44.9 % | no class (one load) |
|  | pu | pu1: 171.2 (1540 read on 7612 cached) | — | control | | |
|  | gu | gu1: 114.7 (1540 read on 7612 cached) | — | -56.4 tok/s | -33.0 % | no class (one load) |

### The draft and the step

Per load, pass 1: draft tokens accepted of drafted (the share), tokens a step (generated tokens over steps, a step being one pass of the main model: generated minus accepted), and the time of a step. Pass 2 is listed where its counters differ.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | pg | pg: 167 of 261 (0.640), 2.88 a step, 131.5 ms a step (pass 2: 168 of 258 (0.651), 2.91 a step, 124.1 ms a step) |
|  | gg | gg: 151 of 312 (0.484), 2.44 a step, 133.8 ms a step (pass 2: 152 of 309 (0.492), 2.46 a step, 131.4 ms a step) |
|  | ph | ph: 167 of 261 (0.640), 2.88 a step, 103.4 ms a step (pass 2: 168 of 258 (0.651), 2.91 a step, 94.0 ms a step) |
|  | gh | gh: 151 of 312 (0.484), 2.44 a step, 104.2 ms a step (pass 2: 152 of 309 (0.492), 2.46 a step, 100.4 ms a step) |
|  | pu | pu1: 167 of 261 (0.640), 2.88 a step, 102.2 ms a step (pass 2: 168 of 258 (0.651), 2.91 a step, 93.6 ms a step) |
|  | gu | gu1: 151 of 312 (0.484), 2.44 a step, 103.7 ms a step (pass 2: 152 of 309 (0.492), 2.46 a step, 99.9 ms a step) |
| 8,192 | pg | pg: 169 of 257 (0.658), 2.94 a step, 142.1 ms a step (pass 2: 172 of 248 (0.694), 3.05 a step, 151.9 ms a step) |
|  | gg | gg: 159 of 286 (0.556), 2.64 a step, 148.6 ms a step |
|  | ph | ph: 169 of 257 (0.658), 2.94 a step, 113.5 ms a step (pass 2: 172 of 248 (0.694), 3.05 a step, 109.1 ms a step) |
|  | gh | gh: 159 of 286 (0.556), 2.64 a step, 119.9 ms a step |
|  | pu | pu1: 169 of 257 (0.658), 2.94 a step, 113.1 ms a step (pass 2: 172 of 248 (0.694), 3.05 a step, 108.3 ms a step) |
|  | gu | gu1: 159 of 286 (0.556), 2.64 a step, 119.5 ms a step |
| 9,216 | pg | pg: 170 of 252 (0.675), 2.98 a step, 139.6 ms a step |
|  | gg | gg: 182 of 218 (0.835), 3.46 a step, 148.3 ms a step |
|  | ph | ph: 170 of 252 (0.675), 2.98 a step, 111.9 ms a step |
|  | gh | gh: 182 of 218 (0.835), 3.46 a step, 119.1 ms a step |
|  | pu | pu1: 170 of 252 (0.675), 2.98 a step, 111.1 ms a step |
|  | gu | gu1: 182 of 218 (0.835), 3.46 a step, 118.5 ms a step |

### The sampler pass

One request per depth and load with the server line's own sampler and a fixed seed: tok/s, the draft counters and the step.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | pg | pg: 23.63 tok/s, 171 of 251 (0.681), 3.01 a step, 127.5 ms a step |
|  | gg | gg: 15.89 tok/s, 137 of 354 (0.387), 2.15 a step, 135.3 ms a step |
|  | ph | ph: 31.19 tok/s, 171 of 251 (0.681), 3.01 a step, 96.6 ms a step |
|  | gh | gh: 20.48 tok/s, 137 of 354 (0.387), 2.15 a step, 105.0 ms a step |
|  | pu | pu1: 31.33 tok/s, 171 of 251 (0.681), 3.01 a step, 96.1 ms a step |
|  | gu | gu1: 20.65 tok/s, 137 of 354 (0.387), 2.15 a step, 104.2 ms a step |
| 8,192 | pg | pg: 19.70 tok/s, 168 of 258 (0.651), 2.91 a step, 147.7 ms a step |
|  | gg | gg: 19.03 tok/s, 170 of 252 (0.675), 2.98 a step, 156.4 ms a step |
|  | ph | ph: 25.88 tok/s, 168 of 258 (0.651), 2.91 a step, 112.4 ms a step |
|  | gh | gh: 24.88 tok/s, 170 of 252 (0.675), 2.98 a step, 119.7 ms a step |
|  | pu | pu1: 26.03 tok/s, 168 of 258 (0.651), 2.91 a step, 111.8 ms a step |
|  | gu | gu1: 19.79 tok/s, 170 of 252 (0.675), 2.98 a step, 150.4 ms a step |
| 9,216 | pg | pg: 21.47 tok/s, 172 of 246 (0.699), 3.05 a step, 142.0 ms a step |
|  | gg | gg: 18.59 tok/s, 168 of 261 (0.644), 2.91 a step, 156.5 ms a step |
|  | ph | ph: 27.32 tok/s, 172 of 246 (0.699), 3.05 a step, 111.6 ms a step |
|  | gh | gh: 24.93 tok/s, 168 of 261 (0.644), 2.91 a step, 116.7 ms a step |
|  | pu | pu1: 27.51 tok/s, 172 of 246 (0.699), 3.05 a step, 110.8 ms a step |
|  | gu | gu1: 25.10 tok/s, 168 of 261 (0.644), 2.91 a step, 115.9 ms a step |

### Checks

- Times: every load started on a machine checked quiet (valid=1).
- Prompts: 64 ids at depth 0, 8192 ids at depth 8,192, 9216 ids at depth 9,216; the same ids on every load and pass.
- Tokens out: 256 on every timed request.
- Greedy reply at depth 0: **differs from `pu` in gg, gh, gu.**
- Greedy reply at depth 8,192: **differs from `pu` in gg, gh, gu.**
- Greedy reply at depth 9,216: **differs from `pu` in gg, gh, gu.**
- Primes (read on cached, the cache's source): 8,192: 8128 on 0 (none); 9,216: 1540 on 7612 (checkpoint).
- Graph reallocations of the target, requests with one per load: pg: 0 of 12 requests; gg: 0 of 12 requests; ph: 0 of 12 requests; gh: 0 of 12 requests; pu1: 0 of 12 requests; gu1: 0 of 12 requests.
