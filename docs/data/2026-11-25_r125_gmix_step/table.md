## The deep lane: pg, gg, ph, gh, pu, gu, pm, gm

Model file `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-pmix-fast-00001-of-00003.gguf`, llama-server version: 1.0.0-dev (build 13218, commit eee615cdb9).

256 tokens per request at temperature 0, each request twice per load (pass 1 is the first request at a new depth, pass 2 the same request again). Timed loads, in their order: pg, gg, ph, gh, pu1, gu1, pm1, gm1.

| Arm | Loads | Draft | Context | Its own variables and flags |
|---|---|---|---:|---|
| pg | pg | on | 65536 | `GGML_CUDA_PROFILE=<OUT>/gp-pg.csv GGML_CUDA_PROFILE_MODE=group` |
| gg | gg | on | 65536 | `GGML_CUDA_PROFILE=<OUT>/gp-gg.csv GGML_CUDA_PROFILE_MODE=group`, `-m <MODELS>/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-pmix-fast-gmix-q2-00001-of-00003.gguf` |
| ph | ph | on | 65536 | `GGML_CUDA_PROFILE_HOST=<OUT>/hp-ph.csv LLAMA_STEP_TRACE=<OUT>/st-ph.jsonl` |
| gh | gh | on | 65536 | `GGML_CUDA_PROFILE_HOST=<OUT>/hp-gh.csv LLAMA_STEP_TRACE=<OUT>/st-gh.jsonl`, `-m <MODELS>/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-pmix-fast-gmix-q2-00001-of-00003.gguf` |
| pu (control) | pu1 | on | 65536 | — |
| gu | gu1 | on | 65536 | `-m <MODELS>/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-pmix-fast-gmix-q2-00001-of-00003.gguf` |
| pm | pm1 | on | 65536 | `GGML_CUDA_MMVQ_PASCAL_MT=1` |
| gm | gm1 | on | 65536 | `GGML_CUDA_MMVQ_PASCAL_MT=1`, `-m <MODELS>/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-pmix-fast-gmix-q2-00001-of-00003.gguf` |

### Generation, tok/s

Per load: pass 1, pass 2 → the load, with the load's time per token. A load's speed is its two passes' tokens over their time. Spread: the difference between the arm's fastest and slowest load. Class against `pu`: gain when the mean is above the control's by more than the larger of the two spreads, loss when below by more than it, else negligible.

| Depth (tokens) | Arm | Per load | Spread | Against `pu` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | pg | pg: 21.71, 22.69 → **22.19** (45.07 ms a token) | — | -6.69 tok/s | -23.2 % | no class (one load) |
|  | gg | gg: 16.45, 16.99 → **16.72** (59.82 ms a token) | — | -12.16 tok/s | -42.1 % | no class (one load) |
|  | ph | ph: 27.19, 30.27 → **28.65** (34.90 ms a token) | — | -0.22 tok/s | -0.8 % | no class (one load) |
|  | gh | gh: 20.64, 21.54 → **21.08** (47.43 ms a token) | — | -7.79 tok/s | -27.0 % | no class (one load) |
|  | pu | pu1: 27.50, 30.39 → **28.87** (34.64 ms a token) | — | control | | |
|  | gu | gu1: 20.74, 21.63 → **21.18** (47.22 ms a token) | — | -7.69 tok/s | -26.6 % | no class (one load) |
|  | pm | pm1: 27.50, 30.45 → **28.90** (34.60 ms a token) | — | +0.03 tok/s | +0.1 % | no class (one load) |
|  | gm | gm1: 20.54, 21.67 → **21.09** (47.41 ms a token) | — | -7.78 tok/s | -26.9 % | no class (one load) |
| 8,192 | pg | pg: 20.24, 20.54 → **20.39** (49.04 ms a token) | — | -5.43 tok/s | -21.0 % | no class (one load) |
|  | gg | gg: 15.39, 16.38 → **15.87** (63.02 ms a token) | — | -9.95 tok/s | -38.5 % | no class (one load) |
|  | ph | ph: 25.22, 27.46 → **26.29** (38.03 ms a token) | — | +0.47 tok/s | +1.8 % | no class (one load) |
|  | gh | gh: 19.77, 20.48 → **20.12** (49.71 ms a token) | — | -5.70 tok/s | -22.1 % | no class (one load) |
|  | pu | pu1: 24.31, 27.53 → **25.82** (38.73 ms a token) | — | control | | |
|  | gu | gu1: 19.86, 20.56 → **20.21** (49.49 ms a token) | — | -5.61 tok/s | -21.7 % | no class (one load) |
|  | pm | pm1: 25.45, 27.55 → **26.46** (37.79 ms a token) | — | +0.64 tok/s | +2.5 % | no class (one load) |
|  | gm | gm1: 19.90, 20.48 → **20.19** (49.54 ms a token) | — | -5.63 tok/s | -21.8 % | no class (one load) |

### Generation over all depths, tok/s

Every depth's tokens over their time, per load.

| Depth (tokens) | Arm | Per load | Spread | Against `pu` | | Class |
|---:|---|---|---:|---:|---:|---|
| all | pg | pg: **21.25** | — | -6.01 tok/s | -22.0 % | no class (one load) |
|  | gg | gg: **16.28** | — | -10.98 tok/s | -40.3 % | no class (one load) |
|  | ph | ph: **27.42** | — | +0.16 tok/s | +0.6 % | no class (one load) |
|  | gh | gh: **20.59** | — | -6.67 tok/s | -24.5 % | no class (one load) |
|  | pu | pu1: **27.26** | — | control | | |
|  | gu | gu1: **20.68** | — | -6.58 tok/s | -24.1 % | no class (one load) |
|  | pm | pm1: **27.63** | — | +0.37 tok/s | +1.3 % | no class (one load) |
|  | gm | gm1: **20.63** | — | -6.63 tok/s | -24.3 % | no class (one load) |

### The step on pass 1 (the first request at a new depth), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `pu`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `pu` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | pg | pg: 132.52 (89 steps) | — | +27.92 ms | +26.7 % | no class (one load) |
|  | gg | gg: 148.21 (105 steps) | — | +43.61 ms | +41.7 % | no class (one load) |
|  | ph | ph: 105.77 (89 steps) | — | +1.18 ms | +1.1 % | no class (one load) |
|  | gh | gh: 118.12 (105 steps) | — | +13.52 ms | +12.9 % | no class (one load) |
|  | pu | pu1: 104.60 (89 steps) | — | control | | |
|  | gu | gu1: 117.55 (105 steps) | — | +12.95 ms | +12.4 % | no class (one load) |
|  | pm | pm1: 104.59 (89 steps) | — | -0.00 ms | -0.0 % | no class (one load) |
|  | gm | gm1: 118.69 (105 steps) | — | +14.10 ms | +13.5 % | no class (one load) |
| 8,192 | pg | pg: 145.35 (87 steps) | — | +24.31 ms | +20.1 % | no class (one load) |
|  | gg | gg: 171.50 (97 steps) | — | +50.47 ms | +41.7 % | no class (one load) |
|  | ph | ph: 116.65 (87 steps) | — | -4.38 ms | -3.6 % | no class (one load) |
|  | gh | gh: 133.50 (97 steps) | — | +12.46 ms | +10.3 % | no class (one load) |
|  | pu | pu1: 121.04 (87 steps) | — | control | | |
|  | gu | gu1: 132.87 (97 steps) | — | +11.83 ms | +9.8 % | no class (one load) |
|  | pm | pm1: 115.62 (87 steps) | — | -5.41 ms | -4.5 % | no class (one load) |
|  | gm | gm1: 132.64 (97 steps) | — | +11.60 ms | +9.6 % | no class (one load) |

### The step on pass 2 (the same request again), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `pu`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `pu` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | pg | pg: 128.22 (88 steps) | — | +32.49 ms | +33.9 % | no class (one load) |
|  | gg | gg: 144.88 (104 steps) | — | +49.15 ms | +51.3 % | no class (one load) |
|  | ph | ph: 96.11 (88 steps) | — | +0.38 ms | +0.4 % | no class (one load) |
|  | gh | gh: 114.26 (104 steps) | — | +18.53 ms | +19.4 % | no class (one load) |
|  | pu | pu1: 95.73 (88 steps) | — | control | | |
|  | gu | gu1: 113.78 (104 steps) | — | +18.05 ms | +18.9 % | no class (one load) |
|  | pm | pm1: 95.53 (88 steps) | — | -0.20 ms | -0.2 % | no class (one load) |
|  | gm | gm1: 113.57 (104 steps) | — | +17.84 ms | +18.6 % | no class (one load) |
| 8,192 | pg | pg: 148.37 (84 steps) | — | +37.66 ms | +34.0 % | no class (one load) |
|  | gg | gg: 161.11 (97 steps) | — | +50.40 ms | +45.5 % | no class (one load) |
|  | ph | ph: 111.00 (84 steps) | — | +0.29 ms | +0.3 % | no class (one load) |
|  | gh | gh: 128.88 (97 steps) | — | +18.17 ms | +16.4 % | no class (one load) |
|  | pu | pu1: 110.71 (84 steps) | — | control | | |
|  | gu | gu1: 128.37 (97 steps) | — | +17.66 ms | +15.9 % | no class (one load) |
|  | pm | pm1: 110.61 (84 steps) | — | -0.10 ms | -0.1 % | no class (one load) |
|  | gm | gm1: 128.85 (97 steps) | — | +18.14 ms | +16.4 % | no class (one load) |

### Prompt reading, tok/s

Per load: the tokens read at that depth over their time (a prime request reads a chunk on top of the cache; without primes each decode request reads its prompt), with the tokens read and the tokens reused. Requests that read under 32 tokens are left out.

| Depth (tokens) | Arm | Per load | Spread | Against `pu` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | pg | pg: 44.6 (64 read on 0 cached) | — | -10.8 tok/s | -19.5 % | no class (one load) |
|  | gg | gg: 45.2 (64 read on 0 cached) | — | -10.1 tok/s | -18.3 % | no class (one load) |
|  | ph | ph: 53.3 (64 read on 0 cached) | — | -2.0 tok/s | -3.6 % | no class (one load) |
|  | gh | gh: 48.5 (64 read on 0 cached) | — | -6.9 tok/s | -12.4 % | no class (one load) |
|  | pu | pu1: 55.3 (64 read on 0 cached) | — | control | | |
|  | gu | gu1: 48.5 (64 read on 0 cached) | — | -6.8 tok/s | -12.3 % | no class (one load) |
|  | pm | pm1: 53.9 (64 read on 0 cached) | — | -1.4 tok/s | -2.6 % | no class (one load) |
|  | gm | gm1: 48.5 (64 read on 0 cached) | — | -6.9 tok/s | -12.4 % | no class (one load) |
| 8,192 | pg | pg: 82.8 (8128 read on 0 cached) | — | -136.1 tok/s | -62.2 % | no class (one load) |
|  | gg | gg: 76.8 (8128 read on 0 cached) | — | -142.1 tok/s | -64.9 % | no class (one load) |
|  | ph | ph: 142.6 (8128 read on 0 cached) | — | -76.3 tok/s | -34.9 % | no class (one load) |
|  | gh | gh: 126.2 (8128 read on 0 cached) | — | -92.7 tok/s | -42.3 % | no class (one load) |
|  | pu | pu1: 218.9 (8128 read on 0 cached) | — | control | | |
|  | gu | gu1: 148.0 (8128 read on 0 cached) | — | -70.9 tok/s | -32.4 % | no class (one load) |
|  | pm | pm1: 217.9 (8128 read on 0 cached) | — | -1.0 tok/s | -0.5 % | no class (one load) |
|  | gm | gm1: 142.4 (8128 read on 0 cached) | — | -76.5 tok/s | -35.0 % | no class (one load) |

### The draft and the step

Per load, pass 1: draft tokens accepted of drafted (the share), tokens a step (generated tokens over steps, a step being one pass of the main model: generated minus accepted), and the time of a step. Pass 2 is listed where its counters differ.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | pg | pg: 167 of 261 (0.640), 2.88 a step, 132.5 ms a step (pass 2: 168 of 258 (0.651), 2.91 a step, 128.2 ms a step) |
|  | gg | gg: 151 of 312 (0.484), 2.44 a step, 148.2 ms a step (pass 2: 152 of 309 (0.492), 2.46 a step, 144.9 ms a step) |
|  | ph | ph: 167 of 261 (0.640), 2.88 a step, 105.8 ms a step (pass 2: 168 of 258 (0.651), 2.91 a step, 96.1 ms a step) |
|  | gh | gh: 151 of 312 (0.484), 2.44 a step, 118.1 ms a step (pass 2: 152 of 309 (0.492), 2.46 a step, 114.3 ms a step) |
|  | pu | pu1: 167 of 261 (0.640), 2.88 a step, 104.6 ms a step (pass 2: 168 of 258 (0.651), 2.91 a step, 95.7 ms a step) |
|  | gu | gu1: 151 of 312 (0.484), 2.44 a step, 117.5 ms a step (pass 2: 152 of 309 (0.492), 2.46 a step, 113.8 ms a step) |
|  | pm | pm1: 167 of 261 (0.640), 2.88 a step, 104.6 ms a step (pass 2: 168 of 258 (0.651), 2.91 a step, 95.5 ms a step) |
|  | gm | gm1: 151 of 312 (0.484), 2.44 a step, 118.7 ms a step (pass 2: 152 of 309 (0.492), 2.46 a step, 113.6 ms a step) |
| 8,192 | pg | pg: 169 of 257 (0.658), 2.94 a step, 145.3 ms a step (pass 2: 172 of 248 (0.694), 3.05 a step, 148.4 ms a step) |
|  | gg | gg: 159 of 286 (0.556), 2.64 a step, 171.5 ms a step |
|  | ph | ph: 169 of 257 (0.658), 2.94 a step, 116.7 ms a step (pass 2: 172 of 248 (0.694), 3.05 a step, 111.0 ms a step) |
|  | gh | gh: 159 of 286 (0.556), 2.64 a step, 133.5 ms a step |
|  | pu | pu1: 169 of 257 (0.658), 2.94 a step, 121.0 ms a step (pass 2: 172 of 248 (0.694), 3.05 a step, 110.7 ms a step) |
|  | gu | gu1: 159 of 286 (0.556), 2.64 a step, 132.9 ms a step |
|  | pm | pm1: 169 of 257 (0.658), 2.94 a step, 115.6 ms a step (pass 2: 172 of 248 (0.694), 3.05 a step, 110.6 ms a step) |
|  | gm | gm1: 159 of 286 (0.556), 2.64 a step, 132.6 ms a step |

### The sampler pass

One request per depth and load with the server line's own sampler and a fixed seed: tok/s, the draft counters and the step.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | pg | pg: 23.26 tok/s, 171 of 251 (0.681), 3.01 a step, 129.5 ms a step |
|  | gg | gg: 14.40 tok/s, 137 of 354 (0.387), 2.15 a step, 149.4 ms a step |
|  | ph | ph: 30.50 tok/s, 171 of 251 (0.681), 3.01 a step, 98.7 ms a step |
|  | gh | gh: 18.11 tok/s, 137 of 354 (0.387), 2.15 a step, 118.8 ms a step |
|  | pu | pu1: 30.57 tok/s, 171 of 251 (0.681), 3.01 a step, 98.5 ms a step |
|  | gu | gu1: 18.20 tok/s, 137 of 354 (0.387), 2.15 a step, 118.2 ms a step |
|  | pm | pm1: 30.69 tok/s, 171 of 251 (0.681), 3.01 a step, 98.1 ms a step |
|  | gm | gm1: 18.21 tok/s, 137 of 354 (0.387), 2.15 a step, 118.1 ms a step |
| 8,192 | pg | pg: 19.15 tok/s, 168 of 258 (0.651), 2.91 a step, 151.9 ms a step |
|  | gg | gg: 17.86 tok/s, 170 of 252 (0.675), 2.98 a step, 166.7 ms a step |
|  | ph | ph: 25.33 tok/s, 168 of 258 (0.651), 2.91 a step, 114.9 ms a step |
|  | gh | gh: 22.43 tok/s, 170 of 252 (0.675), 2.98 a step, 132.7 ms a step |
|  | pu | pu1: 25.25 tok/s, 168 of 258 (0.651), 2.91 a step, 115.2 ms a step |
|  | gu | gu1: 22.58 tok/s, 170 of 252 (0.675), 2.98 a step, 131.8 ms a step |
|  | pm | pm1: 25.47 tok/s, 168 of 258 (0.651), 2.91 a step, 114.2 ms a step |
|  | gm | gm1: 21.55 tok/s, 170 of 252 (0.675), 2.98 a step, 138.1 ms a step |

### Checks

- Times: every load started on a machine checked quiet (valid=1).
- Prompts: 64 ids at depth 0, 8192 ids at depth 8,192; the same ids on every load and pass.
- Tokens out: 256 on every timed request.
- Greedy reply at depth 0: **differs from `pu` in gg, gh, gu, gm.**
- Greedy reply at depth 8,192: **differs from `pu` in gg, gh, gu, gm.**
- Primes (read on cached, the cache's source): 8,192: 8128 on 0 (none).
- Graph reallocations of the target, requests with one per load: pg: 0 of 8 requests; gg: 0 of 8 requests; ph: 0 of 8 requests; gh: 0 of 8 requests; pu1: 0 of 8 requests; gu1: 0 of 8 requests; pm1: 0 of 8 requests; gm1: 0 of 8 requests.
