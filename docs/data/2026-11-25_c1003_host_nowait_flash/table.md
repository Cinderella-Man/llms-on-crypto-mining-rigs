## The deep lane: u, hw, hn

Model file `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf`, llama-server version: 1.0.0-dev (build 13218, commit eee615cdb9).

256 tokens per request at temperature 0, each request twice per load (pass 1 is the first request at a new depth, pass 2 the same request again). Timed loads, in their order: u1, hw1, hn1, u2.

| Arm | Loads | Draft | Context | Its own variables and flags |
|---|---|---|---:|---|
| u (control) | u1, u2 | on | 65536 | — |
| hw | hw1 | on | 65536 | `GGML_CUDA_PROFILE_HOST=<OUT>/hp-hw.csv LLAMA_STEP_TRACE=<OUT>/st-hw.jsonl` |
| hn | hn1 | on | 65536 | `GGML_CUDA_PROFILE_HOST=<OUT>/hp-hn.csv LLAMA_STEP_TRACE=<OUT>/st-hn.jsonl GGML_CUDA_PROFILE_HOST_NOWAIT=1` |

### Generation, tok/s

Per load: pass 1, pass 2 → the load, with the load's time per token. A load's speed is its two passes' tokens over their time. Spread: the difference between the arm's fastest and slowest load. Class against `u`: gain when the mean is above the control's by more than the larger of the two spreads, loss when below by more than it, else negligible.

| Depth (tokens) | Arm | Per load | Spread | Against `u` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | u | u1: 22.42, 23.37 → **22.89** (43.69 ms a token); u2: 22.21, 23.36 → **22.77** (43.92 ms a token) | 0.12 | control | | |
|  | hw | hw1: 22.01, 23.26 → **22.62** (44.22 ms a token) | — | -0.21 tok/s | -0.9 % | no class (one load) |
|  | hn | hn1: 21.99, 23.23 → **22.59** (44.26 ms a token) | — | -0.24 tok/s | -1.0 % | no class (one load) |
| 8,192 | u | u1: 29.31, 30.70 → **29.99** (33.35 ms a token); u2: 29.24, 30.66 → **29.93** (33.41 ms a token) | 0.06 | control | | |
|  | hw | hw1: 28.85, 30.72 → **29.75** (33.61 ms a token) | — | -0.21 tok/s | -0.7 % | no class (one load) |
|  | hn | hn1: 29.03, 30.56 → **29.77** (33.59 ms a token) | — | -0.19 tok/s | -0.6 % | no class (one load) |

### Generation over all depths, tok/s

Every depth's tokens over their time, per load.

| Depth (tokens) | Arm | Per load | Spread | Against `u` | | Class |
|---:|---|---|---:|---:|---:|---|
| all | u | u1: **25.96**; u2: **25.86** | 0.10 | control | | |
|  | hw | hw1: **25.70** | — | -0.21 tok/s | -0.8 % | no class (one load) |
|  | hn | hn1: **25.69** | — | -0.22 tok/s | -0.9 % | no class (one load) |

### The step on pass 1 (the first request at a new depth), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `u`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `u` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | u | u1: 104.74 (109 steps); u2: 105.74 (109 steps) | 0.99 | control | | |
|  | hw | hw1: 106.70 (109 steps) | — | +1.46 ms | +1.4 % | no class (one load) |
|  | hn | hn1: 106.80 (109 steps) | — | +1.56 ms | +1.5 % | no class (one load) |
| 8,192 | u | u1: 119.66 (73 steps); u2: 119.95 (73 steps) | 0.29 | control | | |
|  | hw | hw1: 121.56 (73 steps) | — | +1.75 ms | +1.5 % | no class (one load) |
|  | hn | hn1: 120.81 (73 steps) | — | +1.01 ms | +0.8 % | no class (one load) |

### The step on pass 2 (the same request again), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `u`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `u` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | u | u1: 101.43 (108 steps); u2: 101.48 (108 steps) | 0.05 | control | | |
|  | hw | hw1: 101.93 (108 steps) | — | +0.47 ms | +0.5 % | no class (one load) |
|  | hn | hn1: 102.05 (108 steps) | — | +0.59 ms | +0.6 % | no class (one load) |
| 8,192 | u | u1: 115.81 (72 steps); u2: 115.97 (72 steps) | 0.16 | control | | |
|  | hw | hw1: 115.75 (72 steps) | — | -0.14 ms | -0.1 % | no class (one load) |
|  | hn | hn1: 116.35 (72 steps) | — | +0.46 ms | +0.4 % | no class (one load) |

### Prompt reading, tok/s

Per load: the tokens read at that depth over their time (a prime request reads a chunk on top of the cache; without primes each decode request reads its prompt), with the tokens read and the tokens reused. Requests that read under 32 tokens are left out.

| Depth (tokens) | Arm | Per load | Spread | Against `u` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | u | u1: 53.5 (64 read on 0 cached); u2: 50.9 (64 read on 0 cached) | 2.6 | control | | |
|  | hw | hw1: 51.2 (64 read on 0 cached) | — | -1.0 tok/s | -2.0 % | no class (one load) |
|  | hn | hn1: 51.6 (64 read on 0 cached) | — | -0.6 tok/s | -1.2 % | no class (one load) |
| 8,192 | u | u1: 222.2 (8128 read on 0 cached); u2: 214.9 (8128 read on 0 cached) | 7.3 | control | | |
|  | hw | hw1: 149.8 (8128 read on 0 cached) | — | -68.8 tok/s | -31.5 % | no class (one load) |
|  | hn | hn1: 213.3 (8128 read on 0 cached) | — | -5.3 tok/s | -2.4 % | no class (one load) |

### The draft and the step

Per load, pass 1: draft tokens accepted of drafted (the share), tokens a step (generated tokens over steps, a step being one pass of the main model: generated minus accepted), and the time of a step. Pass 2 is listed where its counters differ.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | u | u1: 147 of 321 (0.458), 2.35 a step, 104.7 ms a step (pass 2: 148 of 318 (0.465), 2.37 a step, 101.4 ms a step); u2: 147 of 321 (0.458), 2.35 a step, 105.7 ms a step (pass 2: 148 of 318 (0.465), 2.37 a step, 101.5 ms a step) |
|  | hw | hw1: 147 of 321 (0.458), 2.35 a step, 106.7 ms a step (pass 2: 148 of 318 (0.465), 2.37 a step, 101.9 ms a step) |
|  | hn | hn1: 147 of 321 (0.458), 2.35 a step, 106.8 ms a step (pass 2: 148 of 318 (0.465), 2.37 a step, 102.0 ms a step) |
| 8,192 | u | u1: 183 of 214 (0.855), 3.51 a step, 119.7 ms a step (pass 2: 184 of 211 (0.872), 3.56 a step, 115.8 ms a step); u2: 183 of 214 (0.855), 3.51 a step, 120.0 ms a step (pass 2: 184 of 211 (0.872), 3.56 a step, 116.0 ms a step) |
|  | hw | hw1: 183 of 214 (0.855), 3.51 a step, 121.6 ms a step (pass 2: 184 of 211 (0.872), 3.56 a step, 115.8 ms a step) |
|  | hn | hn1: 183 of 214 (0.855), 3.51 a step, 120.8 ms a step (pass 2: 184 of 211 (0.872), 3.56 a step, 116.4 ms a step) |

### The sampler pass

One request per depth and load with the server line's own sampler and a fixed seed: tok/s, the draft counters and the step.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | u | u1: 25.66 tok/s, 159 of 285 (0.558), 2.64 a step, 102.8 ms a step; u2: 25.06 tok/s, 159 of 285 (0.558), 2.64 a step, 105.3 ms a step |
|  | hw | hw1: 25.23 tok/s, 159 of 285 (0.558), 2.64 a step, 104.6 ms a step |
|  | hn | hn1: 25.22 tok/s, 159 of 285 (0.558), 2.64 a step, 104.6 ms a step |
| 8,192 | u | u1: 24.11 tok/s, 169 of 257 (0.658), 2.94 a step, 122.1 ms a step; u2: 24.10 tok/s, 169 of 257 (0.658), 2.94 a step, 122.1 ms a step |
|  | hw | hw1: 23.63 tok/s, 169 of 257 (0.658), 2.94 a step, 124.5 ms a step |
|  | hn | hn1: 23.97 tok/s, 169 of 257 (0.658), 2.94 a step, 122.8 ms a step |

### Checks

- Times: every load started on a machine checked quiet (valid=1).
- Prompts: 64 ids at depth 0, 8192 ids at depth 8,192; the same ids on every load and pass.
- Tokens out: 256 on every timed request.
- Greedy reply at depth 0: the same on every pass and load of every arm.
- Greedy reply at depth 8,192: the same on every pass and load of every arm.
- Primes (read on cached, the cache's source): 8,192: 8128 on 0 (none).
- Graph reallocations of the target, requests with one per load: u1: 0 of 8 requests; hw1: 0 of 8 requests; hn1: 0 of 8 requests; u2: 0 of 8 requests.
