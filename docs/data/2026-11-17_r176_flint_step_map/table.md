## The deep lane: h1, g1, s8, s4, s2, h2, g2

Model file `flint-near-q8-p065.gguf`, llama-server version: 1.0.0-dev (build 12832, commit eb312b7bce).

256 tokens per request at temperature 0, each request twice per load (pass 1 is the first request at a new depth, pass 2 the same request again). Timed loads, in their order: h1, g1, s8, s4, s2, h2, g2.

| Arm | Loads | Draft | Context | Its own variables and flags |
|---|---|---|---:|---|
| h1 (control) | h1 | off | 262144 | `GGML_CUDA_PROFILE_HOST=<OUT>/hp-h1.csv LLAMA_STEP_TRACE=<OUT>/st-h1.jsonl` |
| g1 | g1 | off | 262144 | `GGML_CUDA_PROFILE=<OUT>/gp-g1.csv GGML_CUDA_PROFILE_MODE=group` |
| s8 | s8 | off | 262144 | `GGML_CUDA_PROFILE_HOST=<OUT>/hp-s8.csv LLAMA_STEP_TRACE=<OUT>/st-s8.jsonl`, `--spec-type ngram-simple --spec-ngram-simple-size-n 1 --spec-ngram-simple-size-m 7 --spec-synth-len 8` |
| s4 | s4 | off | 262144 | `GGML_CUDA_PROFILE_HOST=<OUT>/hp-s4.csv LLAMA_STEP_TRACE=<OUT>/st-s4.jsonl`, `--spec-type ngram-simple --spec-ngram-simple-size-n 1 --spec-ngram-simple-size-m 3 --spec-synth-len 4` |
| s2 | s2 | off | 262144 | `GGML_CUDA_PROFILE_HOST=<OUT>/hp-s2.csv LLAMA_STEP_TRACE=<OUT>/st-s2.jsonl`, `--spec-type ngram-simple --spec-ngram-simple-size-n 1 --spec-ngram-simple-size-m 1 --spec-synth-len 2` |
| h2 | h2 | off | 262144 | `GGML_CUDA_PROFILE_HOST=<OUT>/hp-h2.csv LLAMA_STEP_TRACE=<OUT>/st-h2.jsonl` |
| g2 | g2 | off | 262144 | `GGML_CUDA_PROFILE=<OUT>/gp-g2.csv GGML_CUDA_PROFILE_MODE=group` |

### Generation, tok/s

Per load: pass 1, pass 2 → the load, with the load's time per token. A load's speed is its two passes' tokens over their time. Spread: the difference between the arm's fastest and slowest load. Class against `h1`: gain when the mean is above the control's by more than the larger of the two spreads, loss when below by more than it, else negligible.

| Depth (tokens) | Arm | Per load | Spread | Against `h1` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | h1 | h1: 27.17, 28.59 → **27.86** (35.89 ms a token) | — | control | | |
|  | g1 | g1: 13.61, 13.94 → **13.78** (72.59 ms a token) | — | -14.08 tok/s | -50.6 % | no class (one load) |
|  | s8 | s8: 28.69, 28.90 → **28.79** (34.73 ms a token) | — | +0.93 tok/s | +3.4 % | no class (one load) |
|  | s4 | s4: 26.56, 28.01 → **27.27** (36.67 ms a token) | — | -0.59 tok/s | -2.1 % | no class (one load) |
|  | s2 | s2: 25.59, 27.01 → **26.28** (38.05 ms a token) | — | -1.58 tok/s | -5.7 % | no class (one load) |
|  | h2 | h2: 25.93, 28.65 → **27.22** (36.73 ms a token) | — | -0.64 tok/s | -2.3 % | no class (one load) |
|  | g2 | g2: 13.10, 13.84 → **13.46** (74.30 ms a token) | — | -14.40 tok/s | -51.7 % | no class (one load) |
| 8,192 | h1 | h1: 20.27, 20.86 → **20.56** (48.64 ms a token) | — | control | | |
|  | g1 | g1: 8.90, 8.69 → **8.79** (113.73 ms a token) | — | -11.76 tok/s | -57.2 % | no class (one load) |
|  | s8 | s8: 27.64, 27.59 → **27.61** (36.21 ms a token) | — | +7.06 tok/s | +34.3 % | no class (one load) |
|  | s4 | s4: 26.79, 26.82 → **26.80** (37.31 ms a token) | — | +6.25 tok/s | +30.4 % | no class (one load) |
|  | s2 | s2: 25.71, 25.75 → **25.73** (38.87 ms a token) | — | +5.17 tok/s | +25.2 % | no class (one load) |
|  | h2 | h2: 20.77, 21.26 → **21.02** (47.58 ms a token) | — | +0.46 tok/s | +2.2 % | no class (one load) |
|  | g2 | g2: 8.90, 8.69 → **8.79** (113.74 ms a token) | — | -11.77 tok/s | -57.2 % | no class (one load) |

### Generation over all depths, tok/s

Every depth's tokens over their time, per load.

| Depth (tokens) | Arm | Per load | Spread | Against `h1` | | Class |
|---:|---|---|---:|---:|---:|---|
| all | h1 | h1: **23.66** | — | control | | |
|  | g1 | g1: **10.73** | — | -12.92 tok/s | -54.6 % | no class (one load) |
|  | s8 | s8: **28.19** | — | +4.53 tok/s | +19.2 % | no class (one load) |
|  | s4 | s4: **27.03** | — | +3.38 tok/s | +14.3 % | no class (one load) |
|  | s2 | s2: **26.00** | — | +2.34 tok/s | +9.9 % | no class (one load) |
|  | h2 | h2: **23.72** | — | +0.06 tok/s | +0.3 % | no class (one load) |
|  | g2 | g2: **10.64** | — | -13.02 tok/s | -55.0 % | no class (one load) |

### The step on pass 1 (the first request at a new depth), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `h1`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `h1` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | h1 | h1: 36.80 (256 steps) | — | control | | |
|  | g1 | g1: 73.47 (256 steps) | — | +36.66 ms | +99.6 % | no class (one load) |
|  | s8 | s8: 182.12 (49 steps) | — | +145.32 ms | +394.8 % | no class (one load) |
|  | s4 | s4: 77.72 (124 steps) | — | +40.92 ms | +111.2 % | no class (one load) |
|  | s2 | s2: 66.24 (151 steps) | — | +29.44 ms | +80.0 % | no class (one load) |
|  | h2 | h2: 38.56 (256 steps) | — | +1.76 ms | +4.8 % | no class (one load) |
|  | g2 | g2: 76.35 (256 steps) | — | +39.55 ms | +107.5 % | no class (one load) |
| 8,192 | h1 | h1: 49.34 (256 steps) | — | control | | |
|  | g1 | g1: 112.40 (256 steps) | — | +63.06 ms | +127.8 % | no class (one load) |
|  | s8 | s8: 264.65 (35 steps) | — | +215.31 ms | +436.4 % | no class (one load) |
|  | s4 | s4: 147.03 (65 steps) | — | +97.69 ms | +198.0 % | no class (one load) |
|  | s2 | s2: 74.86 (133 steps) | — | +25.52 ms | +51.7 % | no class (one load) |
|  | h2 | h2: 48.14 (256 steps) | — | -1.20 ms | -2.4 % | no class (one load) |
|  | g2 | g2: 112.33 (256 steps) | — | +63.00 ms | +127.7 % | no class (one load) |

### The step on pass 2 (the same request again), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `h1`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `h1` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | h1 | h1: 34.98 (256 steps) | — | control | | |
|  | g1 | g1: 71.71 (256 steps) | — | +36.73 ms | +105.0 % | no class (one load) |
|  | s8 | s8: 180.76 (49 steps) | — | +145.78 ms | +416.7 % | no class (one load) |
|  | s4 | s4: 73.70 (124 steps) | — | +38.72 ms | +110.7 % | no class (one load) |
|  | s2 | s2: 62.77 (151 steps) | — | +27.79 ms | +79.4 % | no class (one load) |
|  | h2 | h2: 34.90 (256 steps) | — | -0.08 ms | -0.2 % | no class (one load) |
|  | g2 | g2: 72.24 (256 steps) | — | +37.26 ms | +106.5 % | no class (one load) |
| 8,192 | h1 | h1: 47.95 (256 steps) | — | control | | |
|  | g1 | g1: 115.05 (256 steps) | — | +67.10 ms | +140.0 % | no class (one load) |
|  | s8 | s8: 265.10 (35 steps) | — | +217.15 ms | +452.9 % | no class (one load) |
|  | s4 | s4: 146.85 (65 steps) | — | +98.90 ms | +206.3 % | no class (one load) |
|  | s2 | s2: 74.76 (133 steps) | — | +26.81 ms | +55.9 % | no class (one load) |
|  | h2 | h2: 47.03 (256 steps) | — | -0.92 ms | -1.9 % | no class (one load) |
|  | g2 | g2: 115.14 (256 steps) | — | +67.19 ms | +140.1 % | no class (one load) |

### Prompt reading, tok/s

Per load: the tokens read at that depth over their time (a prime request reads a chunk on top of the cache; without primes each decode request reads its prompt), with the tokens read and the tokens reused. Requests that read under 32 tokens are left out.

| Depth (tokens) | Arm | Per load | Spread | Against `h1` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | h1 | h1: 85.2 (64 read on 0 cached) | — | control | | |
|  | g1 | g1: 73.1 (64 read on 0 cached) | — | -12.1 tok/s | -14.2 % | no class (one load) |
|  | s8 | s8: 76.2 (64 read on 0 cached) | — | -9.0 tok/s | -10.6 % | no class (one load) |
|  | s4 | s4: 75.3 (64 read on 0 cached) | — | -10.0 tok/s | -11.7 % | no class (one load) |
|  | s2 | s2: 61.1 (64 read on 0 cached) | — | -24.1 tok/s | -28.3 % | no class (one load) |
|  | h2 | h2: 85.1 (64 read on 0 cached) | — | -0.1 tok/s | -0.1 % | no class (one load) |
|  | g2 | g2: 79.0 (64 read on 0 cached) | — | -6.2 tok/s | -7.3 % | no class (one load) |
| 8,192 | h1 | h1: 244.4 (8128 read on 0 cached) | — | control | | |
|  | g1 | g1: 122.6 (8128 read on 0 cached) | — | -121.9 tok/s | -49.9 % | no class (one load) |
|  | s8 | s8: 217.9 (8128 read on 0 cached) | — | -26.6 tok/s | -10.9 % | no class (one load) |
|  | s4 | s4: 194.9 (8128 read on 0 cached) | — | -49.6 tok/s | -20.3 % | no class (one load) |
|  | s2 | s2: 207.7 (8128 read on 0 cached) | — | -36.7 tok/s | -15.0 % | no class (one load) |
|  | h2 | h2: 234.2 (8128 read on 0 cached) | — | -10.2 tok/s | -4.2 % | no class (one load) |
|  | g2 | g2: 117.8 (8128 read on 0 cached) | — | -126.7 tok/s | -51.8 % | no class (one load) |

### The draft and the step

Per load, pass 1: draft tokens accepted of drafted (the share), tokens a step (generated tokens over steps, a step being one pass of the main model: generated minus accepted), and the time of a step. Pass 2 is listed where its counters differ.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | h1 | h1: no draft, 1.00 a step, 36.8 ms a step |
|  | g1 | g1: no draft, 1.00 a step, 73.5 ms a step |
|  | s8 | s8: 207 of 207 (1.000), 5.22 a step, 182.1 ms a step |
|  | s4 | s4: 132 of 132 (1.000), 2.06 a step, 77.7 ms a step |
|  | s2 | s2: 105 of 105 (1.000), 1.70 a step, 66.2 ms a step |
|  | h2 | h2: no draft, 1.00 a step, 38.6 ms a step |
|  | g2 | g2: no draft, 1.00 a step, 76.4 ms a step |
| 8,192 | h1 | h1: no draft, 1.00 a step, 49.3 ms a step |
|  | g1 | g1: no draft, 1.00 a step, 112.4 ms a step |
|  | s8 | s8: 221 of 221 (1.000), 7.31 a step, 264.7 ms a step |
|  | s4 | s4: 191 of 191 (1.000), 3.94 a step, 147.0 ms a step |
|  | s2 | s2: 123 of 123 (1.000), 1.92 a step, 74.9 ms a step |
|  | h2 | h2: no draft, 1.00 a step, 48.1 ms a step |
|  | g2 | g2: no draft, 1.00 a step, 112.3 ms a step |

### Checks

- Times: every load started on a machine checked quiet (valid=1).
- Prompts: 64 ids at depth 0, 8192 ids at depth 8,192; the same ids on every load and pass.
- Tokens out: 256 on every timed request.
- Greedy reply at depth 0: **differs from `h1` in s8, s4, s2.**
- Greedy reply at depth 8,192: **differs from `h1` in s8, s4, s2.**
- Primes (read on cached, the cache's source): 8,192: 8128 on 0 (none).
- Graph reallocations of the target, requests with one per load: h1: 0 of 6 requests; g1: 0 of 6 requests; s8: 5 of 6 requests (38 same size, 6 changed); s4: 4 of 6 requests (36 same size, 4 changed); s2: 5 of 6 requests (38 same size, 5 changed); h2: 0 of 6 requests; g2: 0 of 6 requests. **`s8` reallocates where `h1` does not.** **`s4` reallocates where `h1` does not.** **`s2` reallocates where `h1` does not.**
