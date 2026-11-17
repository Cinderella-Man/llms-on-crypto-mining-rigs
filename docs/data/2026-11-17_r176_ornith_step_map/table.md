## The deep lane: w1h1, w1g1, w1h2, w1g2, w2h1, w2g1, w3h1, w3g1, w4h1, w4g1

Model file `Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf`, llama-server version: 1.0.0-dev (build 12832, commit eb312b7bce).

256 tokens per request at temperature 0, each request twice per load (pass 1 is the first request at a new depth, pass 2 the same request again). Timed loads, in their order: w1h1, w1g1, w1h2, w1g2, w2h1, w2g1, w3h1, w3g1, w4h1, w4g1.

| Arm | Loads | Draft | Context | Its own variables and flags |
|---|---|---|---:|---|
| w1h1 (control) | w1h1 | off | 49152 | `GGML_CUDA_PROFILE_HOST=<OUT>/hp-w1h1.csv LLAMA_STEP_TRACE=<OUT>/st-w1h1.jsonl` |
| w1g1 | w1g1 | off | 49152 | `GGML_CUDA_PROFILE=<OUT>/gp-w1g1.csv GGML_CUDA_PROFILE_MODE=group` |
| w1h2 | w1h2 | off | 49152 | `GGML_CUDA_PROFILE_HOST=<OUT>/hp-w1h2.csv LLAMA_STEP_TRACE=<OUT>/st-w1h2.jsonl` |
| w1g2 | w1g2 | off | 49152 | `GGML_CUDA_PROFILE=<OUT>/gp-w1g2.csv GGML_CUDA_PROFILE_MODE=group` |
| w2h1 | w2h1 | off | 16384 | `GGML_CUDA_PROFILE_HOST=<OUT>/hp-w2h1.csv LLAMA_STEP_TRACE=<OUT>/st-w2h1.jsonl`, `--spec-type draft-mtp --spec-draft-n-max 1 -ts 6,6,6,6,6,5,5,5,5` |
| w2g1 | w2g1 | off | 16384 | `GGML_CUDA_PROFILE=<OUT>/gp-w2g1.csv GGML_CUDA_PROFILE_MODE=group`, `--spec-type draft-mtp --spec-draft-n-max 1 -ts 6,6,6,6,6,5,5,5,5` |
| w3h1 | w3h1 | off | 16384 | `GGML_CUDA_PROFILE_HOST=<OUT>/hp-w3h1.csv LLAMA_STEP_TRACE=<OUT>/st-w3h1.jsonl`, `--spec-type draft-mtp --spec-draft-n-max 2 -ts 6,6,6,6,6,5,5,5,5` |
| w3g1 | w3g1 | off | 16384 | `GGML_CUDA_PROFILE=<OUT>/gp-w3g1.csv GGML_CUDA_PROFILE_MODE=group`, `--spec-type draft-mtp --spec-draft-n-max 2 -ts 6,6,6,6,6,5,5,5,5` |
| w4h1 | w4h1 | off | 16384 | `GGML_CUDA_PROFILE_HOST=<OUT>/hp-w4h1.csv LLAMA_STEP_TRACE=<OUT>/st-w4h1.jsonl`, `--spec-type draft-mtp --spec-draft-n-max 3 -ts 6,6,6,6,6,5,5,5,5` |
| w4g1 | w4g1 | off | 16384 | `GGML_CUDA_PROFILE=<OUT>/gp-w4g1.csv GGML_CUDA_PROFILE_MODE=group`, `--spec-type draft-mtp --spec-draft-n-max 3 -ts 6,6,6,6,6,5,5,5,5` |

### Generation, tok/s

Per load: pass 1, pass 2 → the load, with the load's time per token. A load's speed is its two passes' tokens over their time. Spread: the difference between the arm's fastest and slowest load. Class against `w1h1`: gain when the mean is above the control's by more than the larger of the two spreads, loss when below by more than it, else negligible.

| Depth (tokens) | Arm | Per load | Spread | Against `w1h1` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | w1h1 | w1h1: 18.23, 18.62 → **18.43** (54.27 ms a token) | — | control | | |
|  | w1g1 | w1g1: 14.23, 14.45 → **14.34** (69.75 ms a token) | — | -4.09 tok/s | -22.2 % | no class (one load) |
|  | w1h2 | w1h2: 18.07, 18.62 → **18.34** (54.53 ms a token) | — | -0.09 tok/s | -0.5 % | no class (one load) |
|  | w1g2 | w1g2: 14.21, 14.45 → **14.33** (69.79 ms a token) | — | -4.10 tok/s | -22.2 % | no class (one load) |
|  | w2h1 | w2h1: 17.16, 17.59 → **17.37** (57.57 ms a token) | — | -1.06 tok/s | -5.7 % | no class (one load) |
|  | w2g1 | w2g1: 14.49, 14.74 → **14.61** (68.44 ms a token) | — | -3.82 tok/s | -20.7 % | no class (one load) |
|  | w3h1 | w3h1: 17.32, 17.94 → **17.63** (56.73 ms a token) | — | -0.80 tok/s | -4.3 % | no class (one load) |
|  | w3g1 | w3g1: 15.30, 15.58 → **15.43** (64.79 ms a token) | — | -2.99 tok/s | -16.2 % | no class (one load) |
|  | w4h1 | w4h1: 15.77, 16.31 → **16.04** (62.36 ms a token) | — | -2.39 tok/s | -13.0 % | no class (one load) |
|  | w4g1 | w4g1: 14.28, 14.44 → **14.36** (69.63 ms a token) | — | -4.07 tok/s | -22.1 % | no class (one load) |
| 8,192 | w1h1 | w1h1: 15.51, 15.54 → **15.53** (64.41 ms a token) | — | control | | |
|  | w1g1 | w1g1: 12.45, 12.45 → **12.45** (80.31 ms a token) | — | -3.07 tok/s | -19.8 % | no class (one load) |
|  | w1h2 | w1h2: 15.50, 15.54 → **15.52** (64.43 ms a token) | — | -0.01 tok/s | -0.0 % | no class (one load) |
|  | w1g2 | w1g2: 12.44, 12.43 → **12.44** (80.39 ms a token) | — | -3.09 tok/s | -19.9 % | no class (one load) |
|  | w2h1 | w2h1: 15.34, 15.47 → **15.40** (64.92 ms a token) | — | -0.12 tok/s | -0.8 % | no class (one load) |
|  | w2g1 | w2g1: 13.06, 13.04 → **13.05** (76.62 ms a token) | — | -2.48 tok/s | -15.9 % | no class (one load) |
|  | w3h1 | w3h1: 15.54, 15.72 → **15.63** (63.98 ms a token) | — | +0.10 tok/s | +0.7 % | no class (one load) |
|  | w3g1 | w3g1: 13.75, 13.72 → **13.73** (72.81 ms a token) | — | -1.79 tok/s | -11.5 % | no class (one load) |
|  | w4h1 | w4h1: 15.10, 15.29 → **15.19** (65.82 ms a token) | — | -0.33 tok/s | -2.1 % | no class (one load) |
|  | w4g1 | w4g1: 13.56, 13.56 → **13.56** (73.75 ms a token) | — | -1.97 tok/s | -12.7 % | no class (one load) |

### Generation over all depths, tok/s

Every depth's tokens over their time, per load.

| Depth (tokens) | Arm | Per load | Spread | Against `w1h1` | | Class |
|---:|---|---|---:|---:|---:|---|
| all | w1h1 | w1h1: **16.85** | — | control | | |
|  | w1g1 | w1g1: **13.33** | — | -3.52 tok/s | -20.9 % | no class (one load) |
|  | w1h2 | w1h2: **16.81** | — | -0.04 tok/s | -0.2 % | no class (one load) |
|  | w1g2 | w1g2: **13.32** | — | -3.54 tok/s | -21.0 % | no class (one load) |
|  | w2h1 | w2h1: **16.33** | — | -0.53 tok/s | -3.1 % | no class (one load) |
|  | w2g1 | w2g1: **13.79** | — | -3.07 tok/s | -18.2 % | no class (one load) |
|  | w3h1 | w3h1: **16.57** | — | -0.28 tok/s | -1.7 % | no class (one load) |
|  | w3g1 | w3g1: **14.54** | — | -2.32 tok/s | -13.8 % | no class (one load) |
|  | w4h1 | w4h1: **15.60** | — | -1.25 tok/s | -7.4 % | no class (one load) |
|  | w4g1 | w4g1: **13.95** | — | -2.90 tok/s | -17.2 % | no class (one load) |

### The step on pass 1 (the first request at a new depth), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `w1h1`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `w1h1` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | w1h1 | w1h1: 54.84 (256 steps) | — | control | | |
|  | w1g1 | w1g1: 70.28 (256 steps) | — | +15.44 ms | +28.1 % | no class (one load) |
|  | w1h2 | w1h2: 55.35 (256 steps) | — | +0.51 ms | +0.9 % | no class (one load) |
|  | w1g2 | w1g2: 70.38 (256 steps) | — | +15.53 ms | +28.3 % | no class (one load) |
|  | w2h1 | w2h1: 110.54 (135 steps) | — | +55.69 ms | +101.5 % | no class (one load) |
|  | w2g1 | w2g1: 130.90 (135 steps) | — | +76.06 ms | +138.7 % | no class (one load) |
|  | w3h1 | w3h1: 147.78 (100 steps) | — | +92.94 ms | +169.5 % | no class (one load) |
|  | w3g1 | w3g1: 167.37 (100 steps) | — | +112.53 ms | +205.2 % | no class (one load) |
|  | w4h1 | w4h1: 178.37 (91 steps) | — | +123.52 ms | +225.2 % | no class (one load) |
|  | w4g1 | w4g1: 196.94 (91 steps) | — | +142.09 ms | +259.1 % | no class (one load) |
| 8,192 | w1h1 | w1h1: 64.49 (256 steps) | — | control | | |
|  | w1g1 | w1g1: 80.30 (256 steps) | — | +15.81 ms | +24.5 % | no class (one load) |
|  | w1h2 | w1h2: 64.52 (256 steps) | — | +0.04 ms | +0.1 % | no class (one load) |
|  | w1g2 | w1g2: 80.35 (256 steps) | — | +15.87 ms | +24.6 % | no class (one load) |
|  | w2h1 | w2h1: 123.61 (135 steps) | — | +59.13 ms | +91.7 % | no class (one load) |
|  | w2g1 | w2g1: 145.19 (135 steps) | — | +80.71 ms | +125.2 % | no class (one load) |
|  | w3h1 | w3h1: 168.13 (98 steps) | — | +103.65 ms | +160.7 % | no class (one load) |
|  | w3g1 | w3g1: 190.02 (98 steps) | — | +125.53 ms | +194.7 % | no class (one load) |
|  | w4h1 | w4h1: 201.85 (84 steps) | — | +137.36 ms | +213.0 % | no class (one load) |
|  | w4g1 | w4g1: 224.70 (84 steps) | — | +160.21 ms | +248.4 % | no class (one load) |

### The step on pass 2 (the same request again), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `w1h1`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `w1h1` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | w1h1 | w1h1: 53.69 (256 steps) | — | control | | |
|  | w1g1 | w1g1: 69.22 (256 steps) | — | +15.53 ms | +28.9 % | no class (one load) |
|  | w1h2 | w1h2: 53.71 (256 steps) | — | +0.02 ms | +0.0 % | no class (one load) |
|  | w1g2 | w1g2: 69.21 (256 steps) | — | +15.51 ms | +28.9 % | no class (one load) |
|  | w2h1 | w2h1: 107.80 (135 steps) | — | +54.11 ms | +100.8 % | no class (one load) |
|  | w2g1 | w2g1: 128.67 (135 steps) | — | +74.97 ms | +139.6 % | no class (one load) |
|  | w3h1 | w3h1: 142.69 (100 steps) | — | +88.99 ms | +165.7 % | no class (one load) |
|  | w3g1 | w3g1: 164.35 (100 steps) | — | +110.66 ms | +206.1 % | no class (one load) |
|  | w4h1 | w4h1: 172.49 (91 steps) | — | +118.80 ms | +221.3 % | no class (one load) |
|  | w4g1 | w4g1: 194.84 (91 steps) | — | +141.15 ms | +262.9 % | no class (one load) |
| 8,192 | w1h1 | w1h1: 64.33 (256 steps) | — | control | | |
|  | w1g1 | w1g1: 80.32 (256 steps) | — | +15.99 ms | +24.9 % | no class (one load) |
|  | w1h2 | w1h2: 64.34 (256 steps) | — | +0.01 ms | +0.0 % | no class (one load) |
|  | w1g2 | w1g2: 80.43 (256 steps) | — | +16.10 ms | +25.0 % | no class (one load) |
|  | w2h1 | w2h1: 122.62 (135 steps) | — | +58.29 ms | +90.6 % | no class (one load) |
|  | w2g1 | w2g1: 145.41 (135 steps) | — | +81.08 ms | +126.0 % | no class (one load) |
|  | w3h1 | w3h1: 166.12 (98 steps) | — | +101.79 ms | +158.2 % | no class (one load) |
|  | w3g1 | w3g1: 190.36 (98 steps) | — | +126.03 ms | +195.9 % | no class (one load) |
|  | w4h1 | w4h1: 199.33 (84 steps) | — | +135.00 ms | +209.9 % | no class (one load) |
|  | w4g1 | w4g1: 224.83 (84 steps) | — | +160.50 ms | +249.5 % | no class (one load) |

### Prompt reading, tok/s

Per load: the tokens read at that depth over their time (a prime request reads a chunk on top of the cache; without primes each decode request reads its prompt), with the tokens read and the tokens reused. Requests that read under 32 tokens are left out.

| Depth (tokens) | Arm | Per load | Spread | Against `w1h1` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | w1h1 | w1h1: 40.7 (64 read on 0 cached) | — | control | | |
|  | w1g1 | w1g1: 37.1 (64 read on 0 cached) | — | -3.6 tok/s | -8.9 % | no class (one load) |
|  | w1h2 | w1h2: 40.7 (64 read on 0 cached) | — | +0.0 tok/s | +0.0 % | no class (one load) |
|  | w1g2 | w1g2: 37.3 (64 read on 0 cached) | — | -3.4 tok/s | -8.3 % | no class (one load) |
|  | w2h1 | w2h1: 32.5 (64 read on 0 cached) | — | -8.2 tok/s | -20.1 % | no class (one load) |
|  | w2g1 | w2g1: 28.8 (64 read on 0 cached) | — | -11.9 tok/s | -29.2 % | no class (one load) |
|  | w3h1 | w3h1: 33.6 (64 read on 0 cached) | — | -7.1 tok/s | -17.5 % | no class (one load) |
|  | w3g1 | w3g1: 31.3 (64 read on 0 cached) | — | -9.4 tok/s | -23.0 % | no class (one load) |
|  | w4h1 | w4h1: 32.4 (64 read on 0 cached) | — | -8.3 tok/s | -20.3 % | no class (one load) |
|  | w4g1 | w4g1: 30.3 (64 read on 0 cached) | — | -10.4 tok/s | -25.6 % | no class (one load) |
| 8,192 | w1h1 | w1h1: 118.9 (8128 read on 0 cached) | — | control | | |
|  | w1g1 | w1g1: 60.8 (8128 read on 0 cached) | — | -58.0 tok/s | -48.8 % | no class (one load) |
|  | w1h2 | w1h2: 118.9 (8128 read on 0 cached) | — | +0.0 tok/s | +0.0 % | no class (one load) |
|  | w1g2 | w1g2: 61.2 (8128 read on 0 cached) | — | -57.6 tok/s | -48.5 % | no class (one load) |
|  | w2h1 | w2h1: 83.1 (8128 read on 0 cached) | — | -35.7 tok/s | -30.0 % | no class (one load) |
|  | w2g1 | w2g1: 45.8 (8128 read on 0 cached) | — | -73.1 tok/s | -61.5 % | no class (one load) |
|  | w3h1 | w3h1: 83.2 (8128 read on 0 cached) | — | -35.6 tok/s | -30.0 % | no class (one load) |
|  | w3g1 | w3g1: 45.9 (8128 read on 0 cached) | — | -73.0 tok/s | -61.4 % | no class (one load) |
|  | w4h1 | w4h1: 82.9 (8128 read on 0 cached) | — | -35.9 tok/s | -30.2 % | no class (one load) |
|  | w4g1 | w4g1: 45.9 (8128 read on 0 cached) | — | -72.9 tok/s | -61.3 % | no class (one load) |

### The draft and the step

Per load, pass 1: draft tokens accepted of drafted (the share), tokens a step (generated tokens over steps, a step being one pass of the main model: generated minus accepted), and the time of a step. Pass 2 is listed where its counters differ.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | w1h1 | w1h1: no draft, 1.00 a step, 54.8 ms a step |
|  | w1g1 | w1g1: no draft, 1.00 a step, 70.3 ms a step |
|  | w1h2 | w1h2: no draft, 1.00 a step, 55.4 ms a step |
|  | w1g2 | w1g2: no draft, 1.00 a step, 70.4 ms a step |
|  | w2h1 | w2h1: 121 of 134 (0.903), 1.90 a step, 110.5 ms a step |
|  | w2g1 | w2g1: 121 of 134 (0.903), 1.90 a step, 130.9 ms a step |
|  | w3h1 | w3h1: 156 of 197 (0.792), 2.56 a step, 147.8 ms a step |
|  | w3g1 | w3g1: 156 of 197 (0.792), 2.56 a step, 167.4 ms a step |
|  | w4h1 | w4h1: 165 of 265 (0.623), 2.81 a step, 178.4 ms a step |
|  | w4g1 | w4g1: 165 of 265 (0.623), 2.81 a step, 196.9 ms a step |
| 8,192 | w1h1 | w1h1: no draft, 1.00 a step, 64.5 ms a step |
|  | w1g1 | w1g1: no draft, 1.00 a step, 80.3 ms a step |
|  | w1h2 | w1h2: no draft, 1.00 a step, 64.5 ms a step |
|  | w1g2 | w1g2: no draft, 1.00 a step, 80.4 ms a step |
|  | w2h1 | w2h1: 121 of 133 (0.910), 1.90 a step, 123.6 ms a step |
|  | w2g1 | w2g1: 121 of 133 (0.910), 1.90 a step, 145.2 ms a step |
|  | w3h1 | w3h1: 158 of 193 (0.819), 2.61 a step, 168.1 ms a step |
|  | w3g1 | w3g1: 158 of 193 (0.819), 2.61 a step, 190.0 ms a step |
|  | w4h1 | w4h1: 172 of 247 (0.696), 3.05 a step, 201.8 ms a step |
|  | w4g1 | w4g1: 172 of 247 (0.696), 3.05 a step, 224.7 ms a step |

### Checks

- Times: every load started on a machine checked quiet (valid=1).
- Prompts: 64 ids at depth 0, 8192 ids at depth 8,192; the same ids on every load and pass.
- Tokens out: 256 on every timed request.
- Greedy reply at depth 0: **differs from `w1h1` in w4h1, w4g1.**
- Greedy reply at depth 8,192: **differs from `w1h1` in w2h1, w2g1, w3h1, w3g1, w4h1, w4g1.**
- Primes (read on cached, the cache's source): 8,192: 8128 on 0 (none).
- Graph reallocations of the target, requests with one per load: w1h1: 0 of 6 requests; w1g1: 0 of 6 requests; w1h2: 0 of 6 requests; w1g2: 0 of 6 requests; w2h1: 0 of 6 requests; w2g1: 0 of 6 requests; w3h1: 0 of 6 requests; w3g1: 0 of 6 requests; w4h1: 0 of 6 requests; w4g1: 0 of 6 requests.
