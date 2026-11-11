## The deep lane: dev, stage, check, hostdev, hoststage

Model file `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf`, llama-server version: 0.9.0-dev (build 12581, commit c408dabcb3).

256 tokens per request at temperature 0, each request twice per load (pass 1 is the first request at a new depth, pass 2 the same request again). Timed loads, in their order: dev, stage, check, hostdev, hoststage.

| Arm | Loads | Draft | Context | Its own variables and flags |
|---|---|---|---:|---|
| dev (control) | dev | on | 65536 | `LLAMA_QSA_KPOOL_DEV=1` |
| stage | stage | on | 65536 | `LLAMA_QSA_KPOOL_DEV=1 LLAMA_QSA_KPOOL_STAGE=1` |
| check | check | on | 65536 | `LLAMA_QSA_KPOOL_DEV=1 LLAMA_QSA_KPOOL_STAGE=1 LLAMA_QSA_KPOOL_DEV_CHECK=1`, `-lv 4` |
| hostdev | hostdev | on | 65536 | `LLAMA_QSA_KPOOL_DEV=1 LLAMA_QSA_INPUT_STATS=1 GGML_CUDA_PROFILE_HOST=$OUT/hostdev.csv`, `-lv 4` |
| hoststage | hoststage | on | 65536 | `LLAMA_QSA_KPOOL_DEV=1 LLAMA_QSA_KPOOL_STAGE=1 LLAMA_QSA_INPUT_STATS=1 GGML_CUDA_PROFILE_HOST=$OUT/hoststage.csv`, `-lv 4` |

### Generation, tok/s

Per load: pass 1, pass 2 → the load, with the load's time per token. A load's speed is its two passes' tokens over their time. Spread: the difference between the arm's fastest and slowest load. Class against `dev`: gain when the mean is above the control's by more than the larger of the two spreads, loss when below by more than it, else negligible.

| Depth (tokens) | Arm | Per load | Spread | Against `dev` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | dev | dev: 26.07, 28.09 → **27.04** (36.98 ms a token) | — | control | | |
|  | stage | stage: 26.13, 28.08 → **27.07** (36.94 ms a token) | — | +0.03 tok/s | +0.1 % | no class (one load) |
|  | check | check: 22.56, 23.69 → **23.11** (43.27 ms a token) | — | -3.93 tok/s | -14.5 % | no class (one load) |
|  | hostdev | hostdev: 25.98, 27.94 → **26.93** (37.14 ms a token) | — | -0.11 tok/s | -0.4 % | no class (one load) |
|  | hoststage | hoststage: 25.90, 27.98 → **26.90** (37.18 ms a token) | — | -0.14 tok/s | -0.5 % | no class (one load) |
| 16,384 | dev | dev: 21.50, 24.22 → **22.78** (43.90 ms a token) | — | control | | |
|  | stage | stage: 21.53, 24.32 → **22.84** (43.79 ms a token) | — | +0.06 tok/s | +0.2 % | no class (one load) |
|  | check | check: 19.11, 21.07 → **20.04** (49.89 ms a token) | — | -2.74 tok/s | -12.0 % | no class (one load) |
|  | hostdev | hostdev: 20.61, 24.18 → **22.25** (44.94 ms a token) | — | -0.53 tok/s | -2.3 % | no class (one load) |
|  | hoststage | hoststage: 21.49, 24.25 → **22.79** (43.89 ms a token) | — | +0.01 tok/s | +0.0 % | no class (one load) |
| 49,152 | dev | dev: 16.57, 19.88 → **18.07** (55.33 ms a token) | — | control | | |
|  | stage | stage: 16.57, 19.89 → **18.08** (55.31 ms a token) | — | +0.00 tok/s | +0.0 % | no class (one load) |
|  | check | check: 12.20, 17.47 → **14.37** (69.61 ms a token) | — | -3.71 tok/s | -20.5 % | no class (one load) |
|  | hostdev | hostdev: 15.63, 19.85 → **17.49** (57.17 ms a token) | — | -0.58 tok/s | -3.2 % | no class (one load) |
|  | hoststage | hoststage: 16.58, 19.92 → **18.10** (55.26 ms a token) | — | +0.02 tok/s | +0.1 % | no class (one load) |

### Generation over all depths, tok/s

Every depth's tokens over their time, per load.

| Depth (tokens) | Arm | Per load | Spread | Against `dev` | | Class |
|---:|---|---|---:|---:|---:|---|
| all | dev | dev: **22.03** | — | control | | |
|  | stage | stage: **22.05** | — | +0.03 tok/s | +0.1 % | no class (one load) |
|  | check | check: **18.43** | — | -3.59 tok/s | -16.3 % | no class (one load) |
|  | hostdev | hostdev: **21.54** | — | -0.48 tok/s | -2.2 % | no class (one load) |
|  | hoststage | hoststage: **22.01** | — | -0.02 tok/s | -0.1 % | no class (one load) |

### The step on pass 1 (the first request at a new depth), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `dev`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `dev` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | dev | dev: 125.89 (78 steps) | — | control | | |
|  | stage | stage: 125.60 (78 steps) | — | -0.29 ms | -0.2 % | no class (one load) |
|  | check | check: 145.47 (78 steps) | — | +19.58 ms | +15.6 % | no class (one load) |
|  | hostdev | hostdev: 126.31 (78 steps) | — | +0.42 ms | +0.3 % | no class (one load) |
|  | hoststage | hoststage: 126.72 (78 steps) | — | +0.83 ms | +0.7 % | no class (one load) |
| 16,384 | dev | dev: 165.40 (72 steps) | — | control | | |
|  | stage | stage: 165.18 (72 steps) | — | -0.23 ms | -0.1 % | no class (one load) |
|  | check | check: 186.01 (72 steps) | — | +20.61 ms | +12.5 % | no class (one load) |
|  | hostdev | hostdev: 172.55 (72 steps) | — | +7.14 ms | +4.3 % | no class (one load) |
|  | hoststage | hoststage: 165.44 (72 steps) | — | +0.04 ms | +0.0 % | no class (one load) |
| 49,152 | dev | dev: 208.81 (74 steps) | — | control | | |
|  | stage | stage: 208.81 (74 steps) | — | +0.01 ms | +0.0 % | no class (one load) |
|  | check | check: 283.67 (74 steps) | — | +74.86 ms | +35.9 % | no class (one load) |
|  | hostdev | hostdev: 221.29 (74 steps) | — | +12.49 ms | +6.0 % | no class (one load) |
|  | hoststage | hoststage: 208.64 (74 steps) | — | -0.17 ms | -0.1 % | no class (one load) |

### The step on pass 2 (the same request again), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `dev`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `dev` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | dev | dev: 121.52 (75 steps) | — | control | | |
|  | stage | stage: 121.54 (75 steps) | — | +0.02 ms | +0.0 % | no class (one load) |
|  | check | check: 144.07 (75 steps) | — | +22.55 ms | +18.6 % | no class (one load) |
|  | hostdev | hostdev: 122.15 (75 steps) | — | +0.63 ms | +0.5 % | no class (one load) |
|  | hoststage | hoststage: 122.01 (75 steps) | — | +0.49 ms | +0.4 % | no class (one load) |
| 16,384 | dev | dev: 146.77 (72 steps) | — | control | | |
|  | stage | stage: 146.22 (72 steps) | — | -0.55 ms | -0.4 % | no class (one load) |
|  | check | check: 168.79 (72 steps) | — | +22.02 ms | +15.0 % | no class (one load) |
|  | hostdev | hostdev: 147.05 (72 steps) | — | +0.27 ms | +0.2 % | no class (one load) |
|  | hoststage | hoststage: 146.64 (72 steps) | — | -0.14 ms | -0.1 % | no class (one load) |
| 49,152 | dev | dev: 174.00 (74 steps) | — | control | | |
|  | stage | stage: 173.89 (74 steps) | — | -0.11 ms | -0.1 % | no class (one load) |
|  | check | check: 197.98 (74 steps) | — | +23.98 ms | +13.8 % | no class (one load) |
|  | hostdev | hostdev: 174.24 (74 steps) | — | +0.24 ms | +0.1 % | no class (one load) |
|  | hoststage | hoststage: 173.71 (74 steps) | — | -0.29 ms | -0.2 % | no class (one load) |

### Prompt reading, tok/s

Per load: the tokens read at that depth over their time (a prime request reads a chunk on top of the cache; without primes each decode request reads its prompt), with the tokens read and the tokens reused. Requests that read under 32 tokens are left out.

| Depth (tokens) | Arm | Per load | Spread | Against `dev` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | dev | dev: 32.5 (64 read on 0 cached) | — | control | | |
|  | stage | stage: 33.0 (64 read on 0 cached) | — | +0.5 tok/s | +1.6 % | no class (one load) |
|  | check | check: 31.9 (64 read on 0 cached) | — | -0.6 tok/s | -1.9 % | no class (one load) |
|  | hostdev | hostdev: 33.0 (64 read on 0 cached) | — | +0.5 tok/s | +1.4 % | no class (one load) |
|  | hoststage | hoststage: 32.5 (64 read on 0 cached) | — | +0.0 tok/s | +0.1 % | no class (one load) |
| 16,384 | dev | dev: 130.4 (16320 read on 0 cached) | — | control | | |
|  | stage | stage: 128.3 (16320 read on 0 cached) | — | -2.1 tok/s | -1.6 % | no class (one load) |
|  | check | check: 68.5 (16320 read on 0 cached) | — | -61.9 tok/s | -47.5 % | no class (one load) |
|  | hostdev | hostdev: 114.9 (16320 read on 0 cached) | — | -15.5 tok/s | -11.9 % | no class (one load) |
|  | hoststage | hoststage: 115.0 (16320 read on 0 cached) | — | -15.4 tok/s | -11.8 % | no class (one load) |
| 49,152 | dev | dev: 82.6 (33284 read on 15804 cached) | — | control | | |
|  | stage | stage: 82.6 (33284 read on 15804 cached) | — | +0.1 tok/s | +0.1 % | no class (one load) |
|  | check | check: 42.7 (33284 read on 15804 cached) | — | -39.9 tok/s | -48.3 % | no class (one load) |
|  | hostdev | hostdev: 73.6 (33284 read on 15804 cached) | — | -8.9 tok/s | -10.8 % | no class (one load) |
|  | hoststage | hoststage: 73.7 (33284 read on 15804 cached) | — | -8.8 tok/s | -10.7 % | no class (one load) |

### The draft and the step

Per load, pass 1: draft tokens accepted of drafted (the share), tokens a step (generated tokens over steps, a step being one pass of the main model: generated minus accepted), and the time of a step. Pass 2 is listed where its counters differ.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | dev | dev: 178 of 229 (0.777), 3.28 a step, 125.9 ms a step (pass 2: 181 of 220 (0.823), 3.41 a step, 121.5 ms a step) |
|  | stage | stage: 178 of 229 (0.777), 3.28 a step, 125.6 ms a step (pass 2: 181 of 220 (0.823), 3.41 a step, 121.5 ms a step) |
|  | check | check: 178 of 229 (0.777), 3.28 a step, 145.5 ms a step (pass 2: 181 of 220 (0.823), 3.41 a step, 144.1 ms a step) |
|  | hostdev | hostdev: 178 of 229 (0.777), 3.28 a step, 126.3 ms a step (pass 2: 181 of 220 (0.823), 3.41 a step, 122.2 ms a step) |
|  | hoststage | hoststage: 178 of 229 (0.777), 3.28 a step, 126.7 ms a step (pass 2: 181 of 220 (0.823), 3.41 a step, 122.0 ms a step) |
| 16,384 | dev | dev: 184 of 210 (0.876), 3.56 a step, 165.4 ms a step |
|  | stage | stage: 184 of 210 (0.876), 3.56 a step, 165.2 ms a step |
|  | check | check: 184 of 210 (0.876), 3.56 a step, 186.0 ms a step |
|  | hostdev | hostdev: 184 of 210 (0.876), 3.56 a step, 172.5 ms a step |
|  | hoststage | hoststage: 184 of 210 (0.876), 3.56 a step, 165.4 ms a step |
| 49,152 | dev | dev: 182 of 218 (0.835), 3.46 a step, 208.8 ms a step |
|  | stage | stage: 182 of 218 (0.835), 3.46 a step, 208.8 ms a step |
|  | check | check: 182 of 218 (0.835), 3.46 a step, 283.7 ms a step |
|  | hostdev | hostdev: 182 of 218 (0.835), 3.46 a step, 221.3 ms a step |
|  | hoststage | hoststage: 182 of 218 (0.835), 3.46 a step, 208.6 ms a step |

### Checks

- Times: every load started on a machine checked quiet (valid=1).
- Prompts: 64 ids at depth 0, 16384 ids at depth 16,384, 49152 ids at depth 49,152; the same ids on every load and pass.
- Tokens out: 256 on every timed request.
- Greedy reply at depth 0: the same on every pass and load of every arm.
- Greedy reply at depth 16,384: the same on every pass and load of every arm.
- Greedy reply at depth 49,152: the same on every pass and load of every arm.
- Primes (read on cached, the cache's source): 16,384: 16320 on 0 (none); 49,152: 33284 on 15804 (checkpoint).
