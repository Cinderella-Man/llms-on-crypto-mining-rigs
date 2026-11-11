## The deep lane: kpool, mask, check, hostmask

Model file `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf`, llama-server version: 0.9.0-dev (build 12581, commit c408dabcb3).

256 tokens per request at temperature 0, each request twice per load (pass 1 is the first request at a new depth, pass 2 the same request again). Timed loads, in their order: kpool, mask, check, hostmask.

| Arm | Loads | Draft | Context | Its own variables and flags |
|---|---|---|---:|---|
| kpool (control) | kpool | on | 65536 | `LLAMA_QSA_KPOOL_DEV=1` |
| mask | mask | on | 65536 | `LLAMA_QSA_KPOOL_DEV=1 LLAMA_QSA_POOL_MASK_DEV=1` |
| check | check | on | 65536 | `LLAMA_QSA_KPOOL_DEV=1 LLAMA_QSA_POOL_MASK_DEV=1 LLAMA_QSA_KPOOL_DEV_CHECK=1 LLAMA_QSA_POOL_MASK_DEV_CHECK=1`, `-lv 4` |
| hostmask | hostmask | on | 65536 | `LLAMA_QSA_KPOOL_DEV=1 LLAMA_QSA_POOL_MASK_DEV=1 LLAMA_QSA_INPUT_STATS=1 GGML_CUDA_PROFILE_HOST=$OUT/hostmask.csv`, `-lv 4` |

### Generation, tok/s

Per load: pass 1, pass 2 → the load, with the load's time per token. A load's speed is its two passes' tokens over their time. Spread: the difference between the arm's fastest and slowest load. Class against `kpool`: gain when the mean is above the control's by more than the larger of the two spreads, loss when below by more than it, else negligible.

| Depth (tokens) | Arm | Per load | Spread | Against `kpool` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | kpool | kpool: 26.05, 27.94 → **26.96** (37.09 ms a token) | — | control | | |
|  | mask | mask: 26.13, 28.07 → **27.07** (36.94 ms a token) | — | +0.11 tok/s | +0.4 % | no class (one load) |
|  | check | check: 22.21, 23.29 → **22.73** (43.99 ms a token) | — | -4.23 tok/s | -15.7 % | no class (one load) |
|  | hostmask | hostmask: 26.02, 27.99 → **26.97** (37.08 ms a token) | — | +0.00 tok/s | +0.0 % | no class (one load) |
| 16,384 | kpool | kpool: 21.47, 24.24 → **22.77** (43.92 ms a token) | — | control | | |
|  | mask | mask: 18.18, 24.45 → **20.86** (47.95 ms a token) | — | -1.91 tok/s | -8.4 % | no class (one load) |
|  | check | check: 18.58, 20.40 → **19.45** (51.42 ms a token) | — | -3.32 tok/s | -14.6 % | no class (one load) |
|  | hostmask | hostmask: 21.62, 24.33 → **22.90** (43.67 ms a token) | — | +0.13 tok/s | +0.6 % | no class (one load) |
| 49,152 | kpool | kpool: 16.60, 19.90 → **18.10** (55.25 ms a token) | — | control | | |
|  | mask | mask: 16.74, 20.11 → **18.27** (54.73 ms a token) | — | +0.17 tok/s | +0.9 % | no class (one load) |
|  | check | check: 14.53, 16.67 → **15.53** (64.39 ms a token) | — | -2.57 tok/s | -14.2 % | no class (one load) |
|  | hostmask | hostmask: 16.71, 20.02 → **18.21** (54.91 ms a token) | — | +0.11 tok/s | +0.6 % | no class (one load) |

### Generation over all depths, tok/s

Every depth's tokens over their time, per load.

| Depth (tokens) | Arm | Per load | Spread | Against `kpool` | | Class |
|---:|---|---|---:|---:|---:|---|
| all | kpool | kpool: **22.02** | — | control | | |
|  | mask | mask: **21.49** | — | -0.53 tok/s | -2.4 % | no class (one load) |
|  | check | check: **18.77** | — | -3.24 tok/s | -14.7 % | no class (one load) |
|  | hostmask | hostmask: **22.11** | — | +0.10 tok/s | +0.4 % | no class (one load) |

### The step on pass 1 (the first request at a new depth), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `kpool`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `kpool` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | kpool | kpool: 125.98 (78 steps) | — | control | | |
|  | mask | mask: 125.58 (78 steps) | — | -0.40 ms | -0.3 % | no class (one load) |
|  | check | check: 147.80 (78 steps) | — | +21.82 ms | +17.3 % | no class (one load) |
|  | hostmask | hostmask: 126.15 (78 steps) | — | +0.17 ms | +0.1 % | no class (one load) |
| 16,384 | kpool | kpool: 165.63 (72 steps) | — | control | | |
|  | mask | mask: 195.55 (72 steps) | — | +29.91 ms | +18.1 % | no class (one load) |
|  | check | check: 191.38 (72 steps) | — | +25.75 ms | +15.5 % | no class (one load) |
|  | hostmask | hostmask: 164.44 (72 steps) | — | -1.19 ms | -0.7 % | no class (one load) |
| 49,152 | kpool | kpool: 208.38 (74 steps) | — | control | | |
|  | mask | mask: 206.63 (74 steps) | — | -1.74 ms | -0.8 % | no class (one load) |
|  | check | check: 238.02 (74 steps) | — | +29.64 ms | +14.2 % | no class (one load) |
|  | hostmask | hostmask: 207.07 (74 steps) | — | -1.31 ms | -0.6 % | no class (one load) |

### The step on pass 2 (the same request again), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `kpool`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `kpool` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | kpool | kpool: 122.18 (75 steps) | — | control | | |
|  | mask | mask: 121.60 (75 steps) | — | -0.58 ms | -0.5 % | no class (one load) |
|  | check | check: 146.56 (75 steps) | — | +24.38 ms | +20.0 % | no class (one load) |
|  | hostmask | hostmask: 121.96 (75 steps) | — | -0.22 ms | -0.2 % | no class (one load) |
| 16,384 | kpool | kpool: 146.70 (72 steps) | — | control | | |
|  | mask | mask: 145.43 (72 steps) | — | -1.27 ms | -0.9 % | no class (one load) |
|  | check | check: 174.26 (72 steps) | — | +27.56 ms | +18.8 % | no class (one load) |
|  | hostmask | hostmask: 146.11 (72 steps) | — | -0.59 ms | -0.4 % | no class (one load) |
| 49,152 | kpool | kpool: 173.87 (74 steps) | — | control | | |
|  | mask | mask: 172.04 (74 steps) | — | -1.83 ms | -1.1 % | no class (one load) |
|  | check | check: 207.47 (74 steps) | — | +33.60 ms | +19.3 % | no class (one load) |
|  | hostmask | hostmask: 172.82 (74 steps) | — | -1.05 ms | -0.6 % | no class (one load) |

### Prompt reading, tok/s

Per load: the tokens read at that depth over their time (a prime request reads a chunk on top of the cache; without primes each decode request reads its prompt), with the tokens read and the tokens reused. Requests that read under 32 tokens are left out.

| Depth (tokens) | Arm | Per load | Spread | Against `kpool` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | kpool | kpool: 32.9 (64 read on 0 cached) | — | control | | |
|  | mask | mask: 32.8 (64 read on 0 cached) | — | -0.1 tok/s | -0.2 % | no class (one load) |
|  | check | check: 32.2 (64 read on 0 cached) | — | -0.7 tok/s | -2.2 % | no class (one load) |
|  | hostmask | hostmask: 32.6 (64 read on 0 cached) | — | -0.3 tok/s | -0.8 % | no class (one load) |
| 16,384 | kpool | kpool: 131.6 (16320 read on 0 cached) | — | control | | |
|  | mask | mask: 132.0 (16320 read on 0 cached) | — | +0.5 tok/s | +0.4 % | no class (one load) |
|  | check | check: 68.5 (16320 read on 0 cached) | — | -63.0 tok/s | -47.9 % | no class (one load) |
|  | hostmask | hostmask: 117.2 (16320 read on 0 cached) | — | -14.4 tok/s | -10.9 % | no class (one load) |
| 49,152 | kpool | kpool: 82.3 (33284 read on 15804 cached) | — | control | | |
|  | mask | mask: 84.5 (33284 read on 15804 cached) | — | +2.1 tok/s | +2.6 % | no class (one load) |
|  | check | check: 41.8 (33284 read on 15804 cached) | — | -40.6 tok/s | -49.3 % | no class (one load) |
|  | hostmask | hostmask: 75.1 (33284 read on 15804 cached) | — | -7.2 tok/s | -8.7 % | no class (one load) |

### The draft and the step

Per load, pass 1: draft tokens accepted of drafted (the share), tokens a step (generated tokens over steps, a step being one pass of the main model: generated minus accepted), and the time of a step. Pass 2 is listed where its counters differ.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | kpool | kpool: 178 of 229 (0.777), 3.28 a step, 126.0 ms a step (pass 2: 181 of 220 (0.823), 3.41 a step, 122.2 ms a step) |
|  | mask | mask: 178 of 229 (0.777), 3.28 a step, 125.6 ms a step (pass 2: 181 of 220 (0.823), 3.41 a step, 121.6 ms a step) |
|  | check | check: 178 of 229 (0.777), 3.28 a step, 147.8 ms a step (pass 2: 181 of 220 (0.823), 3.41 a step, 146.6 ms a step) |
|  | hostmask | hostmask: 178 of 229 (0.777), 3.28 a step, 126.1 ms a step (pass 2: 181 of 220 (0.823), 3.41 a step, 122.0 ms a step) |
| 16,384 | kpool | kpool: 184 of 210 (0.876), 3.56 a step, 165.6 ms a step |
|  | mask | mask: 184 of 210 (0.876), 3.56 a step, 195.5 ms a step |
|  | check | check: 184 of 210 (0.876), 3.56 a step, 191.4 ms a step |
|  | hostmask | hostmask: 184 of 210 (0.876), 3.56 a step, 164.4 ms a step |
| 49,152 | kpool | kpool: 182 of 218 (0.835), 3.46 a step, 208.4 ms a step |
|  | mask | mask: 182 of 218 (0.835), 3.46 a step, 206.6 ms a step |
|  | check | check: 182 of 218 (0.835), 3.46 a step, 238.0 ms a step |
|  | hostmask | hostmask: 182 of 218 (0.835), 3.46 a step, 207.1 ms a step |

### Checks

- Times: every load started on a machine checked quiet (valid=1).
- Prompts: 64 ids at depth 0, 16384 ids at depth 16,384, 49152 ids at depth 49,152; the same ids on every load and pass.
- Tokens out: 256 on every timed request.
- Greedy reply at depth 0: the same on every pass and load of every arm.
- Greedy reply at depth 16,384: the same on every pass and load of every arm.
- Greedy reply at depth 49,152: the same on every pass and load of every arm.
- Primes (read on cached, the cache's source): 16,384: 16320 on 0 (none); 49,152: 33284 on 15804 (checkpoint).
