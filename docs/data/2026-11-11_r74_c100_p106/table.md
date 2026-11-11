## The deep lane: dev, check, host, hostdev

Model file `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf`, llama-server version: 0.9.0-dev (build 12581, commit c408dabcb3).

256 tokens per request at temperature 0, each request twice per load (pass 1 is the first request at a new depth, pass 2 the same request again). Timed loads, in their order: dev, check, host, hostdev.

| Arm | Loads | Draft | Context | Its own variables and flags |
|---|---|---|---:|---|
| dev (control) | dev | on | 65536 | `LLAMA_QSA_KPOOL_DEV=1` |
| check | check | on | 65536 | `LLAMA_QSA_KPOOL_DEV=1 LLAMA_QSA_KPOOL_DEV_CHECK=1`, `-lv 4` |
| host | host | on | 65536 | `LLAMA_QSA_INPUT_STATS=1 GGML_CUDA_PROFILE_HOST=$OUT/host.csv`, `-lv 4` |
| hostdev | hostdev | on | 65536 | `LLAMA_QSA_KPOOL_DEV=1 LLAMA_QSA_INPUT_STATS=1 GGML_CUDA_PROFILE_HOST=$OUT/hostdev.csv`, `-lv 4` |

### Generation, tok/s

Per load: pass 1, pass 2 → the load, with the load's time per token. A load's speed is its two passes' tokens over their time. Spread: the difference between the arm's fastest and slowest load. Class against `dev`: gain when the mean is above the control's by more than the larger of the two spreads, loss when below by more than it, else negligible.

| Depth (tokens) | Arm | Per load | Spread | Against `dev` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | dev | dev: 26.18, 28.06 → **27.08** (36.92 ms a token) | — | control | | |
|  | check | check: 22.56, 23.71 → **23.12** (43.25 ms a token) | — | -3.96 tok/s | -14.6 % | no class (one load) |
|  | host | host: 26.12, 27.97 → **27.01** (37.02 ms a token) | — | -0.07 tok/s | -0.3 % | no class (one load) |
|  | hostdev | hostdev: 26.06, 27.99 → **26.99** (37.05 ms a token) | — | -0.10 tok/s | -0.4 % | no class (one load) |
| 16,384 | dev | dev: 19.46, 24.25 → **21.59** (46.31 ms a token) | — | control | | |
|  | check | check: 19.09, 21.02 → **20.01** (49.98 ms a token) | — | -1.59 tok/s | -7.3 % | no class (one load) |
|  | host | host: 21.15, 23.90 → **22.44** (44.57 ms a token) | — | +0.84 tok/s | +3.9 % | no class (one load) |
|  | hostdev | hostdev: 21.47, 24.19 → **22.75** (43.96 ms a token) | — | +1.16 tok/s | +5.4 % | no class (one load) |
| 49,152 | dev | dev: 15.19, 19.89 → **17.22** (58.07 ms a token) | — | control | | |
|  | check | check: 15.12, 17.45 → **16.20** (61.72 ms a token) | — | -1.02 tok/s | -5.9 % | no class (one load) |
|  | host | host: 15.21, 19.19 → **16.97** (58.94 ms a token) | — | -0.25 tok/s | -1.5 % | no class (one load) |
|  | hostdev | hostdev: 13.99, 19.87 → **16.42** (60.91 ms a token) | — | -0.80 tok/s | -4.7 % | no class (one load) |

### Generation over all depths, tok/s

Every depth's tokens over their time, per load.

| Depth (tokens) | Arm | Per load | Spread | Against `dev` | | Class |
|---:|---|---|---:|---:|---:|---|
| all | dev | dev: **21.23** | — | control | | |
|  | check | check: **19.36** | — | -1.87 tok/s | -8.8 % | no class (one load) |
|  | host | host: **21.35** | — | +0.12 tok/s | +0.5 % | no class (one load) |
|  | hostdev | hostdev: **21.14** | — | -0.09 tok/s | -0.4 % | no class (one load) |

### The step on pass 1 (the first request at a new depth), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `dev`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `dev` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | dev | dev: 125.38 (78 steps) | — | control | | |
|  | check | check: 145.49 (78 steps) | — | +20.12 ms | +16.0 % | no class (one load) |
|  | host | host: 125.67 (78 steps) | — | +0.29 ms | +0.2 % | no class (one load) |
|  | hostdev | hostdev: 125.94 (78 steps) | — | +0.57 ms | +0.5 % | no class (one load) |
| 16,384 | dev | dev: 182.70 (72 steps) | — | control | | |
|  | check | check: 186.26 (72 steps) | — | +3.56 ms | +1.9 % | no class (one load) |
|  | host | host: 168.12 (72 steps) | — | -14.57 ms | -8.0 % | no class (one load) |
|  | hostdev | hostdev: 165.60 (72 steps) | — | -17.10 ms | -9.4 % | no class (one load) |
| 49,152 | dev | dev: 227.80 (74 steps) | — | control | | |
|  | check | check: 228.84 (74 steps) | — | +1.03 ms | +0.5 % | no class (one load) |
|  | host | host: 227.48 (74 steps) | — | -0.32 ms | -0.1 % | no class (one load) |
|  | hostdev | hostdev: 247.25 (74 steps) | — | +19.45 ms | +8.5 % | no class (one load) |

### The step on pass 2 (the same request again), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `dev`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `dev` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | dev | dev: 121.66 (75 steps) | — | control | | |
|  | check | check: 143.97 (75 steps) | — | +22.31 ms | +18.3 % | no class (one load) |
|  | host | host: 122.04 (75 steps) | — | +0.37 ms | +0.3 % | no class (one load) |
|  | hostdev | hostdev: 121.97 (75 steps) | — | +0.30 ms | +0.3 % | no class (one load) |
| 16,384 | dev | dev: 146.62 (72 steps) | — | control | | |
|  | check | check: 169.16 (72 steps) | — | +22.54 ms | +15.4 % | no class (one load) |
|  | host | host: 148.80 (72 steps) | — | +2.18 ms | +1.5 % | no class (one load) |
|  | hostdev | hostdev: 146.97 (72 steps) | — | +0.35 ms | +0.2 % | no class (one load) |
| 49,152 | dev | dev: 173.97 (74 steps) | — | control | | |
|  | check | check: 198.21 (74 steps) | — | +24.24 ms | +13.9 % | no class (one load) |
|  | host | host: 180.31 (74 steps) | — | +6.34 ms | +3.6 % | no class (one load) |
|  | hostdev | hostdev: 174.15 (74 steps) | — | +0.18 ms | +0.1 % | no class (one load) |

### Prompt reading, tok/s

Per load: the tokens read at that depth over their time (a prime request reads a chunk on top of the cache; without primes each decode request reads its prompt), with the tokens read and the tokens reused. Requests that read under 32 tokens are left out.

| Depth (tokens) | Arm | Per load | Spread | Against `dev` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | dev | dev: 32.6 (64 read on 0 cached) | — | control | | |
|  | check | check: 32.0 (64 read on 0 cached) | — | -0.7 tok/s | -2.1 % | no class (one load) |
|  | host | host: 31.9 (64 read on 0 cached) | — | -0.8 tok/s | -2.4 % | no class (one load) |
|  | hostdev | hostdev: 32.7 (64 read on 0 cached) | — | +0.1 tok/s | +0.2 % | no class (one load) |
| 16,384 | dev | dev: 131.6 (16320 read on 0 cached) | — | control | | |
|  | check | check: 68.8 (16320 read on 0 cached) | — | -62.8 tok/s | -47.7 % | no class (one load) |
|  | host | host: 116.2 (16320 read on 0 cached) | — | -15.4 tok/s | -11.7 % | no class (one load) |
|  | hostdev | hostdev: 115.8 (16320 read on 0 cached) | — | -15.8 tok/s | -12.0 % | no class (one load) |
| 49,152 | dev | dev: 82.9 (33284 read on 15804 cached) | — | control | | |
|  | check | check: 42.8 (33284 read on 15804 cached) | — | -40.1 tok/s | -48.4 % | no class (one load) |
|  | host | host: 73.8 (33284 read on 15804 cached) | — | -9.1 tok/s | -10.9 % | no class (one load) |
|  | hostdev | hostdev: 73.6 (33284 read on 15804 cached) | — | -9.3 tok/s | -11.2 % | no class (one load) |

### The draft and the step

Per load, pass 1: draft tokens accepted of drafted (the share), tokens a step (generated tokens over steps, a step being one pass of the main model: generated minus accepted), and the time of a step. Pass 2 is listed where its counters differ.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | dev | dev: 178 of 229 (0.777), 3.28 a step, 125.4 ms a step (pass 2: 181 of 220 (0.823), 3.41 a step, 121.7 ms a step) |
|  | check | check: 178 of 229 (0.777), 3.28 a step, 145.5 ms a step (pass 2: 181 of 220 (0.823), 3.41 a step, 144.0 ms a step) |
|  | host | host: 178 of 229 (0.777), 3.28 a step, 125.7 ms a step (pass 2: 181 of 220 (0.823), 3.41 a step, 122.0 ms a step) |
|  | hostdev | hostdev: 178 of 229 (0.777), 3.28 a step, 125.9 ms a step (pass 2: 181 of 220 (0.823), 3.41 a step, 122.0 ms a step) |
| 16,384 | dev | dev: 184 of 210 (0.876), 3.56 a step, 182.7 ms a step |
|  | check | check: 184 of 210 (0.876), 3.56 a step, 186.3 ms a step |
|  | host | host: 184 of 210 (0.876), 3.56 a step, 168.1 ms a step |
|  | hostdev | hostdev: 184 of 210 (0.876), 3.56 a step, 165.6 ms a step |
| 49,152 | dev | dev: 182 of 218 (0.835), 3.46 a step, 227.8 ms a step |
|  | check | check: 182 of 218 (0.835), 3.46 a step, 228.8 ms a step |
|  | host | host: 182 of 218 (0.835), 3.46 a step, 227.5 ms a step |
|  | hostdev | hostdev: 182 of 218 (0.835), 3.46 a step, 247.3 ms a step |

### Checks

- Times: every load started on a machine checked quiet (valid=1).
- Prompts: 64 ids at depth 0, 16384 ids at depth 16,384, 49152 ids at depth 49,152; the same ids on every load and pass.
- Tokens out: 256 on every timed request.
- Greedy reply at depth 0: the same on every pass and load of every arm.
- Greedy reply at depth 16,384: the same on every pass and load of every arm.
- Greedy reply at depth 49,152: the same on every pass and load of every arm.
- Primes (read on cached, the cache's source): 16,384: 16320 on 0 (none); 49,152: 33284 on 15804 (checkpoint).
