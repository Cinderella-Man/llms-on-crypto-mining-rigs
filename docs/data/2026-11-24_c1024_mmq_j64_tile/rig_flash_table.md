## The deep lane: unset, j64

Model file `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf`, llama-server version: 1.0.0-dev (build 13219, commit ebec6544e3).

256 tokens per request at temperature 0, each request twice per load (pass 1 is the first request at a new depth, pass 2 the same request again). Timed loads, in their order: unset1, j641, unset2, j642.

| Arm | Loads | Draft | Context | Its own variables and flags |
|---|---|---|---:|---|
| unset (control) | unset1, unset2 | on | 65536 | — |
| j64 | j641, j642 | on | 65536 | `GGML_CUDA_MMQ_J64_I64=1` |

### Generation, tok/s

Per load: pass 1, pass 2 → the load, with the load's time per token. A load's speed is its two passes' tokens over their time. Spread: the difference between the arm's fastest and slowest load. Class against `unset`: gain when the mean is above the control's by more than the larger of the two spreads, loss when below by more than it, else negligible.

| Depth (tokens) | Arm | Per load | Spread | Against `unset` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | unset | unset1: 23.19, 24.22 → **23.69** (42.21 ms a token); unset2: 22.92, 24.18 → **23.54** (42.49 ms a token) | 0.16 | control | | |
|  | j64 | j641: 22.96, 24.23 → **23.58** (42.41 ms a token); j642: 22.94, 24.20 → **23.55** (42.46 ms a token) | 0.03 | -0.05 tok/s | -0.2 % | negligible |
| 4,096 | unset | unset1: 28.73, 29.83 → **29.27** (34.17 ms a token); unset2: 28.61, 29.86 → **29.22** (34.22 ms a token) | 0.04 | control | | |
|  | j64 | j641: 28.74, 29.76 → **29.24** (34.20 ms a token); j642: 28.70, 29.84 → **29.26** (34.17 ms a token) | 0.02 | +0.01 tok/s | +0.0 % | negligible |
| 8,192 | unset | unset1: 25.68, 29.04 → **27.26** (36.69 ms a token); unset2: 25.67, 29.05 → **27.25** (36.69 ms a token) | 0.00 | control | | |
|  | j64 | j641: 25.63, 29.09 → **27.25** (36.70 ms a token); j642: 25.59, 29.07 → **27.22** (36.74 ms a token) | 0.03 | -0.02 tok/s | -0.1 % | negligible |

### Generation over all depths, tok/s

Every depth's tokens over their time, per load.

| Depth (tokens) | Arm | Per load | Spread | Against `unset` | | Class |
|---:|---|---|---:|---:|---:|---|
| all | unset | unset1: **26.53**; unset2: **26.45** | 0.08 | control | | |
|  | j64 | j641: **26.48**; j642: **26.46** | 0.01 | -0.02 tok/s | -0.1 % | negligible |

### The step on pass 1 (the first request at a new depth), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `unset`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `unset` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | unset | unset1: 101.30 (109 steps); unset2: 102.46 (109 steps) | 1.16 | control | | |
|  | j64 | j641: 102.27 (109 steps); j642: 102.37 (109 steps) | 0.10 | +0.45 ms | +0.4 % | negligible |
| 4,096 | unset | unset1: 112.81 (79 steps); unset2: 113.25 (79 steps) | 0.44 | control | | |
|  | j64 | j641: 112.75 (79 steps); j642: 112.90 (79 steps) | 0.15 | -0.20 ms | -0.2 % | negligible |
| 8,192 | unset | unset1: 120.11 (83 steps); unset2: 120.16 (83 steps) | 0.06 | control | | |
|  | j64 | j641: 120.33 (83 steps); j642: 120.53 (83 steps) | 0.20 | +0.30 ms | +0.2 % | loss |

### The step on pass 2 (the same request again), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `unset`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `unset` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | unset | unset1: 97.88 (108 steps); unset2: 98.02 (108 steps) | 0.15 | control | | |
|  | j64 | j641: 97.83 (108 steps); j642: 97.95 (108 steps) | 0.11 | -0.06 ms | -0.1 % | negligible |
| 4,096 | unset | unset1: 108.63 (79 steps); unset2: 108.53 (79 steps) | 0.10 | control | | |
|  | j64 | j641: 108.91 (79 steps); j642: 108.58 (79 steps) | 0.33 | +0.16 ms | +0.1 % | negligible |
| 8,192 | unset | unset1: 113.02 (78 steps); unset2: 112.98 (78 steps) | 0.04 | control | | |
|  | j64 | j641: 112.83 (78 steps); j642: 112.91 (78 steps) | 0.08 | -0.14 ms | -0.1 % | gain |

### Prompt reading, tok/s

Per load: the tokens read at that depth over their time (a prime request reads a chunk on top of the cache; without primes each decode request reads its prompt), with the tokens read and the tokens reused. Requests that read under 32 tokens are left out.

| Depth (tokens) | Arm | Per load | Spread | Against `unset` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | unset | unset1: 53.6 (64 read on 0 cached); unset2: 53.2 (64 read on 0 cached) | 0.4 | control | | |
|  | j64 | j641: 53.0 (64 read on 0 cached); j642: 51.5 (64 read on 0 cached) | 1.5 | -1.2 tok/s | -2.2 % | negligible |
| 4,096 | unset | unset1: 185.3 (4032 read on 0 cached); unset2: 187.7 (4032 read on 0 cached) | 2.4 | control | | |
|  | j64 | j641: 193.8 (4032 read on 0 cached); j642: 196.2 (4032 read on 0 cached) | 2.4 | +8.5 tok/s | +4.6 % | gain |
| 8,192 | unset | unset1: 212.0 (4612 read on 3516 cached); unset2: 225.2 (4612 read on 3516 cached) | 13.2 | control | | |
|  | j64 | j641: 214.6 (4612 read on 3516 cached); j642: 219.3 (4612 read on 3516 cached) | 4.8 | -1.7 tok/s | -0.8 % | negligible |

### The draft and the step

Per load, pass 1: draft tokens accepted of drafted (the share), tokens a step (generated tokens over steps, a step being one pass of the main model: generated minus accepted), and the time of a step. Pass 2 is listed where its counters differ.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | unset | unset1: 147 of 321 (0.458), 2.35 a step, 101.3 ms a step (pass 2: 148 of 318 (0.465), 2.37 a step, 97.9 ms a step); unset2: 147 of 321 (0.458), 2.35 a step, 102.5 ms a step (pass 2: 148 of 318 (0.465), 2.37 a step, 98.0 ms a step) |
|  | j64 | j641: 147 of 321 (0.458), 2.35 a step, 102.3 ms a step (pass 2: 148 of 318 (0.465), 2.37 a step, 97.8 ms a step); j642: 147 of 321 (0.458), 2.35 a step, 102.4 ms a step (pass 2: 148 of 318 (0.465), 2.37 a step, 97.9 ms a step) |
| 4,096 | unset | unset1: 177 of 231 (0.766), 3.24 a step, 112.8 ms a step; unset2: 177 of 231 (0.766), 3.24 a step, 113.2 ms a step |
|  | j64 | j641: 177 of 231 (0.766), 3.24 a step, 112.7 ms a step; j642: 177 of 231 (0.766), 3.24 a step, 112.9 ms a step |
| 8,192 | unset | unset1: 173 of 245 (0.706), 3.08 a step, 120.1 ms a step (pass 2: 178 of 230 (0.774), 3.28 a step, 113.0 ms a step); unset2: 173 of 245 (0.706), 3.08 a step, 120.2 ms a step (pass 2: 178 of 230 (0.774), 3.28 a step, 113.0 ms a step) |
|  | j64 | j641: 173 of 245 (0.706), 3.08 a step, 120.3 ms a step (pass 2: 178 of 230 (0.774), 3.28 a step, 112.8 ms a step); j642: 173 of 245 (0.706), 3.08 a step, 120.5 ms a step (pass 2: 178 of 230 (0.774), 3.28 a step, 112.9 ms a step) |

### The sampler pass

One request per depth and load with the server line's own sampler and a fixed seed: tok/s, the draft counters and the step.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | unset | unset1: 26.54 tok/s, 159 of 285 (0.558), 2.64 a step, 99.4 ms a step; unset2: 26.26 tok/s, 159 of 285 (0.558), 2.64 a step, 100.5 ms a step |
|  | j64 | j641: 26.30 tok/s, 159 of 285 (0.558), 2.64 a step, 100.3 ms a step; j642: 26.15 tok/s, 159 of 285 (0.558), 2.64 a step, 100.9 ms a step |
| 4,096 | unset | unset1: 28.96 tok/s, 177 of 232 (0.763), 3.24 a step, 111.9 ms a step; unset2: 29.01 tok/s, 177 of 232 (0.763), 3.24 a step, 111.7 ms a step |
|  | j64 | j641: 29.06 tok/s, 177 of 232 (0.763), 3.24 a step, 111.5 ms a step; j642: 28.97 tok/s, 177 of 232 (0.763), 3.24 a step, 111.8 ms a step |
| 8,192 | unset | unset1: 29.57 tok/s, 181 of 222 (0.815), 3.41 a step, 115.4 ms a step; unset2: 29.97 tok/s, 181 of 222 (0.815), 3.41 a step, 113.9 ms a step |
|  | j64 | j641: 30.07 tok/s, 181 of 222 (0.815), 3.41 a step, 113.5 ms a step; j642: 30.04 tok/s, 181 of 222 (0.815), 3.41 a step, 113.6 ms a step |

### Checks

- Times: every load started on a machine checked quiet (valid=1).
- Prompts: 64 ids at depth 0, 4096 ids at depth 4,096, 8192 ids at depth 8,192; the same ids on every load and pass.
- Tokens out: 256 on every timed request.
- Greedy reply at depth 0: the same on every pass and load of every arm.
- Greedy reply at depth 4,096: the same on every pass and load of every arm.
- Greedy reply at depth 8,192: the same on every pass and load of every arm.
- Primes (read on cached, the cache's source): 4,096: 4032 on 0 (none); 8,192: 4612 on 3516 (checkpoint).
- Graph reallocations of the target, requests with one per load: unset1: 0 of 12 requests; j641: 0 of 12 requests; unset2: 0 of 12 requests; j642: 0 of 12 requests.
