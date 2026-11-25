## The deep lane: c1, s0

Model file `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf`, llama-server version: 1.0.0-dev (build 13218, commit eee615cdb9).

256 tokens per request at temperature 0, each request twice per load (pass 1 is the first request at a new depth, pass 2 the same request again). Timed loads, in their order: c1a, s0a, s0b, c1b.

| Arm | Loads | Draft | Context | Its own variables and flags |
|---|---|---|---:|---|
| c1 (control) | c1a, c1b | on | 65536 | `LLAMA_DRAFT_SHORTLIST=32768,8192,1` |
| s0 | s0a, s0b | on | 65536 | `LLAMA_DRAFT_SHORTLIST=32768,8192,0` |

### Generation, tok/s

Per load: pass 1, pass 2 → the load, with the load's time per token. A load's speed is its two passes' tokens over their time. Spread: the difference between the arm's fastest and slowest load. Class against `c1`: gain when the mean is above the control's by more than the larger of the two spreads, loss when below by more than it, else negligible.

| Depth (tokens) | Arm | Per load | Spread | Against `c1` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | c1 | c1a: 23.51, 24.44 → **23.97** (41.72 ms a token); c1b: 23.27, 24.40 → **23.82** (41.97 ms a token) | 0.14 | control | | |
|  | s0 | s0a: 23.38, 24.27 → **23.82** (41.99 ms a token); s0b: 23.37, 24.38 → **23.87** (41.90 ms a token) | 0.05 | -0.06 tok/s | -0.2 % | negligible |
| 4,096 | c1 | c1a: 28.64, 29.79 → **29.20** (34.24 ms a token); c1b: 28.71, 29.78 → **29.23** (34.21 ms a token) | 0.03 | control | | |
|  | s0 | s0a: 27.77, 29.81 → **28.75** (34.78 ms a token); s0b: 26.20, 29.73 → **27.85** (35.91 ms a token) | 0.90 | -0.91 tok/s | -3.1 % | loss |
| 8,192 | c1 | c1a: 25.83, 28.96 → **27.30** (36.62 ms a token); c1b: 25.81, 28.98 → **27.30** (36.63 ms a token) | 0.00 | control | | |
|  | s0 | s0a: 25.28, 28.99 → **27.01** (37.03 ms a token); s0b: 25.26, 28.93 → **26.97** (37.08 ms a token) | 0.04 | -0.32 tok/s | -1.2 % | loss |

### Generation over all depths, tok/s

Every depth's tokens over their time, per load.

| Depth (tokens) | Arm | Per load | Spread | Against `c1` | | Class |
|---:|---|---|---:|---:|---:|---|
| all | c1 | c1a: **26.65**; c1b: **26.59** | 0.05 | control | | |
|  | s0 | s0a: **26.36**; s0b: **26.11** | 0.25 | -0.38 tok/s | -1.4 % | loss |

### The step on pass 1 (the first request at a new depth), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `c1`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `c1` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | c1 | c1a: 100.82 (108 steps); c1b: 101.85 (108 steps) | 1.03 | control | | |
|  | s0 | s0a: 101.38 (108 steps); s0b: 101.43 (108 steps) | 0.05 | +0.07 ms | +0.1 % | negligible |
| 4,096 | c1 | c1a: 113.16 (79 steps); c1b: 112.89 (79 steps) | 0.27 | control | | |
|  | s0 | s0a: 116.71 (79 steps); s0b: 123.70 (79 steps) | 6.99 | +7.18 ms | +6.3 % | loss |
| 8,192 | c1 | c1a: 119.42 (83 steps); c1b: 119.52 (83 steps) | 0.10 | control | | |
|  | s0 | s0a: 117.74 (86 steps); s0b: 117.87 (86 steps) | 0.13 | -1.67 ms | -1.4 % | gain |

### The step on pass 2 (the same request again), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `c1`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `c1` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | c1 | c1a: 97.88 (107 steps); c1b: 98.04 (107 steps) | 0.17 | control | | |
|  | s0 | s0a: 98.59 (107 steps); s0b: 98.12 (107 steps) | 0.47 | +0.39 ms | +0.4 % | negligible |
| 4,096 | c1 | c1a: 108.78 (79 steps); c1b: 108.82 (79 steps) | 0.04 | control | | |
|  | s0 | s0a: 108.70 (79 steps); s0b: 109.01 (79 steps) | 0.31 | +0.05 ms | +0.0 % | negligible |
| 8,192 | c1 | c1a: 113.34 (78 steps); c1b: 113.24 (78 steps) | 0.10 | control | | |
|  | s0 | s0a: 113.22 (78 steps); s0b: 113.46 (78 steps) | 0.23 | +0.05 ms | +0.0 % | negligible |

### Prompt reading, tok/s

Per load: the tokens read at that depth over their time (a prime request reads a chunk on top of the cache; without primes each decode request reads its prompt), with the tokens read and the tokens reused. Requests that read under 32 tokens are left out.

| Depth (tokens) | Arm | Per load | Spread | Against `c1` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | c1 | c1a: 53.6 (64 read on 0 cached); c1b: 51.1 (64 read on 0 cached) | 2.5 | control | | |
|  | s0 | s0a: 51.4 (64 read on 0 cached); s0b: 53.4 (64 read on 0 cached) | 2.0 | +0.1 tok/s | +0.2 % | negligible |
| 4,096 | c1 | c1a: 197.9 (4032 read on 0 cached); c1b: 196.7 (4032 read on 0 cached) | 1.2 | control | | |
|  | s0 | s0a: 194.2 (4032 read on 0 cached); s0b: 198.0 (4032 read on 0 cached) | 3.8 | -1.2 tok/s | -0.6 % | negligible |
| 8,192 | c1 | c1a: 211.6 (4612 read on 3516 cached); c1b: 218.6 (4612 read on 3516 cached) | 7.0 | control | | |
|  | s0 | s0a: 218.2 (4612 read on 3516 cached); s0b: 215.9 (4612 read on 3516 cached) | 2.3 | +1.9 tok/s | +0.9 % | negligible |

### The draft and the step

Per load, pass 1: draft tokens accepted of drafted (the share), tokens a step (generated tokens over steps, a step being one pass of the main model: generated minus accepted), and the time of a step. Pass 2 is listed where its counters differ.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | c1 | c1a: 148 of 318 (0.465), 2.37 a step, 100.8 ms a step (pass 2: 149 of 315 (0.473), 2.39 a step, 97.9 ms a step); c1b: 148 of 318 (0.465), 2.37 a step, 101.8 ms a step (pass 2: 149 of 315 (0.473), 2.39 a step, 98.0 ms a step) |
|  | s0 | s0a: 148 of 318 (0.465), 2.37 a step, 101.4 ms a step (pass 2: 149 of 315 (0.473), 2.39 a step, 98.6 ms a step); s0b: 148 of 318 (0.465), 2.37 a step, 101.4 ms a step (pass 2: 149 of 315 (0.473), 2.39 a step, 98.1 ms a step) |
| 4,096 | c1 | c1a: 177 of 231 (0.766), 3.24 a step, 113.2 ms a step; c1b: 177 of 231 (0.766), 3.24 a step, 112.9 ms a step |
|  | s0 | s0a: 177 of 231 (0.766), 3.24 a step, 116.7 ms a step; s0b: 177 of 231 (0.766), 3.24 a step, 123.7 ms a step |
| 8,192 | c1 | c1a: 173 of 245 (0.706), 3.08 a step, 119.4 ms a step (pass 2: 178 of 230 (0.774), 3.28 a step, 113.3 ms a step); c1b: 173 of 245 (0.706), 3.08 a step, 119.5 ms a step (pass 2: 178 of 230 (0.774), 3.28 a step, 113.2 ms a step) |
|  | s0 | s0a: 170 of 254 (0.669), 2.98 a step, 117.7 ms a step (pass 2: 178 of 230 (0.774), 3.28 a step, 113.2 ms a step); s0b: 170 of 254 (0.669), 2.98 a step, 117.9 ms a step (pass 2: 178 of 230 (0.774), 3.28 a step, 113.5 ms a step) |

### The sampler pass

One request per depth and load with the server line's own sampler and a fixed seed: tok/s, the draft counters and the step.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | c1 | c1a: 22.65 tok/s, 145 of 329 (0.441), 2.31 a step, 101.8 ms a step; c1b: 22.52 tok/s, 145 of 329 (0.441), 2.31 a step, 102.4 ms a step |
|  | s0 | s0a: 22.66 tok/s, 145 of 329 (0.441), 2.31 a step, 101.8 ms a step; s0b: 22.67 tok/s, 145 of 329 (0.441), 2.31 a step, 101.7 ms a step |
| 4,096 | c1 | c1a: 30.02 tok/s, 179 of 228 (0.785), 3.32 a step, 110.7 ms a step; c1b: 30.15 tok/s, 179 of 228 (0.785), 3.32 a step, 110.3 ms a step |
|  | s0 | s0a: 29.44 tok/s, 178 of 230 (0.774), 3.28 a step, 111.5 ms a step; s0b: 29.36 tok/s, 178 of 230 (0.774), 3.28 a step, 111.8 ms a step |
| 8,192 | c1 | c1a: 28.43 tok/s, 181 of 220 (0.823), 3.41 a step, 120.1 ms a step; c1b: 29.60 tok/s, 181 of 220 (0.823), 3.41 a step, 115.3 ms a step |
|  | s0 | s0a: 29.62 tok/s, 181 of 220 (0.823), 3.41 a step, 115.2 ms a step; s0b: 29.39 tok/s, 181 of 220 (0.823), 3.41 a step, 116.1 ms a step |

### Checks

- Times: every load started on a machine checked quiet (valid=1).
- Prompts: 64 ids at depth 0, 4096 ids at depth 4,096, 8192 ids at depth 8,192; the same ids on every load and pass.
- Tokens out: 256 on every timed request.
- Greedy reply at depth 0: the same on every pass and load of every arm.
- Greedy reply at depth 4,096: the same on every pass and load of every arm.
- Greedy reply at depth 8,192: the same on every pass and load of every arm.
- Primes (read on cached, the cache's source): 4,096: 4032 on 0 (none); 8,192: 4612 on 3516 (checkpoint).
- Graph reallocations of the target, requests with one per load: c1a: 0 of 12 requests; s0a: 0 of 12 requests; s0b: 0 of 12 requests; c1b: 0 of 12 requests.
