## The deep lane: c, fb1

Model file `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf`, llama-server version: 1.0.0-dev (build 12967, commit 6026911533).

256 tokens per request at temperature 0, each request twice per load (pass 1 is the first request at a new depth, pass 2 the same request again). Timed loads, in their order: ca, fb1a, fb1b, cb.

| Arm | Loads | Draft | Context | Its own variables and flags |
|---|---|---|---:|---|
| c (control) | ca, cb | on | 65536 | `LLAMA_DRAFT_SHORTLIST=32768,8192,2` |
| fb1 | fb1a, fb1b | on | 65536 | `LLAMA_DRAFT_SHORTLIST=32768,8192,1` |

### Generation, tok/s

Per load: pass 1, pass 2 → the load, with the load's time per token. A load's speed is its two passes' tokens over their time. Spread: the difference between the arm's fastest and slowest load. Class against `c`: gain when the mean is above the control's by more than the larger of the two spreads, loss when below by more than it, else negligible.

| Depth (tokens) | Arm | Per load | Spread | Against `c` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | c | ca: 22.84, 26.30 → **24.45** (40.90 ms a token); cb: 24.85, 26.29 → **25.55** (39.14 ms a token) | 1.10 | control | | |
|  | fb1 | fb1a: 25.17, 26.51 → **25.82** (38.73 ms a token); fb1b: 25.21, 26.53 → **25.85** (38.68 ms a token) | 0.03 | +0.84 tok/s | +3.4 % | negligible |
| 4,096 | c | ca: 31.09, 32.56 → **31.81** (31.44 ms a token); cb: 31.22, 32.54 → **31.86** (31.38 ms a token) | 0.06 | control | | |
|  | fb1 | fb1a: 31.28, 32.59 → **31.92** (31.33 ms a token); fb1b: 31.23, 32.51 → **31.86** (31.39 ms a token) | 0.06 | +0.05 tok/s | +0.2 % | negligible |
| 8,192 | c | ca: 27.71, 31.61 → **29.53** (33.86 ms a token); cb: 27.74, 31.64 → **29.56** (33.83 ms a token) | 0.03 | control | | |
|  | fb1 | fb1a: 28.08, 31.63 → **29.75** (33.61 ms a token); fb1b: 28.03, 31.65 → **29.73** (33.64 ms a token) | 0.02 | +0.19 tok/s | +0.7 % | gain |

### Generation over all depths, tok/s

Every depth's tokens over their time, per load.

| Depth (tokens) | Arm | Per load | Spread | Against `c` | | Class |
|---:|---|---|---:|---:|---:|---|
| all | c | ca: **28.25**; cb: **28.75** | 0.50 | control | | |
|  | fb1 | fb1a: **28.94**; fb1b: **28.93** | 0.01 | +0.43 tok/s | +1.5 % | negligible |

### The step on pass 1 (the first request at a new depth), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `c`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `c` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | c | ca: 102.81 (109 steps); cb: 94.53 (109 steps) | 8.28 | control | | |
|  | fb1 | fb1a: 94.17 (108 steps); fb1b: 94.04 (108 steps) | 0.13 | -4.56 ms | -4.6 % | negligible |
| 4,096 | c | ca: 104.24 (79 steps); cb: 103.81 (79 steps) | 0.43 | control | | |
|  | fb1 | fb1a: 103.61 (79 steps); fb1b: 103.76 (79 steps) | 0.15 | -0.34 ms | -0.3 % | negligible |
| 8,192 | c | ca: 111.30 (83 steps); cb: 111.20 (83 steps) | 0.10 | control | | |
|  | fb1 | fb1a: 109.82 (83 steps); fb1b: 110.05 (83 steps) | 0.22 | -1.31 ms | -1.2 % | gain |

### The step on pass 2 (the same request again), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `c`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `c` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | c | ca: 90.13 (108 steps); cb: 90.17 (108 steps) | 0.04 | control | | |
|  | fb1 | fb1a: 90.26 (107 steps); fb1b: 90.18 (107 steps) | 0.09 | +0.07 ms | +0.1 % | negligible |
| 4,096 | c | ca: 99.52 (79 steps); cb: 99.58 (79 steps) | 0.05 | control | | |
|  | fb1 | fb1a: 99.44 (79 steps); fb1b: 99.67 (79 steps) | 0.23 | +0.01 ms | +0.0 % | negligible |
| 8,192 | c | ca: 103.82 (78 steps); cb: 103.72 (78 steps) | 0.10 | control | | |
|  | fb1 | fb1a: 103.75 (78 steps); fb1b: 103.69 (78 steps) | 0.07 | -0.05 ms | -0.1 % | negligible |

### Prompt reading, tok/s

Per load: the tokens read at that depth over their time (a prime request reads a chunk on top of the cache; without primes each decode request reads its prompt), with the tokens read and the tokens reused. Requests that read under 32 tokens are left out.

| Depth (tokens) | Arm | Per load | Spread | Against `c` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | c | ca: 59.0 (64 read on 0 cached); cb: 58.8 (64 read on 0 cached) | 0.2 | control | | |
|  | fb1 | fb1a: 56.7 (64 read on 0 cached); fb1b: 58.5 (64 read on 0 cached) | 1.8 | -1.3 tok/s | -2.2 % | negligible |
| 4,096 | c | ca: 205.5 (4032 read on 0 cached); cb: 208.4 (4032 read on 0 cached) | 2.9 | control | | |
|  | fb1 | fb1a: 205.9 (4032 read on 0 cached); fb1b: 204.2 (4032 read on 0 cached) | 1.7 | -1.9 tok/s | -0.9 % | negligible |
| 8,192 | c | ca: 236.5 (4612 read on 3516 cached); cb: 234.5 (4612 read on 3516 cached) | 2.0 | control | | |
|  | fb1 | fb1a: 215.5 (4612 read on 3516 cached); fb1b: 222.7 (4612 read on 3516 cached) | 7.2 | -16.5 tok/s | -7.0 % | loss |

### The draft and the step

Per load, pass 1: draft tokens accepted of drafted (the share), tokens a step (generated tokens over steps, a step being one pass of the main model: generated minus accepted), and the time of a step. Pass 2 is listed where its counters differ.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | c | ca: 147 of 321 (0.458), 2.35 a step, 102.8 ms a step (pass 2: 148 of 318 (0.465), 2.37 a step, 90.1 ms a step); cb: 147 of 321 (0.458), 2.35 a step, 94.5 ms a step (pass 2: 148 of 318 (0.465), 2.37 a step, 90.2 ms a step) |
|  | fb1 | fb1a: 148 of 318 (0.465), 2.37 a step, 94.2 ms a step (pass 2: 149 of 315 (0.473), 2.39 a step, 90.3 ms a step); fb1b: 148 of 318 (0.465), 2.37 a step, 94.0 ms a step (pass 2: 149 of 315 (0.473), 2.39 a step, 90.2 ms a step) |
| 4,096 | c | ca: 177 of 231 (0.766), 3.24 a step, 104.2 ms a step; cb: 177 of 231 (0.766), 3.24 a step, 103.8 ms a step |
|  | fb1 | fb1a: 177 of 231 (0.766), 3.24 a step, 103.6 ms a step; fb1b: 177 of 231 (0.766), 3.24 a step, 103.8 ms a step |
| 8,192 | c | ca: 173 of 245 (0.706), 3.08 a step, 111.3 ms a step (pass 2: 178 of 230 (0.774), 3.28 a step, 103.8 ms a step); cb: 173 of 245 (0.706), 3.08 a step, 111.2 ms a step (pass 2: 178 of 230 (0.774), 3.28 a step, 103.7 ms a step) |
|  | fb1 | fb1a: 173 of 245 (0.706), 3.08 a step, 109.8 ms a step (pass 2: 178 of 230 (0.774), 3.28 a step, 103.8 ms a step); fb1b: 173 of 245 (0.706), 3.08 a step, 110.0 ms a step (pass 2: 178 of 230 (0.774), 3.28 a step, 103.7 ms a step) |

### The sampler pass

One request per depth and load with the server line's own sampler and a fixed seed: tok/s, the draft counters and the step.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | c | ca: 28.40 tok/s, 159 of 285 (0.558), 2.64 a step, 92.9 ms a step; cb: 28.41 tok/s, 159 of 285 (0.558), 2.64 a step, 92.9 ms a step |
|  | fb1 | fb1a: 24.37 tok/s, 145 of 329 (0.441), 2.31 a step, 94.6 ms a step; fb1b: 24.36 tok/s, 145 of 329 (0.441), 2.31 a step, 94.7 ms a step |
| 4,096 | c | ca: 28.40 tok/s, 177 of 232 (0.763), 3.24 a step, 114.1 ms a step; cb: 31.64 tok/s, 177 of 232 (0.763), 3.24 a step, 102.4 ms a step |
|  | fb1 | fb1a: 33.03 tok/s, 179 of 228 (0.785), 3.32 a step, 100.7 ms a step; fb1b: 32.96 tok/s, 179 of 228 (0.785), 3.32 a step, 100.9 ms a step |
| 8,192 | c | ca: 30.23 tok/s, 181 of 222 (0.815), 3.41 a step, 112.9 ms a step; cb: 32.58 tok/s, 181 of 222 (0.815), 3.41 a step, 104.8 ms a step |
|  | fb1 | fb1a: 32.22 tok/s, 181 of 220 (0.823), 3.41 a step, 106.0 ms a step; fb1b: 32.28 tok/s, 181 of 220 (0.823), 3.41 a step, 105.8 ms a step |

### Checks

- Times: every load started on a machine checked quiet (valid=1).
- Prompts: 64 ids at depth 0, 4096 ids at depth 4,096, 8192 ids at depth 8,192; the same ids on every load and pass.
- Tokens out: 256 on every timed request.
- Greedy reply at depth 0: the same on every pass and load of every arm.
- Greedy reply at depth 4,096: the same on every pass and load of every arm.
- Greedy reply at depth 8,192: the same on every pass and load of every arm.
- Primes (read on cached, the cache's source): 4,096: 4032 on 0 (none); 8,192: 4612 on 3516 (checkpoint).
- Graph reallocations of the target, requests with one per load: ca: 0 of 12 requests; fb1a: 0 of 12 requests; fb1b: 0 of 12 requests; cb: 0 of 12 requests.
