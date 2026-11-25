## The deep lane: c, off

Model file `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf`, llama-server version: 1.0.0-dev (build 13218, commit eee615cdb9).

256 tokens per request at temperature 0, each request twice per load (pass 1 is the first request at a new depth, pass 2 the same request again). Timed loads, in their order: ca, offa, offb, cb.

| Arm | Loads | Draft | Context | Its own variables and flags |
|---|---|---|---:|---|
| c (control) | ca, cb | on | 65536 | `LLAMA_MTP_CATCHUP_OVERLAP=1` |
| off | offa, offb | on | 65536 | `LLAMA_MTP_CATCHUP_OVERLAP=0` |

### Generation, tok/s

Per load: pass 1, pass 2 → the load, with the load's time per token. A load's speed is its pass 1. Spread: the difference between the arm's fastest and slowest load. Class against `c`: gain when the mean is above the control's by more than the larger of the two spreads, loss when below by more than it, else negligible.

| Depth (tokens) | Arm | Per load | Spread | Against `c` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | c | ca: 22.29, 24.17 → **22.29** (44.86 ms a token); cb: 22.87, 24.15 → **22.87** (43.73 ms a token) | 0.58 | control | | |
|  | off | offa: 22.93, 24.16 → **22.93** (43.62 ms a token); offb: 22.93, 24.16 → **22.93** (43.61 ms a token) | 0.00 | +0.35 tok/s | +1.5 % | negligible |
| 4,096 | c | ca: 28.34, 29.81 → **28.34** (35.29 ms a token); cb: 27.66, 29.79 → **27.66** (36.16 ms a token) | 0.68 | control | | |
|  | off | offa: 28.65, 29.79 → **28.65** (34.91 ms a token); offb: 28.67, 29.75 → **28.67** (34.88 ms a token) | 0.02 | +0.66 tok/s | +2.4 % | negligible |
| 8,192 | c | ca: 25.60, 28.99 → **25.60** (39.06 ms a token); cb: 25.65, 29.04 → **25.65** (38.98 ms a token) | 0.05 | control | | |
|  | off | offa: 25.58, 28.98 → **25.58** (39.10 ms a token); offb: 25.55, 28.96 → **25.55** (39.14 ms a token) | 0.03 | -0.06 tok/s | -0.2 % | loss |

### Generation over all depths, tok/s

Every depth's tokens over their time, per load.

| Depth (tokens) | Arm | Per load | Spread | Against `c` | | Class |
|---:|---|---|---:|---:|---:|---|
| all | c | ca: **25.17**; cb: **25.24** | 0.07 | control | | |
|  | off | offa: **25.51**; offb: **25.51** | 0.00 | +0.30 tok/s | +1.2 % | gain |

### The step on pass 1 (the first request at a new depth), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `c`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `c` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | c | ca: 105.37 (109 steps); cb: 102.70 (109 steps) | 2.67 | control | | |
|  | off | offa: 102.44 (109 steps); offb: 102.42 (109 steps) | 0.02 | -1.60 ms | -1.5 % | negligible |
| 4,096 | c | ca: 114.35 (79 steps); cb: 117.17 (79 steps) | 2.82 | control | | |
|  | off | offa: 113.11 (79 steps); offb: 113.02 (79 steps) | 0.09 | -2.69 ms | -2.3 % | negligible |
| 8,192 | c | ca: 120.47 (83 steps); cb: 120.22 (83 steps) | 0.25 | control | | |
|  | off | offa: 120.58 (83 steps); offb: 120.71 (83 steps) | 0.12 | +0.30 ms | +0.2 % | loss |

### The step on pass 2 (the same request again), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `c`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `c` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | c | ca: 98.08 (108 steps); cb: 98.14 (108 steps) | 0.07 | control | | |
|  | off | offa: 98.11 (108 steps); offb: 98.12 (108 steps) | 0.01 | +0.00 ms | +0.0 % | negligible |
| 4,096 | c | ca: 108.70 (79 steps); cb: 108.76 (79 steps) | 0.06 | control | | |
|  | off | offa: 108.77 (79 steps); offb: 108.91 (79 steps) | 0.14 | +0.11 ms | +0.1 % | negligible |
| 8,192 | c | ca: 113.20 (78 steps); cb: 113.04 (78 steps) | 0.16 | control | | |
|  | off | offa: 113.27 (78 steps); offb: 113.32 (78 steps) | 0.05 | +0.18 ms | +0.2 % | loss |

### Prompt reading, tok/s

Per load: the tokens read at that depth over their time (a prime request reads a chunk on top of the cache; without primes each decode request reads its prompt), with the tokens read and the tokens reused. Requests that read under 32 tokens are left out.

| Depth (tokens) | Arm | Per load | Spread | Against `c` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | c | ca: 52.3 (64 read on 0 cached); cb: 51.9 (64 read on 0 cached) | 0.4 | control | | |
|  | off | offa: 51.8 (64 read on 0 cached); offb: 51.6 (64 read on 0 cached) | 0.2 | -0.4 tok/s | -0.9 % | loss |
| 4,096 | c | ca: 186.2 (4032 read on 0 cached); cb: 197.5 (4032 read on 0 cached) | 11.3 | control | | |
|  | off | offa: 196.7 (4032 read on 0 cached); offb: 198.8 (4032 read on 0 cached) | 2.1 | +5.9 tok/s | +3.1 % | negligible |
| 8,192 | c | ca: 221.6 (4612 read on 3516 cached); cb: 212.7 (4612 read on 3516 cached) | 9.0 | control | | |
|  | off | offa: 215.9 (4612 read on 3516 cached); offb: 220.8 (4612 read on 3516 cached) | 4.9 | +1.2 tok/s | +0.6 % | negligible |

### The draft and the step

Per load, pass 1: draft tokens accepted of drafted (the share), tokens a step (generated tokens over steps, a step being one pass of the main model: generated minus accepted), and the time of a step. Pass 2 is listed where its counters differ.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | c | ca: 147 of 321 (0.458), 2.35 a step, 105.4 ms a step (pass 2: 148 of 318 (0.465), 2.37 a step, 98.1 ms a step); cb: 147 of 321 (0.458), 2.35 a step, 102.7 ms a step (pass 2: 148 of 318 (0.465), 2.37 a step, 98.1 ms a step) |
|  | off | offa: 147 of 321 (0.458), 2.35 a step, 102.4 ms a step (pass 2: 148 of 318 (0.465), 2.37 a step, 98.1 ms a step); offb: 147 of 321 (0.458), 2.35 a step, 102.4 ms a step (pass 2: 148 of 318 (0.465), 2.37 a step, 98.1 ms a step) |
| 4,096 | c | ca: 177 of 231 (0.766), 3.24 a step, 114.4 ms a step; cb: 177 of 231 (0.766), 3.24 a step, 117.2 ms a step |
|  | off | offa: 177 of 231 (0.766), 3.24 a step, 113.1 ms a step; offb: 177 of 231 (0.766), 3.24 a step, 113.0 ms a step |
| 8,192 | c | ca: 173 of 245 (0.706), 3.08 a step, 120.5 ms a step (pass 2: 178 of 230 (0.774), 3.28 a step, 113.2 ms a step); cb: 173 of 245 (0.706), 3.08 a step, 120.2 ms a step (pass 2: 178 of 230 (0.774), 3.28 a step, 113.0 ms a step) |
|  | off | offa: 173 of 245 (0.706), 3.08 a step, 120.6 ms a step (pass 2: 178 of 230 (0.774), 3.28 a step, 113.3 ms a step); offb: 173 of 245 (0.706), 3.08 a step, 120.7 ms a step (pass 2: 178 of 230 (0.774), 3.28 a step, 113.3 ms a step) |

### The sampler pass

One request per depth and load with the server line's own sampler and a fixed seed: tok/s, the draft counters and the step.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | c | ca: 26.19 tok/s, 159 of 285 (0.558), 2.64 a step, 100.8 ms a step; cb: 26.22 tok/s, 159 of 285 (0.558), 2.64 a step, 100.7 ms a step |
|  | off | offa: 26.15 tok/s, 159 of 285 (0.558), 2.64 a step, 100.9 ms a step; offb: 26.21 tok/s, 159 of 285 (0.558), 2.64 a step, 100.7 ms a step |
| 4,096 | c | ca: 28.97 tok/s, 177 of 232 (0.763), 3.24 a step, 111.8 ms a step; cb: 28.98 tok/s, 177 of 232 (0.763), 3.24 a step, 111.8 ms a step |
|  | off | offa: 26.21 tok/s, 177 of 232 (0.763), 3.24 a step, 123.6 ms a step; offb: 28.96 tok/s, 177 of 232 (0.763), 3.24 a step, 111.9 ms a step |
| 8,192 | c | ca: 30.01 tok/s, 181 of 222 (0.815), 3.41 a step, 113.7 ms a step; cb: 29.90 tok/s, 181 of 222 (0.815), 3.41 a step, 114.1 ms a step |
|  | off | offa: 27.31 tok/s, 181 of 222 (0.815), 3.41 a step, 125.0 ms a step; offb: 29.81 tok/s, 181 of 222 (0.815), 3.41 a step, 114.5 ms a step |

### Checks

- Times: every load started on a machine checked quiet (valid=1).
- Prompts: 64 ids at depth 0, 4096 ids at depth 4,096, 8192 ids at depth 8,192; the same ids on every load and pass.
- Tokens out: 256 on every timed request.
- Greedy reply at depth 0: the same on every pass and load of every arm.
- Greedy reply at depth 4,096: the same on every pass and load of every arm.
- Greedy reply at depth 8,192: the same on every pass and load of every arm.
- Primes (read on cached, the cache's source): 4,096: 4032 on 0 (none); 8,192: 4612 on 3516 (checkpoint).
- Graph reallocations of the target, requests with one per load: ca: 0 of 12 requests; offa: 0 of 12 requests; offb: 0 of 12 requests; cb: 0 of 12 requests.
