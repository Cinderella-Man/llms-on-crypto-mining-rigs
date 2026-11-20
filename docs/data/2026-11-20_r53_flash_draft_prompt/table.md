## The deep lane: draft, nodraft

Model file `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf`, llama-server version: 1.0.0-dev (build 12967, commit 6026911533).

16 tokens per request at temperature 0, each request twice per load (pass 1 is the first request at a new depth, pass 2 the same request again). Timed loads, in their order: d1, n1, n2, d2.

| Arm | Loads | Draft | Context | Its own variables and flags |
|---|---|---|---:|---|
| draft (control) | d1, d2 | on | 65536 | — |
| nodraft | n1, n2 | off | 65536 | — |

### Generation, tok/s

Per load: pass 1, pass 2 → the load, with the load's time per token. A load's speed is its two passes' tokens over their time. Spread: the difference between the arm's fastest and slowest load. Class against `draft`: gain when the mean is above the control's by more than the larger of the two spreads, loss when below by more than it, else negligible.

| Depth (tokens) | Arm | Per load | Spread | Against `draft` | | Class |
|---:|---|---|---:|---:|---:|---|
| 8,192 | draft | d1: 13.35, 17.31 → **15.07** (66.34 ms a token); d2: 13.31, 17.28 → **15.03** (66.51 ms a token) | 0.04 | control | | |
|  | nodraft | n1: 16.43, 19.29 → **17.75** (56.35 ms a token); n2: 16.40, 19.27 → **17.72** (56.43 ms a token) | 0.02 | +2.68 tok/s | +17.8 % | gain |
| 16,384 | draft | d1: 16.71, 23.18 → **19.42** (51.50 ms a token); d2: 16.67, 23.22 → **19.41** (51.53 ms a token) | 0.01 | control | | |
|  | nodraft | n1: 16.61, 19.10 → **17.77** (56.27 ms a token); n2: 16.70, 19.09 → **17.82** (56.12 ms a token) | 0.05 | -1.62 tok/s | -8.3 % | loss |
| 32,768 | draft | d1: 14.09, 16.59 → **15.24** (65.62 ms a token); d2: 14.02, 20.52 → **16.66** (60.03 ms a token) | 1.42 | control | | |
|  | nodraft | n1: 15.93, 18.76 → **17.23** (58.03 ms a token); n2: 15.90, 18.76 → **17.21** (58.09 ms a token) | 0.02 | +1.27 tok/s | +8.0 % | negligible |

### Generation over all depths, tok/s

Every depth's tokens over their time, per load.

| Depth (tokens) | Arm | Per load | Spread | Against `draft` | | Class |
|---:|---|---|---:|---:|---:|---|
| all | draft | d1: **16.35**; d2: **16.85** | 0.49 | control | | |
|  | nodraft | n1: **17.58**; n2: **17.58** | 0.00 | +0.98 tok/s | +5.9 % | gain |

### The step on pass 1 (the first request at a new depth), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `draft`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `draft` | | Class |
|---:|---|---|---:|---:|---:|---|
| 8,192 | draft | d1: 133.19 (9 steps); d2: 133.60 (9 steps) | 0.42 | control | | |
|  | nodraft | n1: 60.87 (16 steps); n2: 60.96 (16 steps) | 0.09 | -72.48 ms | -54.3 % | no class (the draft differs) |
| 16,384 | draft | d1: 136.79 (7 steps); d2: 137.11 (7 steps) | 0.33 | control | | |
|  | nodraft | n1: 60.19 (16 steps); n2: 59.87 (16 steps) | 0.32 | -76.92 ms | -56.2 % | no class (the draft differs) |
| 32,768 | draft | d1: 141.90 (8 steps); d2: 142.68 (8 steps) | 0.78 | control | | |
|  | nodraft | n1: 62.77 (16 steps); n2: 62.88 (16 steps) | 0.11 | -79.47 ms | -55.8 % | no class (the draft differs) |

### The step on pass 2 (the same request again), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `draft`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `draft` | | Class |
|---:|---|---|---:|---:|---:|---|
| 8,192 | draft | d1: 102.69 (9 steps); d2: 102.89 (9 steps) | 0.20 | control | | |
|  | nodraft | n1: 51.83 (16 steps); n2: 51.89 (16 steps) | 0.06 | -50.93 ms | -49.5 % | no class (the draft differs) |
| 16,384 | draft | d1: 98.62 (7 steps); d2: 98.46 (7 steps) | 0.17 | control | | |
|  | nodraft | n1: 52.36 (16 steps); n2: 52.37 (16 steps) | 0.02 | -46.17 ms | -46.9 % | no class (the draft differs) |
| 32,768 | draft | d1: 120.58 (8 steps); d2: 97.45 (8 steps) | 23.13 | control | | |
|  | nodraft | n1: 53.30 (16 steps); n2: 53.30 (16 steps) | 0.00 | -55.71 ms | -51.1 % | no class (the draft differs) |

### Prompt reading, tok/s

Per load: the tokens read at that depth over their time (a prime request reads a chunk on top of the cache; without primes each decode request reads its prompt), with the tokens read and the tokens reused. Requests that read under 32 tokens are left out.

| Depth (tokens) | Arm | Per load | Spread | Against `draft` | | Class |
|---:|---|---|---:|---:|---:|---|
| 8,192 | draft | d1: 193.5 (8068 read on 60 cached); d2: 193.6 (8068 read on 60 cached) | 0.0 | control | | |
|  | nodraft | n1: 226.5 (8068 read on 60 cached); n2: 222.8 (8068 read on 60 cached) | 3.7 | +31.1 tok/s | +16.1 % | gain |
| 16,384 | draft | d1: 319.5 (16260 read on 60 cached); d2: 309.9 (16260 read on 60 cached) | 9.6 | control | | |
|  | nodraft | n1: 327.8 (16260 read on 60 cached); n2: 326.6 (16260 read on 60 cached) | 1.2 | +12.5 tok/s | +4.0 % | gain |
| 32,768 | draft | d1: 225.9 (16900 read on 15804 cached); d2: 221.1 (16900 read on 15804 cached) | 4.8 | control | | |
|  | nodraft | n1: 232.6 (16900 read on 15804 cached); n2: 232.5 (16900 read on 15804 cached) | 0.1 | +9.1 tok/s | +4.1 % | gain |

### The draft and the step

Per load, pass 1: draft tokens accepted of drafted (the share), tokens a step (generated tokens over steps, a step being one pass of the main model: generated minus accepted), and the time of a step. Pass 2 is listed where its counters differ.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 8,192 | draft | d1: 7 of 23 (0.304), 1.78 a step, 133.2 ms a step; d2: 7 of 23 (0.304), 1.78 a step, 133.6 ms a step |
|  | nodraft | n1: no draft, 1.00 a step, 60.9 ms a step; n2: no draft, 1.00 a step, 61.0 ms a step |
| 16,384 | draft | d1: 9 of 16 (0.562), 2.29 a step, 136.8 ms a step; d2: 9 of 16 (0.562), 2.29 a step, 137.1 ms a step |
|  | nodraft | n1: no draft, 1.00 a step, 60.2 ms a step; n2: no draft, 1.00 a step, 59.9 ms a step |
| 32,768 | draft | d1: 8 of 16 (0.500), 2.00 a step, 141.9 ms a step; d2: 8 of 16 (0.500), 2.00 a step, 142.7 ms a step |
|  | nodraft | n1: no draft, 1.00 a step, 62.8 ms a step; n2: no draft, 1.00 a step, 62.9 ms a step |

### Checks

- Times: every load started on a machine checked quiet (valid=1).
- Prompts: 8192 ids at depth 8,192, 16384 ids at depth 16,384, 32768 ids at depth 32,768; the same ids on every load and pass.
- Tokens out: 16 on every timed request.
- Greedy reply at depth 8,192: **differs from `draft` in nodraft.**
- Greedy reply at depth 16,384: the same on every pass and load of every arm.
- Greedy reply at depth 32,768: **differs from `draft` in nodraft.**
- Primes (read on cached, the cache's source): 8,192: 8068 on 60 (checkpoint); 16,384: 16260 on 60 (checkpoint); 32,768: 16900 on 15804 (checkpoint).
- Graph reallocations of the target, requests with one per load: d1: 0 of 10 requests; n1: 0 of 10 requests; n2: 0 of 10 requests; d2: 0 of 10 requests.
