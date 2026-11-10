## The deep lane: nodraft, draft

Model file `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf`, llama-server version: 0.8.0-dev (build 12358, commit 1551f5fe1).

256 tokens per request at temperature 0, each request twice per load (pass 1 is the first request at a new depth, pass 2 the same request again). Timed loads, in their order: nodraft1, draft1.

| Arm | Loads | Draft | Context | Its own variables and flags |
|---|---|---|---:|---|
| nodraft (control) | nodraft1 | off | 131072 | — |
| draft | draft1 | on | 131072 | — |

### Generation, tok/s

Per load: pass 1, pass 2 → the load, with the load's time per token. A load's speed is its two passes' tokens over their time. Spread: the difference between the arm's fastest and slowest load. Class against `nodraft`: gain when the mean is above the control's by more than the larger of the two spreads, loss when below by more than it, else negligible.

| Depth (tokens) | Arm | Per load | Spread | Against `nodraft` | | Class |
|---:|---|---|---:|---:|---:|---|
| 49,152 | nodraft | nodraft1: 14.12, 14.55 → **14.33** (69.78 ms a token) | — | control | | |
|  | draft | draft1: 15.01, 18.05 → **16.39** (61.01 ms a token) | — | +2.06 tok/s | +14.4 % | no class (one load) |
| 65,536 | nodraft | nodraft1: 13.39, 13.55 → **13.47** (74.25 ms a token) | — | control | | |
|  | draft | draft1: 17.09, 18.82 → **17.92** (55.82 ms a token) | — | +4.45 tok/s | +33.0 % | no class (one load) |
| 120,000 | nodraft | nodraft1: 10.85, 11.07 → **10.96** (91.25 ms a token) | — | control | | |
|  | draft | draft1: 15.20, 15.29 → **15.24** (65.60 ms a token) | — | +4.28 tok/s | +39.1 % | no class (one load) |

### Generation over all depths, tok/s

Every depth's tokens over their time, per load.

| Depth (tokens) | Arm | Per load | Spread | Against `nodraft` | | Class |
|---:|---|---|---:|---:|---:|---|
| all | nodraft | nodraft1: **12.75** | — | control | | |
|  | draft | draft1: **16.44** | — | +3.69 tok/s | +29.0 % | no class (one load) |

### The step on pass 1 (the first request at a new depth), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `nodraft`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `nodraft` | | Class |
|---:|---|---|---:|---:|---:|---|
| 49,152 | nodraft | nodraft1: 70.84 (256 steps) | — | control | | |
|  | draft | draft1: 218.69 (78 steps) | — | +147.84 ms | +208.7 % | no class (the draft differs) |
| 65,536 | nodraft | nodraft1: 74.68 (256 steps) | — | control | | |
|  | draft | draft1: 217.07 (69 steps) | — | +142.39 ms | +190.7 % | no class (the draft differs) |
| 120,000 | nodraft | nodraft1: 92.15 (256 steps) | — | control | | |
|  | draft | draft1: 251.36 (67 steps) | — | +159.20 ms | +172.8 % | no class (the draft differs) |

### The step on pass 2 (the same request again), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `nodraft`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `nodraft` | | Class |
|---:|---|---|---:|---:|---:|---|
| 49,152 | nodraft | nodraft1: 68.72 (256 steps) | — | control | | |
|  | draft | draft1: 181.81 (78 steps) | — | +113.09 ms | +164.6 % | no class (the draft differs) |
| 65,536 | nodraft | nodraft1: 73.81 (256 steps) | — | control | | |
|  | draft | draft1: 197.12 (69 steps) | — | +123.30 ms | +167.0 % | no class (the draft differs) |
| 120,000 | nodraft | nodraft1: 90.34 (256 steps) | — | control | | |
|  | draft | draft1: 249.94 (67 steps) | — | +159.60 ms | +176.7 % | no class (the draft differs) |

### Prompt reading, tok/s

Per load: the tokens read at that depth over their time (a prime request reads a chunk on top of the cache; without primes each decode request reads its prompt), with the tokens read and the tokens reused. Requests that read under 32 tokens are left out.

| Depth (tokens) | Arm | Per load | Spread | Against `nodraft` | | Class |
|---:|---|---|---:|---:|---:|---|
| 49,152 | nodraft | nodraft1: 146.2 (49028 read on 60 cached) | — | control | | |
|  | draft | draft1: 100.5 (49028 read on 60 cached) | — | -45.7 tok/s | -31.3 % | no class (one load) |
| 65,536 | nodraft | nodraft1: 93.7 (16644 read on 48828 cached) | — | control | | |
|  | draft | draft1: 66.2 (16644 read on 48828 cached) | — | -27.4 tok/s | -29.3 % | no class (one load) |
| 120,000 | nodraft | nodraft1: 73.6 (54724 read on 65212 cached) | — | control | | |
|  | draft | draft1: 49.0 (54724 read on 65212 cached) | — | -24.6 tok/s | -33.5 % | no class (one load) |

### The draft and the step

Per load, pass 1: draft tokens accepted of drafted (the share), tokens a step (generated tokens over steps, a step being one pass of the main model: generated minus accepted), and the time of a step. Pass 2 is listed where its counters differ.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 49,152 | nodraft | nodraft1: no draft, 1.00 a step, 70.8 ms a step |
|  | draft | draft1: 178 of 231 (0.771), 3.28 a step, 218.7 ms a step |
| 65,536 | nodraft | nodraft1: no draft, 1.00 a step, 74.7 ms a step |
|  | draft | draft1: 187 of 201 (0.930), 3.71 a step, 217.1 ms a step |
| 120,000 | nodraft | nodraft1: no draft, 1.00 a step, 92.2 ms a step |
|  | draft | draft1: 189 of 196 (0.964), 3.82 a step, 251.4 ms a step |

### Checks

- Times: every load started on a machine checked quiet (valid=1).
- Prompts: 49152 ids at depth 49,152, 65536 ids at depth 65,536, 120000 ids at depth 120,000; the same ids on every load and pass.
- Tokens out: 256 on every timed request.
- Greedy reply at depth 49,152: **differs from `nodraft` in draft.**
- Greedy reply at depth 65,536: **differs from `nodraft` in draft.**
- Greedy reply at depth 120,000: **differs from `nodraft` in draft.**
- Primes (read on cached, the cache's source): 49,152: 49028 on 60 (checkpoint); 65,536: 16644 on 48828 (checkpoint); 120,000: 54724 on 65212 (checkpoint).
