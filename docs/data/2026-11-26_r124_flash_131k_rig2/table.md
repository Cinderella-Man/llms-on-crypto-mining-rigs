## The deep lane: draft, nodraft

Model file `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf`, llama-server version: 1.0.0-dev (build 13218, commit eee615cdb9).

256 tokens per request at temperature 0, each request twice per load (pass 1 is the first request at a new depth, pass 2 the same request again). Timed loads, in their order: draft1, nodraft1, nodraft2, draft2.

| Arm | Loads | Draft | Context | Its own variables and flags |
|---|---|---|---:|---|
| draft | draft1, draft2 | on | 131072 | — |
| nodraft (control) | nodraft1, nodraft2 | off | 131072 | — |

### Generation, tok/s

Per load: pass 1, pass 2 → the load, with the load's time per token. A load's speed is its two passes' tokens over their time. Spread: the difference between the arm's fastest and slowest load. Class against `nodraft`: gain when the mean is above the control's by more than the larger of the two spreads, loss when below by more than it, else negligible.

| Depth (tokens) | Arm | Per load | Spread | Against `nodraft` | | Class |
|---:|---|---|---:|---:|---:|---|
| 49,152 | draft | draft1: 28.07, 29.20 → **28.63** (34.93 ms a token); draft2: 26.19, 29.16 → **27.60** (36.24 ms a token) | 1.03 | +8.92 tok/s | +46.5 % | gain |
|  | nodraft | nodraft1: 19.01, 19.39 → **19.20** (52.08 ms a token); nodraft2: 18.98, 19.37 → **19.17** (52.16 ms a token) | 0.03 | control | | |
| 65,536 | draft | draft1: 30.58, 31.12 → **30.85** (32.41 ms a token); draft2: 30.59, 31.10 → **30.84** (32.42 ms a token) | 0.01 | +11.88 tok/s | +62.7 % | gain |
|  | nodraft | nodraft1: 18.83, 19.13 → **18.98** (52.69 ms a token); nodraft2: 18.81, 19.09 → **18.95** (52.77 ms a token) | 0.03 | control | | |
| 120,000 | draft | draft1: 23.53, 24.45 → **23.98** (41.70 ms a token); draft2: 23.56, 24.43 → **23.99** (41.69 ms a token) | 0.01 | +6.12 tok/s | +34.3 % | gain |
|  | nodraft | nodraft1: 17.61, 18.17 → **17.88** (55.92 ms a token); nodraft2: 17.53, 18.16 → **17.84** (56.06 ms a token) | 0.04 | control | | |

### Generation over all depths, tok/s

Every depth's tokens over their time, per load.

| Depth (tokens) | Arm | Per load | Spread | Against `nodraft` | | Class |
|---:|---|---|---:|---:|---:|---|
| all | draft | draft1: **27.51**; draft2: **27.19** | 0.32 | +8.70 tok/s | +46.6 % | gain |
|  | nodraft | nodraft1: **18.67**; nodraft2: **18.63** | 0.03 | control | | |

### The step on pass 1 (the first request at a new depth), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `nodraft`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `nodraft` | | Class |
|---:|---|---|---:|---:|---:|---|
| 49,152 | draft | draft1: 116.91 (78 steps); draft2: 125.33 (78 steps) | 8.42 | +68.48 ms | +130.1 % | no class (the draft differs) |
|  | nodraft | nodraft1: 52.60 (256 steps); nodraft2: 52.69 (256 steps) | 0.09 | control | | |
| 65,536 | draft | draft1: 117.90 (71 steps); draft2: 117.87 (71 steps) | 0.03 | +64.75 ms | +121.9 % | no class (the draft differs) |
|  | nodraft | nodraft1: 53.10 (256 steps); nodraft2: 53.18 (256 steps) | 0.08 | control | | |
| 120,000 | draft | draft1: 129.52 (84 steps); draft2: 129.36 (84 steps) | 0.16 | +72.52 ms | +127.4 % | no class (the draft differs) |
|  | nodraft | nodraft1: 56.80 (256 steps); nodraft2: 57.05 (256 steps) | 0.25 | control | | |

### The step on pass 2 (the same request again), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `nodraft`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `nodraft` | | Class |
|---:|---|---|---:|---:|---:|---|
| 49,152 | draft | draft1: 112.38 (78 steps); draft2: 112.54 (78 steps) | 0.16 | +60.86 ms | +118.0 % | no class (the draft differs) |
|  | nodraft | nodraft1: 51.57 (256 steps); nodraft2: 51.63 (256 steps) | 0.06 | control | | |
| 65,536 | draft | draft1: 115.85 (71 steps); draft2: 115.93 (71 steps) | 0.08 | +63.56 ms | +121.5 % | no class (the draft differs) |
|  | nodraft | nodraft1: 52.29 (256 steps); nodraft2: 52.37 (256 steps) | 0.09 | control | | |
| 120,000 | draft | draft1: 126.16 (83 steps); draft2: 126.23 (83 steps) | 0.07 | +71.14 ms | +129.2 % | no class (the draft differs) |
|  | nodraft | nodraft1: 55.05 (256 steps); nodraft2: 55.07 (256 steps) | 0.03 | control | | |

### Prompt reading, tok/s

Per load: the tokens read at that depth over their time (a prime request reads a chunk on top of the cache; without primes each decode request reads its prompt), with the tokens read and the tokens reused. Requests that read under 32 tokens are left out.

| Depth (tokens) | Arm | Per load | Spread | Against `nodraft` | | Class |
|---:|---|---|---:|---:|---:|---|
| 49,152 | draft | draft1: 187.3 (49028 read on 60 cached); draft2: 187.1 (49028 read on 60 cached) | 0.2 | -45.8 tok/s | -19.7 % | loss |
|  | nodraft | nodraft1: 232.9 (49028 read on 60 cached); nodraft2: 233.0 (49028 read on 60 cached) | 0.1 | control | | |
| 65,536 | draft | draft1: 128.8 (16644 read on 48828 cached); draft2: 129.1 (16644 read on 48828 cached) | 0.3 | -25.3 tok/s | -16.4 % | loss |
|  | nodraft | nodraft1: 154.1 (16644 read on 48828 cached); nodraft2: 154.3 (16644 read on 48828 cached) | 0.2 | control | | |
| 120,000 | draft | draft1: 94.5 (54724 read on 65212 cached); draft2: 94.3 (54724 read on 65212 cached) | 0.2 | -23.7 tok/s | -20.1 % | loss |
|  | nodraft | nodraft1: 118.1 (54724 read on 65212 cached); nodraft2: 118.0 (54724 read on 65212 cached) | 0.1 | control | | |

### The draft and the step

Per load, pass 1: draft tokens accepted of drafted (the share), tokens a step (generated tokens over steps, a step being one pass of the main model: generated minus accepted), and the time of a step. Pass 2 is listed where its counters differ.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 49,152 | draft | draft1: 178 of 231 (0.771), 3.28 a step, 116.9 ms a step; draft2: 178 of 231 (0.771), 3.28 a step, 125.3 ms a step |
|  | nodraft | nodraft1: no draft, 1.00 a step, 52.6 ms a step; nodraft2: no draft, 1.00 a step, 52.7 ms a step |
| 65,536 | draft | draft1: 185 of 210 (0.881), 3.61 a step, 117.9 ms a step; draft2: 185 of 210 (0.881), 3.61 a step, 117.9 ms a step |
|  | nodraft | nodraft1: no draft, 1.00 a step, 53.1 ms a step; nodraft2: no draft, 1.00 a step, 53.2 ms a step |
| 120,000 | draft | draft1: 172 of 249 (0.691), 3.05 a step, 129.5 ms a step (pass 2: 173 of 246 (0.703), 3.08 a step, 126.2 ms a step); draft2: 172 of 249 (0.691), 3.05 a step, 129.4 ms a step (pass 2: 173 of 246 (0.703), 3.08 a step, 126.2 ms a step) |
|  | nodraft | nodraft1: no draft, 1.00 a step, 56.8 ms a step; nodraft2: no draft, 1.00 a step, 57.0 ms a step |

### The sampler pass

One request per depth and load with the server line's own sampler and a fixed seed: tok/s, the draft counters and the step.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 49,152 | draft | draft1: 29.39 tok/s, 180 of 222 (0.811), 3.37 a step, 114.6 ms a step; draft2: 26.65 tok/s, 180 of 222 (0.811), 3.37 a step, 126.4 ms a step |
|  | nodraft | nodraft1: 19.24 tok/s, no draft, 1.00 a step, 52.0 ms a step; nodraft2: 19.12 tok/s, no draft, 1.00 a step, 52.3 ms a step |
| 65,536 | draft | draft1: 26.44 tok/s, 174 of 240 (0.725), 3.12 a step, 118.1 ms a step; draft2: 26.46 tok/s, 174 of 240 (0.725), 3.12 a step, 118.0 ms a step |
|  | nodraft | nodraft1: 18.84 tok/s, no draft, 1.00 a step, 53.1 ms a step; nodraft2: 18.83 tok/s, no draft, 1.00 a step, 53.1 ms a step |
| 120,000 | draft | draft1: 28.78 tok/s, 187 of 203 (0.921), 3.71 a step, 128.9 ms a step; draft2: 28.69 tok/s, 187 of 203 (0.921), 3.71 a step, 129.3 ms a step |
|  | nodraft | nodraft1: 17.56 tok/s, no draft, 1.00 a step, 57.0 ms a step; nodraft2: 17.55 tok/s, no draft, 1.00 a step, 57.0 ms a step |

### Checks

- Times: every load started on a machine checked quiet (valid=1).
- Prompts: 49152 ids at depth 49,152, 65536 ids at depth 65,536, 120000 ids at depth 120,000; the same ids on every load and pass.
- Tokens out: 256 on every timed request.
- Greedy reply at depth 49,152: **differs from `nodraft` in draft.**
- Greedy reply at depth 65,536: **differs from `nodraft` in draft.**
- Greedy reply at depth 120,000: **differs from `nodraft` in draft.**
- Primes (read on cached, the cache's source): 49,152: 49028 on 60 (checkpoint); 65,536: 16644 on 48828 (checkpoint); 120,000: 54724 on 65212 (checkpoint).
- Graph reallocations of the target, requests with one per load: draft1: 0 of 13 requests; nodraft1: 0 of 13 requests; nodraft2: 0 of 13 requests; draft2: 0 of 13 requests.
