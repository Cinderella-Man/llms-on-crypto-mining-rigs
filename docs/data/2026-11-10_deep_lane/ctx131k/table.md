## The deep lane: nodraft, draft

Model file `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf`, llama-server version: 0.8.0-dev (build 12358, commit 1551f5fe1).

256 tokens per request at temperature 0, each request twice per load (pass 1 is the first request at a new depth, pass 2 the same request again). Timed loads, in their order: nodraft1, draft1, nodraft2, draft2.

| Arm | Loads | Draft | Context | Its own variables and flags |
|---|---|---|---:|---|
| nodraft (control) | nodraft1, nodraft2 | off | 131072 | — |
| draft | draft1, draft2 | on | 131072 | — |

### Generation, tok/s

Per load: pass 1, pass 2 → the load, with the load's time per token. A load's speed is its two passes' tokens over their time. Spread: the difference between the arm's fastest and slowest load. Class against `nodraft`: gain when the mean is above the control's by more than the larger of the two spreads, loss when below by more than it, else negligible.

| Depth (tokens) | Arm | Per load | Spread | Against `nodraft` | | Class |
|---:|---|---|---:|---:|---:|---|
| 49,152 | nodraft | nodraft1: 14.12, 14.55 → **14.33** (69.78 ms a token); nodraft2: 14.18, 14.51 → **14.34** (69.72 ms a token) | 0.01 | control | | |
|  | draft | draft1: 15.01, 18.05 → **16.39** (61.01 ms a token); draft2: 19.03, 19.44 → **19.23** (51.99 ms a token) | 2.85 | +3.48 tok/s | +24.2 % | gain |
| 65,536 | nodraft | nodraft1: 13.39, 13.55 → **13.47** (74.25 ms a token); nodraft2: 13.29, 13.51 → **13.40** (74.65 ms a token) | 0.07 | control | | |
|  | draft | draft1: 17.09, 18.82 → **17.92** (55.82 ms a token); draft2: 16.91, 18.50 → **17.67** (56.60 ms a token) | 0.25 | +4.36 tok/s | +32.4 % | gain |
| 120,000 | nodraft | nodraft1: 10.85, 11.07 → **10.96** (91.25 ms a token); nodraft2: 10.78, 11.03 → **10.90** (91.74 ms a token) | 0.06 | control | | |
|  | draft | draft1: 15.20, 15.29 → **15.24** (65.60 ms a token); draft2: 12.62, 12.80 → **12.71** (78.69 ms a token) | 2.54 | +3.05 tok/s | +27.9 % | gain |

### Generation over all depths, tok/s

Every depth's tokens over their time, per load.

| Depth (tokens) | Arm | Per load | Spread | Against `nodraft` | | Class |
|---:|---|---|---:|---:|---:|---|
| all | nodraft | nodraft1: **12.75**; nodraft2: **12.71** | 0.05 | control | | |
|  | draft | draft1: **16.44**; draft2: **16.02** | 0.43 | +3.50 tok/s | +27.5 % | gain |

### The step on pass 1 (the first request at a new depth), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `nodraft`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `nodraft` | | Class |
|---:|---|---|---:|---:|---:|---|
| 49,152 | nodraft | nodraft1: 70.84 (256 steps); nodraft2: 70.54 (256 steps) | 0.31 | control | | |
|  | draft | draft1: 218.69 (78 steps); draft2: 186.82 (72 steps) | 31.87 | +132.06 ms | +186.8 % | no class (the draft differs) |
| 65,536 | nodraft | nodraft1: 74.68 (256 steps); nodraft2: 75.25 (256 steps) | 0.58 | control | | |
|  | draft | draft1: 217.07 (69 steps); draft2: 216.29 (70 steps) | 0.78 | +141.72 ms | +189.0 % | no class (the draft differs) |
| 120,000 | nodraft | nodraft1: 92.15 (256 steps); nodraft2: 92.79 (256 steps) | 0.64 | control | | |
|  | draft | draft1: 251.36 (67 steps); draft2: 253.63 (80 steps) | 2.27 | +160.02 ms | +173.0 % | no class (the draft differs) |

### The step on pass 2 (the same request again), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `nodraft`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `nodraft` | | Class |
|---:|---|---|---:|---:|---:|---|
| 49,152 | nodraft | nodraft1: 68.72 (256 steps); nodraft2: 68.91 (256 steps) | 0.19 | control | | |
|  | draft | draft1: 181.81 (78 steps); draft2: 182.88 (72 steps) | 1.06 | +113.53 ms | +165.0 % | no class (the draft differs) |
| 65,536 | nodraft | nodraft1: 73.81 (256 steps); nodraft2: 74.05 (256 steps) | 0.23 | control | | |
|  | draft | draft1: 197.12 (69 steps); draft2: 197.73 (70 steps) | 0.61 | +123.49 ms | +167.0 % | no class (the draft differs) |
| 120,000 | nodraft | nodraft1: 90.34 (256 steps); nodraft2: 90.70 (256 steps) | 0.36 | control | | |
|  | draft | draft1: 249.94 (67 steps); draft2: 249.96 (80 steps) | 0.02 | +159.43 ms | +176.1 % | no class (the draft differs) |

### Prompt reading, tok/s

Per load: the tokens read at that depth over their time (a prime request reads a chunk on top of the cache; without primes each decode request reads its prompt), with the tokens read and the tokens reused. Requests that read under 32 tokens are left out.

| Depth (tokens) | Arm | Per load | Spread | Against `nodraft` | | Class |
|---:|---|---|---:|---:|---:|---|
| 49,152 | nodraft | nodraft1: 146.2 (49028 read on 60 cached); nodraft2: 145.6 (49028 read on 60 cached) | 0.6 | control | | |
|  | draft | draft1: 100.5 (49028 read on 60 cached); draft2: 100.0 (49028 read on 60 cached) | 0.5 | -45.7 tok/s | -31.3 % | loss |
| 65,536 | nodraft | nodraft1: 93.7 (16644 read on 48828 cached); nodraft2: 93.2 (16644 read on 48828 cached) | 0.5 | control | | |
|  | draft | draft1: 66.2 (16644 read on 48828 cached); draft2: 65.4 (16644 read on 48828 cached) | 0.9 | -27.6 tok/s | -29.6 % | loss |
| 120,000 | nodraft | nodraft1: 73.6 (54724 read on 65212 cached); nodraft2: 73.2 (54724 read on 65212 cached) | 0.4 | control | | |
|  | draft | draft1: 49.0 (54724 read on 65212 cached); draft2: 48.9 (54724 read on 65212 cached) | 0.1 | -24.5 tok/s | -33.3 % | loss |

### The draft and the step

Per load, pass 1: draft tokens accepted of drafted (the share), tokens a step (generated tokens over steps, a step being one pass of the main model: generated minus accepted), and the time of a step. Pass 2 is listed where its counters differ.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 49,152 | nodraft | nodraft1: no draft, 1.00 a step, 70.8 ms a step; nodraft2: no draft, 1.00 a step, 70.5 ms a step |
|  | draft | draft1: 178 of 231 (0.771), 3.28 a step, 218.7 ms a step; draft2: 184 of 211 (0.872), 3.56 a step, 186.8 ms a step |
| 65,536 | nodraft | nodraft1: no draft, 1.00 a step, 74.7 ms a step; nodraft2: no draft, 1.00 a step, 75.3 ms a step |
|  | draft | draft1: 187 of 201 (0.930), 3.71 a step, 217.1 ms a step; draft2: 186 of 204 (0.912), 3.66 a step, 216.3 ms a step |
| 120,000 | nodraft | nodraft1: no draft, 1.00 a step, 92.2 ms a step; nodraft2: no draft, 1.00 a step, 92.8 ms a step |
|  | draft | draft1: 189 of 196 (0.964), 3.82 a step, 251.4 ms a step; draft2: 176 of 237 (0.743), 3.20 a step, 253.6 ms a step |

### Checks

- Times: every load started on a machine checked quiet (valid=1).
- Prompts: 49152 ids at depth 49,152, 65536 ids at depth 65,536, 120000 ids at depth 120,000; the same ids on every load and pass.
- Tokens out: 256 on every timed request.
- Greedy reply at depth 49,152: 4 different replies over the 4 loads (`nodraft`: 2 over its 2; `draft`: 2 over its 2); every load wrote its own reply again on pass 2.
- Greedy reply at depth 65,536: 4 different replies over the 4 loads (`nodraft`: 2 over its 2; `draft`: 2 over its 2); every load wrote its own reply again on pass 2.
- Greedy reply at depth 120,000: 4 different replies over the 4 loads (`nodraft`: 2 over its 2; `draft`: 2 over its 2); every load wrote its own reply again on pass 2.
- Primes (read on cached, the cache's source): 49,152: 49028 on 60 (checkpoint); 65,536: 16644 on 48828 (checkpoint); 120,000: 54724 on 65212 (checkpoint).
