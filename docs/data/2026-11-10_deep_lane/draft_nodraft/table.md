## The deep lane: draft, nodraft

Model file `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf`, llama-server version: 0.8.0-dev (build 12358, commit 1551f5fe1).

256 tokens per request at temperature 0, each request twice per load (pass 1 is the first request at a new depth, pass 2 the same request again). Timed loads, in their order: draft1, nodraft1, draft2, nodraft2.

| Arm | Loads | Draft | Context | Its own variables and flags |
|---|---|---|---:|---|
| draft (control) | draft1, draft2 | on | 65536 | — |
| nodraft | nodraft1, nodraft2 | off | 65536 | — |

### Generation, tok/s

Per load: pass 1, pass 2 → the load, with the load's time per token. A load's speed is its two passes' tokens over their time. Spread: the difference between the arm's fastest and slowest load. Class against `draft`: gain when the mean is above the control's by more than the larger of the two spreads, loss when below by more than it, else negligible.

| Depth (tokens) | Arm | Per load | Spread | Against `draft` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | draft | draft1: 26.38, 27.71 → **27.03** (37.00 ms a token); draft2: 25.79, 27.69 → **26.71** (37.44 ms a token) | 0.32 | control | | |
|  | nodraft | nodraft1: 19.04, 19.49 → **19.26** (51.92 ms a token); nodraft2: 19.04, 19.48 → **19.26** (51.92 ms a token) | 0.00 | -7.61 tok/s | -28.3 % | loss |
| 8,192 | draft | draft1: 11.57, 20.16 → **14.70** (68.02 ms a token); draft2: 17.16, 24.11 → **20.05** (49.88 ms a token) | 5.35 | control | | |
|  | nodraft | nodraft1: 16.03, 17.82 → **16.87** (59.26 ms a token); nodraft2: 16.22, 17.82 → **16.98** (58.89 ms a token) | 0.11 | -0.45 tok/s | -2.6 % | negligible |
| 16,384 | draft | draft1: 18.43, 24.06 → **20.87** (47.91 ms a token); draft2: 20.61, 23.92 → **22.14** (45.16 ms a token) | 1.27 | control | | |
|  | nodraft | nodraft1: 16.66, 16.94 → **16.80** (59.53 ms a token); nodraft2: 16.67, 16.96 → **16.81** (59.47 ms a token) | 0.02 | -4.70 tok/s | -21.9 % | loss |
| 32,768 | draft | draft1: 16.80, 18.86 → **17.77** (56.27 ms a token); draft2: 12.76, 15.45 → **13.98** (71.55 ms a token) | 3.79 | control | | |
|  | nodraft | nodraft1: 15.26, 15.65 → **15.45** (64.71 ms a token); nodraft2: 14.96, 15.43 → **15.19** (65.82 ms a token) | 0.26 | -0.55 tok/s | -3.5 % | negligible |
| 49,152 | draft | draft1: 17.86, 17.98 → **17.92** (55.80 ms a token); draft2: 16.83, 17.45 → **17.13** (58.36 ms a token) | 0.79 | control | | |
|  | nodraft | nodraft1: 14.03, 14.53 → **14.28** (70.04 ms a token); nodraft2: 13.76, 14.54 → **14.14** (70.73 ms a token) | 0.14 | -3.32 tok/s | -18.9 % | loss |

### Generation over all depths, tok/s

Every depth's tokens over their time, per load.

| Depth (tokens) | Arm | Per load | Spread | Against `draft` | | Class |
|---:|---|---|---:|---:|---:|---|
| all | draft | draft1: **18.87**; draft2: **19.06** | 0.19 | control | | |
|  | nodraft | nodraft1: **16.37**; nodraft2: **16.30** | 0.07 | -2.63 tok/s | -13.9 % | loss |

### The step on pass 1 (the first request at a new depth), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `draft`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `draft` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | draft | draft1: 124.42 (78 steps); draft2: 127.24 (78 steps) | 2.82 | control | | |
|  | nodraft | nodraft1: 52.52 (256 steps); nodraft2: 52.53 (256 steps) | 0.00 | -73.31 ms | -58.3 % | no class (the draft differs) |
| 8,192 | draft | draft1: 248.63 (89 steps); draft2: 207.18 (72 steps) | 41.45 | control | | |
|  | nodraft | nodraft1: 62.39 (256 steps); nodraft2: 61.66 (256 steps) | 0.74 | -165.88 ms | -72.8 % | no class (the draft differs) |
| 16,384 | draft | draft1: 195.63 (71 steps); draft2: 174.94 (71 steps) | 20.70 | control | | |
|  | nodraft | nodraft1: 60.01 (256 steps); nodraft2: 59.99 (256 steps) | 0.02 | -125.28 ms | -67.6 % | no class (the draft differs) |
| 32,768 | draft | draft1: 185.78 (82 steps); draft2: 194.76 (103 steps) | 8.97 | control | | |
|  | nodraft | nodraft1: 65.54 (256 steps); nodraft2: 66.84 (256 steps) | 1.31 | -124.08 ms | -65.2 % | no class (the draft differs) |
| 49,152 | draft | draft1: 186.13 (77 steps); draft2: 187.79 (81 steps) | 1.66 | control | | |
|  | nodraft | nodraft1: 71.27 (256 steps); nodraft2: 72.66 (256 steps) | 1.38 | -115.00 ms | -61.5 % | no class (the draft differs) |

### The step on pass 2 (the same request again), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `draft`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `draft` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | draft | draft1: 123.20 (75 steps); draft2: 123.26 (75 steps) | 0.06 | control | | |
|  | nodraft | nodraft1: 51.32 (256 steps); nodraft2: 51.32 (256 steps) | 0.00 | -71.91 ms | -58.4 % | no class (the draft differs) |
| 8,192 | draft | draft1: 151.17 (84 steps); draft2: 149.56 (71 steps) | 1.60 | control | | |
|  | nodraft | nodraft1: 56.13 (256 steps); nodraft2: 56.13 (256 steps) | 0.00 | -94.24 ms | -62.7 % | no class (the draft differs) |
| 16,384 | draft | draft1: 149.89 (71 steps); draft2: 150.72 (71 steps) | 0.83 | control | | |
|  | nodraft | nodraft1: 59.05 (256 steps); nodraft2: 58.95 (256 steps) | 0.10 | -91.31 ms | -60.7 % | no class (the draft differs) |
| 32,768 | draft | draft1: 165.57 (82 steps); draft2: 165.72 (100 steps) | 0.15 | control | | |
|  | nodraft | nodraft1: 63.88 (256 steps); nodraft2: 64.80 (256 steps) | 0.92 | -101.30 ms | -61.2 % | no class (the draft differs) |
| 49,152 | draft | draft1: 184.90 (77 steps); draft2: 181.11 (81 steps) | 3.79 | control | | |
|  | nodraft | nodraft1: 68.81 (256 steps); nodraft2: 68.80 (256 steps) | 0.01 | -114.20 ms | -62.4 % | no class (the draft differs) |

### Prompt reading, tok/s

Per load: the tokens read at that depth over their time (a prime request reads a chunk on top of the cache; without primes each decode request reads its prompt), with the tokens read and the tokens reused. Requests that read under 32 tokens are left out.

| Depth (tokens) | Arm | Per load | Spread | Against `draft` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | draft | draft1: 32.9 (64 read on 0 cached); draft2: 32.7 (64 read on 0 cached) | 0.2 | control | | |
|  | nodraft | nodraft1: 33.4 (64 read on 0 cached); nodraft2: 33.1 (64 read on 0 cached) | 0.4 | +0.4 tok/s | +1.4 % | gain |
| 8,192 | draft | draft1: 127.7 (8128 read on 0 cached); draft2: 127.2 (8128 read on 0 cached) | 0.5 | control | | |
|  | nodraft | nodraft1: 179.4 (8128 read on 0 cached); nodraft2: 176.6 (8128 read on 0 cached) | 2.8 | +50.6 tok/s | +39.7 % | gain |
| 16,384 | draft | draft1: 107.6 (8708 read on 7612 cached); draft2: 109.3 (8708 read on 7612 cached) | 1.7 | control | | |
|  | nodraft | nodraft1: 164.9 (8708 read on 7612 cached); nodraft2: 166.6 (8708 read on 7612 cached) | 1.8 | +57.3 tok/s | +52.8 % | gain |
| 32,768 | draft | draft1: 90.3 (16388 read on 16316 cached); draft2: 91.0 (16388 read on 16316 cached) | 0.7 | control | | |
|  | nodraft | nodraft1: 148.3 (16388 read on 16316 cached); nodraft2: 148.1 (16388 read on 16316 cached) | 0.3 | +57.5 tok/s | +63.4 % | gain |
| 49,152 | draft | draft1: 69.9 (16900 read on 32188 cached); draft2: 70.3 (16900 read on 32188 cached) | 0.4 | control | | |
|  | nodraft | nodraft1: 119.5 (16900 read on 32188 cached); nodraft2: 119.4 (16900 read on 32188 cached) | 0.1 | +49.3 tok/s | +70.4 % | gain |

### The draft and the step

Per load, pass 1: draft tokens accepted of drafted (the share), tokens a step (generated tokens over steps, a step being one pass of the main model: generated minus accepted), and the time of a step. Pass 2 is listed where its counters differ.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | draft | draft1: 178 of 229 (0.777), 3.28 a step, 124.4 ms a step (pass 2: 181 of 220 (0.823), 3.41 a step, 123.2 ms a step); draft2: 178 of 229 (0.777), 3.28 a step, 127.2 ms a step (pass 2: 181 of 220 (0.823), 3.41 a step, 123.3 ms a step) |
|  | nodraft | nodraft1: no draft, 1.00 a step, 52.5 ms a step; nodraft2: no draft, 1.00 a step, 52.5 ms a step |
| 8,192 | draft | draft1: 167 of 262 (0.637), 2.88 a step, 248.6 ms a step (pass 2: 172 of 247 (0.696), 3.05 a step, 151.2 ms a step); draft2: 184 of 210 (0.876), 3.56 a step, 207.2 ms a step (pass 2: 185 of 207 (0.894), 3.61 a step, 149.6 ms a step) |
|  | nodraft | nodraft1: no draft, 1.00 a step, 62.4 ms a step; nodraft2: no draft, 1.00 a step, 61.7 ms a step |
| 16,384 | draft | draft1: 185 of 207 (0.894), 3.61 a step, 195.6 ms a step; draft2: 185 of 210 (0.881), 3.61 a step, 174.9 ms a step |
|  | nodraft | nodraft1: no draft, 1.00 a step, 60.0 ms a step; nodraft2: no draft, 1.00 a step, 60.0 ms a step |
| 32,768 | draft | draft1: 174 of 242 (0.719), 3.12 a step, 185.8 ms a step; draft2: 153 of 305 (0.502), 2.49 a step, 194.8 ms a step (pass 2: 156 of 296 (0.527), 2.56 a step, 165.7 ms a step) |
|  | nodraft | nodraft1: no draft, 1.00 a step, 65.5 ms a step; nodraft2: no draft, 1.00 a step, 66.8 ms a step |
| 49,152 | draft | draft1: 179 of 226 (0.792), 3.32 a step, 186.1 ms a step; draft2: 175 of 240 (0.729), 3.16 a step, 187.8 ms a step |
|  | nodraft | nodraft1: no draft, 1.00 a step, 71.3 ms a step; nodraft2: no draft, 1.00 a step, 72.7 ms a step |

### Checks

- Times: every load started on a machine checked quiet (valid=1).
- Prompts: 64 ids at depth 0, 8192 ids at depth 8,192, 16384 ids at depth 16,384, 32768 ids at depth 32,768, 49152 ids at depth 49,152; the same ids on every load and pass.
- Tokens out: 256 on every timed request.
- Greedy reply at depth 0: **differs from `draft` in nodraft.**
- Greedy reply at depth 8,192: 4 different replies over the 4 loads (`draft`: 2 over its 2; `nodraft`: 2 over its 2); every load wrote its own reply again on pass 2.
- Greedy reply at depth 16,384: 4 different replies over the 4 loads (`draft`: 2 over its 2; `nodraft`: 2 over its 2); every load wrote its own reply again on pass 2.
- Greedy reply at depth 32,768: 5 different replies over the 4 loads (`draft`: 2 over its 2; `nodraft`: 3 over its 2); **nodraft2 wrote another reply on pass 2 than on pass 1.**
- Greedy reply at depth 49,152: 4 different replies over the 4 loads (`draft`: 2 over its 2; `nodraft`: 2 over its 2); every load wrote its own reply again on pass 2.
- Primes (read on cached, the cache's source): 8,192: 8128 on 0 (none); 16,384: 8708 on 7612 (checkpoint); 32,768: 16388 on 16316 (checkpoint); 49,152: 16900 on 32188 (checkpoint).
