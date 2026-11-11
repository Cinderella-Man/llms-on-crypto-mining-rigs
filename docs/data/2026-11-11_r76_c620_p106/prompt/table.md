## The deep lane: draft, kvonly, nodraft

Model file `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf`, llama-server version: 0.9.0-dev (build 12581, commit c408dabcb3).

16 tokens per request at temperature 0, each request twice per load (pass 1 is the first request at a new depth, pass 2 the same request again). Timed loads, in their order: draft1, kvonly1, nodraft1, draft2, kvonly2, nodraft2.

| Arm | Loads | Draft | Context | Its own variables and flags |
|---|---|---|---:|---|
| draft (control) | draft1, draft2 | on | 65536 | — |
| kvonly | kvonly1, kvonly2 | on | 65536 | `LLAMA_MTP_KV_ONLY=1` |
| nodraft | nodraft1, nodraft2 | off | 65536 | — |

### Generation, tok/s

Per load: pass 1, pass 2 → the load, with the load's time per token. A load's speed is its two passes' tokens over their time. Spread: the difference between the arm's fastest and slowest load. Class against `draft`: gain when the mean is above the control's by more than the larger of the two spreads, loss when below by more than it, else negligible.

| Depth (tokens) | Arm | Per load | Spread | Against `draft` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | draft | draft1: 7.29, 16.00 → **10.01** (99.86 ms a token); draft2: 7.15, 15.98 → **9.88** (101.20 ms a token) | 0.13 | control | | |
|  | kvonly | kvonly1: 7.18, 16.23 → **9.96** (100.42 ms a token); kvonly2: 7.17, 16.23 → **9.95** (100.52 ms a token) | 0.01 | +0.01 tok/s | +0.1 % | negligible |
|  | nodraft | nodraft1: 19.74, 20.37 → **20.05** (49.87 ms a token); nodraft2: 19.83, 20.39 → **20.11** (49.74 ms a token) | 0.05 | +10.13 tok/s | +101.8 % | gain |
| 8,192 | draft | draft1: 6.92, 7.29 → **7.10** (140.87 ms a token); draft2: 6.48, 7.29 → **6.86** (145.76 ms a token) | 0.24 | control | | |
|  | kvonly | kvonly1: 6.99, 7.38 → **7.18** (139.27 ms a token); kvonly2: 6.98, 7.38 → **7.17** (139.40 ms a token) | 0.01 | +0.20 tok/s | +2.8 % | negligible |
|  | nodraft | nodraft1: 5.64, 17.44 → **8.53** (117.29 ms a token); nodraft2: 6.29, 11.92 → **8.23** (121.46 ms a token) | 0.29 | +1.40 tok/s | +20.1 % | gain |
| 16,384 | draft | draft1: 10.94, 15.81 → **12.93** (77.32 ms a token); draft2: 10.80, 17.31 → **13.30** (75.18 ms a token) | 0.37 | control | | |
|  | kvonly | kvonly1: 16.41, 17.63 → **17.00** (58.84 ms a token); kvonly2: 16.49, 17.65 → **17.05** (58.65 ms a token) | 0.06 | +3.91 tok/s | +29.8 % | gain |
|  | nodraft | nodraft1: 5.42, 17.66 → **8.30** (120.49 ms a token); nodraft2: 9.33, 13.59 → **11.06** (90.39 ms a token) | 2.76 | -3.44 tok/s | -26.2 % | loss |
| 32,768 | draft | draft1: 8.62, 14.42 → **10.79** (92.69 ms a token); draft2: 12.65, 14.74 → **13.62** (73.44 ms a token) | 2.83 | control | | |
|  | kvonly | kvonly1: 14.41, 15.04 → **14.72** (67.93 ms a token); kvonly2: 14.43, 14.98 → **14.70** (68.03 ms a token) | 0.02 | +2.51 tok/s | +20.5 % | negligible |
|  | nodraft | nodraft1: 15.39, 16.33 → **15.84** (63.12 ms a token); nodraft2: 8.12, 16.26 → **10.83** (92.30 ms a token) | 5.01 | +1.14 tok/s | +9.3 % | negligible |
| 49,152 | draft | draft1: 11.78, 17.26 → **14.00** (71.41 ms a token); draft2: 16.60, 17.29 → **16.94** (59.03 ms a token) | 2.94 | control | | |
|  | kvonly | kvonly1: 10.22, 17.52 → **12.91** (77.49 ms a token); kvonly2: 17.00, 17.56 → **17.27** (57.89 ms a token) | 4.37 | -0.38 tok/s | -2.5 % | negligible |
|  | nodraft | nodraft1: 14.75, 15.16 → **14.96** (66.86 ms a token); nodraft2: 13.43, 15.16 → **14.24** (70.21 ms a token) | 0.71 | -0.87 tok/s | -5.6 % | negligible |

### Generation over all depths, tok/s

Every depth's tokens over their time, per load.

| Depth (tokens) | Arm | Per load | Spread | Against `draft` | | Class |
|---:|---|---|---:|---:|---:|---|
| all | draft | draft1: **10.37**; draft2: **11.00** | 0.63 | control | | |
|  | kvonly | kvonly1: **11.26**; kvonly2: **11.78** | 0.52 | +0.84 tok/s | +7.8 % | gain |
|  | nodraft | nodraft1: **11.97**; nodraft2: **11.79** | 0.18 | +1.20 tok/s | +11.2 % | gain |

### The step on pass 1 (the first request at a new depth), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `draft`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `draft` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | draft | draft1: 243.93 (9 steps); draft2: 248.57 (9 steps) | 4.64 | control | | |
|  | kvonly | kvonly1: 247.51 (9 steps); kvonly2: 247.84 (9 steps) | 0.33 | +1.42 ms | +0.6 % | negligible |
|  | nodraft | nodraft1: 50.65 (16 steps); nodraft2: 50.42 (16 steps) | 0.23 | -195.71 ms | -79.5 % | no class (the draft differs) |
| 8,192 | draft | draft1: 256.98 (9 steps); draft2: 274.50 (9 steps) | 17.53 | control | | |
|  | kvonly | kvonly1: 254.16 (9 steps); kvonly2: 254.65 (9 steps) | 0.49 | -11.34 ms | -4.3 % | negligible |
|  | nodraft | nodraft1: 177.25 (16 steps); nodraft2: 159.04 (16 steps) | 18.20 | -97.59 ms | -36.7 % | no class (the draft differs) |
| 16,384 | draft | draft1: 208.88 (7 steps); draft2: 211.64 (7 steps) | 2.76 | control | | |
|  | kvonly | kvonly1: 139.31 (7 steps); kvonly2: 138.61 (7 steps) | 0.70 | -71.30 ms | -33.9 % | gain |
|  | nodraft | nodraft1: 184.36 (16 steps); nodraft2: 107.23 (16 steps) | 77.13 | -64.47 ms | -30.7 % | no class (the draft differs) |
| 32,768 | draft | draft1: 232.01 (8 steps); draft2: 158.11 (8 steps) | 73.90 | control | | |
|  | kvonly | kvonly1: 138.77 (8 steps); kvonly2: 138.61 (8 steps) | 0.16 | -56.37 ms | -28.9 % | negligible |
|  | nodraft | nodraft1: 64.99 (16 steps); nodraft2: 123.09 (16 steps) | 58.11 | -101.02 ms | -51.8 % | no class (the draft differs) |
| 49,152 | draft | draft1: 226.35 (6 steps); draft2: 160.60 (6 steps) | 65.75 | control | | |
|  | kvonly | kvonly1: 261.01 (6 steps); kvonly2: 156.89 (6 steps) | 104.13 | +15.47 ms | +8.0 % | negligible |
|  | nodraft | nodraft1: 67.78 (16 steps); nodraft2: 74.47 (16 steps) | 6.69 | -122.35 ms | -63.2 % | no class (the draft differs) |

### The step on pass 2 (the same request again), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `draft`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `draft` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | draft | draft1: 111.12 (9 steps); draft2: 111.24 (9 steps) | 0.13 | control | | |
|  | kvonly | kvonly1: 109.54 (9 steps); kvonly2: 109.56 (9 steps) | 0.02 | -1.63 ms | -1.5 % | gain |
|  | nodraft | nodraft1: 49.09 (16 steps); nodraft2: 49.06 (16 steps) | 0.03 | -62.11 ms | -55.9 % | no class (the draft differs) |
| 8,192 | draft | draft1: 243.90 (9 steps); draft2: 243.77 (9 steps) | 0.13 | control | | |
|  | kvonly | kvonly1: 241.01 (9 steps); kvonly2: 241.00 (9 steps) | 0.02 | -2.83 ms | -1.2 % | gain |
|  | nodraft | nodraft1: 57.34 (16 steps); nodraft2: 83.87 (16 steps) | 26.53 | -173.23 ms | -71.0 % | no class (the draft differs) |
| 16,384 | draft | draft1: 144.60 (7 steps); draft2: 132.04 (7 steps) | 12.57 | control | | |
|  | kvonly | kvonly1: 129.67 (7 steps); kvonly2: 129.50 (7 steps) | 0.18 | -8.73 ms | -6.3 % | negligible |
|  | nodraft | nodraft1: 56.62 (16 steps); nodraft2: 73.56 (16 steps) | 16.94 | -73.23 ms | -52.9 % | no class (the draft differs) |
| 32,768 | draft | draft1: 138.74 (8 steps); draft2: 135.65 (8 steps) | 3.09 | control | | |
|  | kvonly | kvonly1: 132.94 (8 steps); kvonly2: 133.52 (8 steps) | 0.58 | -3.96 ms | -2.9 % | gain |
|  | nodraft | nodraft1: 61.25 (16 steps); nodraft2: 61.51 (16 steps) | 0.26 | -75.81 ms | -55.3 % | no class (the draft differs) |
| 49,152 | draft | draft1: 154.48 (6 steps); draft2: 154.22 (6 steps) | 0.26 | control | | |
|  | kvonly | kvonly1: 152.25 (6 steps); kvonly2: 151.85 (6 steps) | 0.39 | -2.30 ms | -1.5 % | gain |
|  | nodraft | nodraft1: 65.94 (16 steps); nodraft2: 65.94 (16 steps) | 0.00 | -88.41 ms | -57.3 % | no class (the draft differs) |

### Prompt reading, tok/s

Per load: the tokens read at that depth over their time (a prime request reads a chunk on top of the cache; without primes each decode request reads its prompt), with the tokens read and the tokens reused. Requests that read under 32 tokens are left out.

| Depth (tokens) | Arm | Per load | Spread | Against `draft` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | draft | draft1: 33.6 (64 read on 0 cached); draft2: 32.8 (64 read on 0 cached) | 0.8 | control | | |
|  | kvonly | kvonly1: 33.5 (64 read on 0 cached); kvonly2: 32.9 (64 read on 0 cached) | 0.6 | -0.0 tok/s | -0.0 % | negligible |
|  | nodraft | nodraft1: 34.1 (64 read on 0 cached); nodraft2: 33.7 (64 read on 0 cached) | 0.4 | +0.7 tok/s | +2.2 % | negligible |
| 8,192 | draft | draft1: 138.1 (8128 read on 0 cached); draft2: 138.6 (8128 read on 0 cached) | 0.5 | control | | |
|  | kvonly | kvonly1: 146.4 (8128 read on 0 cached); kvonly2: 147.9 (8128 read on 0 cached) | 1.5 | +8.8 tok/s | +6.4 % | gain |
|  | nodraft | nodraft1: 180.1 (8128 read on 0 cached); nodraft2: 181.4 (8128 read on 0 cached) | 1.3 | +42.4 tok/s | +30.7 % | gain |
| 16,384 | draft | draft1: 114.8 (8708 read on 7612 cached); draft2: 115.1 (8708 read on 7612 cached) | 0.3 | control | | |
|  | kvonly | kvonly1: 122.9 (8708 read on 7612 cached); kvonly2: 123.6 (8708 read on 7612 cached) | 0.7 | +8.3 tok/s | +7.2 % | gain |
|  | nodraft | nodraft1: 167.9 (8708 read on 7612 cached); nodraft2: 168.1 (8708 read on 7612 cached) | 0.2 | +53.0 tok/s | +46.1 % | gain |
| 32,768 | draft | draft1: 93.6 (16388 read on 16316 cached); draft2: 94.4 (16388 read on 16316 cached) | 0.8 | control | | |
|  | kvonly | kvonly1: 103.2 (16388 read on 16316 cached); kvonly2: 103.7 (16388 read on 16316 cached) | 0.5 | +9.4 tok/s | +10.0 % | gain |
|  | nodraft | nodraft1: 150.8 (16388 read on 16316 cached); nodraft2: 150.7 (16388 read on 16316 cached) | 0.1 | +56.7 tok/s | +60.3 % | gain |
| 49,152 | draft | draft1: 72.2 (16900 read on 32188 cached); draft2: 72.3 (16900 read on 32188 cached) | 0.2 | control | | |
|  | kvonly | kvonly1: 80.8 (16900 read on 32188 cached); kvonly2: 80.8 (16900 read on 32188 cached) | 0.1 | +8.5 tok/s | +11.8 % | gain |
|  | nodraft | nodraft1: 121.6 (16900 read on 32188 cached); nodraft2: 120.9 (16900 read on 32188 cached) | 0.7 | +49.0 tok/s | +67.8 % | gain |

### The draft and the step

Per load, pass 1: draft tokens accepted of drafted (the share), tokens a step (generated tokens over steps, a step being one pass of the main model: generated minus accepted), and the time of a step. Pass 2 is listed where its counters differ.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | draft | draft1: 7 of 23 (0.304), 1.78 a step, 243.9 ms a step; draft2: 7 of 23 (0.304), 1.78 a step, 248.6 ms a step |
|  | kvonly | kvonly1: 7 of 23 (0.304), 1.78 a step, 247.5 ms a step; kvonly2: 7 of 23 (0.304), 1.78 a step, 247.8 ms a step |
|  | nodraft | nodraft1: no draft, 1.00 a step, 50.6 ms a step; nodraft2: no draft, 1.00 a step, 50.4 ms a step |
| 8,192 | draft | draft1: 7 of 23 (0.304), 1.78 a step, 257.0 ms a step (pass 2: 7 of 21 (0.333), 1.78 a step, 243.9 ms a step); draft2: 7 of 23 (0.304), 1.78 a step, 274.5 ms a step (pass 2: 7 of 21 (0.333), 1.78 a step, 243.8 ms a step) |
|  | kvonly | kvonly1: 7 of 23 (0.304), 1.78 a step, 254.2 ms a step (pass 2: 7 of 21 (0.333), 1.78 a step, 241.0 ms a step); kvonly2: 7 of 23 (0.304), 1.78 a step, 254.6 ms a step (pass 2: 7 of 21 (0.333), 1.78 a step, 241.0 ms a step) |
|  | nodraft | nodraft1: no draft, 1.00 a step, 177.2 ms a step; nodraft2: no draft, 1.00 a step, 159.0 ms a step |
| 16,384 | draft | draft1: 9 of 18 (0.500), 2.29 a step, 208.9 ms a step; draft2: 9 of 18 (0.500), 2.29 a step, 211.6 ms a step |
|  | kvonly | kvonly1: 9 of 18 (0.500), 2.29 a step, 139.3 ms a step; kvonly2: 9 of 18 (0.500), 2.29 a step, 138.6 ms a step |
|  | nodraft | nodraft1: no draft, 1.00 a step, 184.4 ms a step; nodraft2: no draft, 1.00 a step, 107.2 ms a step |
| 32,768 | draft | draft1: 8 of 16 (0.500), 2.00 a step, 232.0 ms a step; draft2: 8 of 16 (0.500), 2.00 a step, 158.1 ms a step |
|  | kvonly | kvonly1: 8 of 16 (0.500), 2.00 a step, 138.8 ms a step; kvonly2: 8 of 16 (0.500), 2.00 a step, 138.6 ms a step |
|  | nodraft | nodraft1: no draft, 1.00 a step, 65.0 ms a step; nodraft2: no draft, 1.00 a step, 123.1 ms a step |
| 49,152 | draft | draft1: 10 of 14 (0.714), 2.67 a step, 226.4 ms a step; draft2: 10 of 14 (0.714), 2.67 a step, 160.6 ms a step |
|  | kvonly | kvonly1: 10 of 14 (0.714), 2.67 a step, 261.0 ms a step; kvonly2: 10 of 14 (0.714), 2.67 a step, 156.9 ms a step |
|  | nodraft | nodraft1: no draft, 1.00 a step, 67.8 ms a step; nodraft2: no draft, 1.00 a step, 74.5 ms a step |

### Checks

- Times: every load started on a machine checked quiet (valid=1).
- Prompts: 64 ids at depth 0, 8192 ids at depth 8,192, 16384 ids at depth 16,384, 32768 ids at depth 32,768, 49152 ids at depth 49,152; the same ids on every load and pass.
- Tokens out: 16 on every timed request.
- Greedy reply at depth 0: the same on every pass and load of every arm.
- Greedy reply at depth 8,192: the same on every pass and load of every arm.
- Greedy reply at depth 16,384: the same on every pass and load of every arm.
- Greedy reply at depth 32,768: **differs from `draft` in nodraft.**
- Greedy reply at depth 49,152: the same on every pass and load of every arm.
- Primes (read on cached, the cache's source): 8,192: 8128 on 0 (none); 16,384: 8708 on 7612 (checkpoint); 32,768: 16388 on 16316 (checkpoint); 49,152: 16900 on 32188 (checkpoint).
