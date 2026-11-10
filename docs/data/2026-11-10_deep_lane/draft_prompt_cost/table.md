## The deep lane: draft, dub256, nodraft, dub512

Model file `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf`, llama-server version: 0.8.0-dev (build 12358, commit 1551f5fe1).

16 tokens per request at temperature 0, each request twice per load (pass 1 is the first request at a new depth, pass 2 the same request again). Timed loads, in their order: draft1, dub256_1, nodraft1, dub512_1, draft2, dub256_2, nodraft2, dub512_2.

| Arm | Loads | Draft | Context | Its own variables and flags |
|---|---|---|---:|---|
| draft (control) | draft1, draft2 | on | 65536 | — |
| dub256 | dub256_1, dub256_2 | on | 65536 | `--spec-draft-n-ubatch 256` |
| nodraft | nodraft1, nodraft2 | off | 65536 | — |
| dub512 | dub512_1, dub512_2 | on | 65536 | `--spec-draft-n-ubatch 512` |

### Generation, tok/s

Per load: pass 1, pass 2 → the load, with the load's time per token. A load's speed is its two passes' tokens over their time. Spread: the difference between the arm's fastest and slowest load. Class against `draft`: gain when the mean is above the control's by more than the larger of the two spreads, loss when below by more than it, else negligible.

| Depth (tokens) | Arm | Per load | Spread | Against `draft` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | draft | draft1: 7.24, 15.63 → **9.89** (101.08 ms a token); draft2: 7.10, 15.74 → **9.78** (102.22 ms a token) | 0.11 | control | | |
|  | dub256 | dub256_1: 7.11, 15.74 → **9.79** (102.11 ms a token); dub256_2: 7.10, 15.73 → **9.78** (102.24 ms a token) | 0.01 | -0.05 tok/s | -0.5 % | negligible |
|  | nodraft | nodraft1: 19.63, 20.30 → **19.96** (50.10 ms a token); nodraft2: 19.66, 20.30 → **19.97** (50.06 ms a token) | 0.01 | +10.13 tok/s | +103.0 % | gain |
|  | dub512 | dub512_1: 7.09, 15.72 → **9.78** (102.30 ms a token); dub512_2: 7.10, 15.69 → **9.77** (102.34 ms a token) | 0.00 | -0.07 tok/s | -0.7 % | negligible |
| 8,192 | draft | draft1: 4.16, 13.74 → **6.39** (156.62 ms a token); draft2: 4.95, 20.89 → **8.01** (124.89 ms a token) | 1.62 | control | | |
|  | dub256 | dub256_1: 3.83, 12.09 → **5.81** (172.00 ms a token); dub256_2: 3.07, 15.15 → **5.11** (195.67 ms a token) | 0.70 | -1.73 tok/s | -24.1 % | loss |
|  | nodraft | nodraft1: 8.68, 18.51 → **11.81** (84.64 ms a token); nodraft2: 8.68, 18.51 → **11.82** (84.61 ms a token) | 0.00 | +4.62 tok/s | +64.2 % | gain |
|  | dub512 | dub512_1: 5.11, 10.11 → **6.79** (147.31 ms a token); dub512_2: 4.32, 13.39 → **6.54** (153.01 ms a token) | 0.25 | -0.53 tok/s | -7.4 % | negligible |
| 16,384 | draft | draft1: 6.18, 18.15 → **9.22** (108.43 ms a token); draft2: 5.22, 13.26 → **7.49** (133.46 ms a token) | 1.73 | control | | |
|  | dub256 | dub256_1: 4.84, 20.38 → **7.83** (127.73 ms a token); dub256_2: 13.28, 19.80 → **15.90** (62.89 ms a token) | 8.07 | +3.51 tok/s | +42.0 % | negligible |
|  | nodraft | nodraft1: 16.72, 17.65 → **17.17** (58.24 ms a token); nodraft2: 16.70, 17.63 → **17.15** (58.31 ms a token) | 0.02 | +8.80 tok/s | +105.3 % | gain |
|  | dub512 | dub512_1: 13.89, 18.93 → **16.02** (62.42 ms a token); dub512_2: 12.10, 17.83 → **14.42** (69.36 ms a token) | 1.60 | +6.86 tok/s | +82.1 % | gain |
| 32,768 | draft | draft1: 15.11, 15.68 → **15.39** (64.99 ms a token); draft2: 11.87, 13.75 → **12.74** (78.51 ms a token) | 2.65 | control | | |
|  | dub256 | dub256_1: 5.78, 17.31 → **8.67** (115.40 ms a token); dub256_2: 5.79, 15.63 → **8.46** (118.27 ms a token) | 0.21 | -5.50 tok/s | -39.1 % | loss |
|  | nodraft | nodraft1: 15.43, 16.28 → **15.84** (63.12 ms a token); nodraft2: 15.51, 16.27 → **15.88** (62.96 ms a token) | 0.04 | +1.80 tok/s | +12.8 % | negligible |
|  | dub512 | dub512_1: 6.55, 14.59 → **9.04** (110.56 ms a token); dub512_2: 14.17, 14.71 → **14.43** (69.28 ms a token) | 5.39 | -2.32 tok/s | -16.5 % | negligible |
| 49,152 | draft | draft1: 14.21, 14.74 → **14.47** (69.12 ms a token); draft2: 11.76, 13.20 → **12.44** (80.39 ms a token) | 2.03 | control | | |
|  | dub256 | dub256_1: 16.41, 17.01 → **16.70** (59.87 ms a token); dub256_2: 16.01, 16.99 → **16.49** (60.65 ms a token) | 0.21 | +3.14 tok/s | +23.3 % | gain |
|  | nodraft | nodraft1: 14.43, 15.12 → **14.77** (67.70 ms a token); nodraft2: 14.46, 15.12 → **14.79** (67.63 ms a token) | 0.02 | +1.32 tok/s | +9.8 % | negligible |
|  | dub512 | dub512_1: 16.47, 17.20 → **16.83** (59.42 ms a token); dub512_2: 16.64, 17.23 → **16.93** (59.07 ms a token) | 0.10 | +3.43 tok/s | +25.5 % | gain |

### Generation over all depths, tok/s

Every depth's tokens over their time, per load.

| Depth (tokens) | Arm | Per load | Spread | Against `draft` | | Class |
|---:|---|---|---:|---:|---:|---|
| all | draft | draft1: **10.00**; draft2: **9.63** | 0.37 | control | | |
|  | dub256 | dub256_1: **8.66**; dub256_2: **9.26** | 0.60 | -0.85 tok/s | -8.6 % | loss |
|  | nodraft | nodraft1: **15.44**; nodraft2: **15.45** | 0.01 | +5.64 tok/s | +57.5 % | gain |
|  | dub512 | dub512_1: **10.37**; dub512_2: **11.04** | 0.66 | +0.89 tok/s | +9.1 % | gain |

### The step on pass 1 (the first request at a new depth), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `draft`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `draft` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | draft | draft1: 245.67 (9 steps); draft2: 250.46 (9 steps) | 4.79 | control | | |
|  | dub256 | dub256_1: 250.10 (9 steps); dub256_2: 250.52 (9 steps) | 0.42 | +2.24 ms | +0.9 % | negligible |
|  | nodraft | nodraft1: 50.94 (16 steps); nodraft2: 50.87 (16 steps) | 0.07 | -197.16 ms | -79.5 % | no class (the draft differs) |
|  | dub512 | dub512_1: 250.61 (9 steps); dub512_2: 250.56 (9 steps) | 0.06 | +2.52 ms | +1.0 % | negligible |
| 8,192 | draft | draft1: 427.44 (9 steps); draft2: 538.44 (6 steps) | 111.01 | control | | |
|  | dub256 | dub256_1: 418.12 (10 steps); dub256_2: 578.34 (9 steps) | 160.22 | +15.29 ms | +3.2 % | negligible |
|  | nodraft | nodraft1: 115.26 (16 steps); nodraft2: 115.19 (16 steps) | 0.07 | -367.71 ms | -76.1 % | no class (the draft differs) |
|  | dub512 | dub512_1: 313.08 (10 steps); dub512_2: 462.71 (8 steps) | 149.63 | -95.04 ms | -19.7 % | negligible |
| 16,384 | draft | draft1: 369.77 (7 steps); draft2: 437.72 (7 steps) | 67.95 | control | | |
|  | dub256 | dub256_1: 550.41 (6 steps); dub256_2: 200.73 (6 steps) | 349.68 | -28.18 ms | -7.0 % | negligible |
|  | nodraft | nodraft1: 59.81 (16 steps); nodraft2: 59.89 (16 steps) | 0.07 | -343.90 ms | -85.2 % | no class (the draft differs) |
|  | dub512 | dub512_1: 192.02 (6 steps); dub512_2: 220.39 (6 steps) | 28.37 | -197.54 ms | -48.9 % | gain |
| 32,768 | draft | draft1: 151.29 (7 steps); draft2: 192.62 (7 steps) | 41.34 | control | | |
|  | dub256 | dub256_1: 395.48 (7 steps); dub256_2: 394.48 (7 steps) | 1.00 | +223.02 ms | +129.7 % | loss |
|  | nodraft | nodraft1: 64.79 (16 steps); nodraft2: 64.47 (16 steps) | 0.32 | -107.32 ms | -62.4 % | no class (the draft differs) |
|  | dub512 | dub512_1: 305.18 (8 steps); dub512_2: 141.11 (8 steps) | 164.07 | +51.19 ms | +29.8 % | negligible |
| 49,152 | draft | draft1: 160.89 (7 steps); draft2: 170.01 (8 steps) | 9.12 | control | | |
|  | dub256 | dub256_1: 162.53 (6 steps); dub256_2: 166.52 (6 steps) | 4.00 | -0.92 ms | -0.6 % | negligible |
|  | nodraft | nodraft1: 69.28 (16 steps); nodraft2: 69.14 (16 steps) | 0.14 | -96.23 ms | -58.2 % | no class (the draft differs) |
|  | dub512 | dub512_1: 161.87 (6 steps); dub512_2: 160.25 (6 steps) | 1.62 | -4.38 ms | -2.6 % | negligible |

### The step on pass 2 (the same request again), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `draft`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `draft` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | draft | draft1: 113.71 (9 steps); draft2: 112.97 (9 steps) | 0.74 | control | | |
|  | dub256 | dub256_1: 112.97 (9 steps); dub256_2: 113.00 (9 steps) | 0.03 | -0.36 ms | -0.3 % | negligible |
|  | nodraft | nodraft1: 49.26 (16 steps); nodraft2: 49.26 (16 steps) | 0.00 | -64.08 ms | -56.5 % | no class (the draft differs) |
|  | dub512 | dub512_1: 113.11 (9 steps); dub512_2: 113.32 (9 steps) | 0.21 | -0.12 ms | -0.1 % | negligible |
| 8,192 | draft | draft1: 145.60 (8 steps); draft2: 127.65 (6 steps) | 17.95 | control | | |
|  | dub256 | dub256_1: 146.99 (9 steps); dub256_2: 132.04 (8 steps) | 14.95 | +2.88 ms | +2.1 % | negligible |
|  | nodraft | nodraft1: 54.03 (16 steps); nodraft2: 54.03 (16 steps) | 0.00 | -82.60 ms | -60.5 % | no class (the draft differs) |
|  | dub512 | dub512_1: 175.90 (9 steps); dub512_2: 149.35 (8 steps) | 26.56 | +26.00 ms | +19.0 % | negligible |
| 16,384 | draft | draft1: 125.92 (7 steps); draft2: 172.36 (7 steps) | 46.44 | control | | |
|  | dub256 | dub256_1: 130.82 (6 steps); dub256_2: 134.70 (6 steps) | 3.88 | -16.38 ms | -11.0 % | negligible |
|  | nodraft | nodraft1: 56.66 (16 steps); nodraft2: 56.73 (16 steps) | 0.07 | -92.45 ms | -62.0 % | no class (the draft differs) |
|  | dub512 | dub512_1: 140.86 (6 steps); dub512_2: 149.55 (6 steps) | 8.69 | -3.94 ms | -2.6 % | negligible |
| 32,768 | draft | draft1: 145.79 (7 steps); draft2: 166.28 (7 steps) | 20.48 | control | | |
|  | dub256 | dub256_1: 132.07 (7 steps); dub256_2: 146.19 (7 steps) | 14.12 | -16.90 ms | -10.8 % | negligible |
|  | nodraft | nodraft1: 61.44 (16 steps); nodraft2: 61.45 (16 steps) | 0.01 | -94.59 ms | -60.6 % | no class (the draft differs) |
|  | dub512 | dub512_1: 137.06 (8 steps); dub512_2: 136.00 (8 steps) | 1.06 | -19.50 ms | -12.5 % | negligible |
| 49,152 | draft | draft1: 155.08 (7 steps); draft2: 151.54 (8 steps) | 3.54 | control | | |
|  | dub256 | dub256_1: 156.81 (6 steps); dub256_2: 156.93 (6 steps) | 0.13 | +3.56 ms | +2.3 % | loss |
|  | nodraft | nodraft1: 66.12 (16 steps); nodraft2: 66.12 (16 steps) | 0.00 | -87.19 ms | -56.9 % | no class (the draft differs) |
|  | dub512 | dub512_1: 155.02 (6 steps); dub512_2: 154.76 (6 steps) | 0.26 | +1.58 ms | +1.0 % | negligible |

### Prompt reading, tok/s

Per load: the tokens read at that depth over their time (a prime request reads a chunk on top of the cache; without primes each decode request reads its prompt), with the tokens read and the tokens reused. Requests that read under 32 tokens are left out.

| Depth (tokens) | Arm | Per load | Spread | Against `draft` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | draft | draft1: 32.8 (64 read on 0 cached); draft2: 32.8 (64 read on 0 cached) | 0.1 | control | | |
|  | dub256 | dub256_1: 32.8 (64 read on 0 cached); dub256_2: 31.8 (64 read on 0 cached) | 1.0 | -0.5 tok/s | -1.6 % | negligible |
|  | nodraft | nodraft1: 33.5 (64 read on 0 cached); nodraft2: 33.8 (64 read on 0 cached) | 0.3 | +0.9 tok/s | +2.7 % | gain |
|  | dub512 | dub512_1: 32.9 (64 read on 0 cached); dub512_2: 32.6 (64 read on 0 cached) | 0.3 | -0.0 tok/s | -0.1 % | negligible |
| 8,192 | draft | draft1: 122.5 (8128 read on 0 cached); draft2: 129.2 (8128 read on 0 cached) | 6.8 | control | | |
|  | dub256 | dub256_1: 130.2 (8128 read on 0 cached); dub256_2: 129.0 (8128 read on 0 cached) | 1.2 | +3.7 tok/s | +3.0 % | negligible |
|  | nodraft | nodraft1: 177.1 (8128 read on 0 cached); nodraft2: 177.5 (8128 read on 0 cached) | 0.4 | +51.4 tok/s | +40.9 % | gain |
|  | dub512 | dub512_1: 132.8 (8128 read on 0 cached); dub512_2: 131.9 (8128 read on 0 cached) | 0.9 | +6.5 tok/s | +5.2 % | negligible |
| 16,384 | draft | draft1: 107.8 (8708 read on 7612 cached); draft2: 108.9 (8708 read on 7612 cached) | 1.2 | control | | |
|  | dub256 | dub256_1: 110.7 (8708 read on 7612 cached); dub256_2: 110.8 (8708 read on 7612 cached) | 0.1 | +2.4 tok/s | +2.2 % | gain |
|  | nodraft | nodraft1: 167.0 (8708 read on 7612 cached); nodraft2: 166.2 (8708 read on 7612 cached) | 0.8 | +58.3 tok/s | +53.8 % | gain |
|  | dub512 | dub512_1: 113.2 (8708 read on 7612 cached); dub512_2: 112.7 (8708 read on 7612 cached) | 0.5 | +4.6 tok/s | +4.2 % | gain |
| 32,768 | draft | draft1: 90.6 (16388 read on 16316 cached); draft2: 90.4 (16388 read on 16316 cached) | 0.2 | control | | |
|  | dub256 | dub256_1: 92.4 (16388 read on 16316 cached); dub256_2: 92.1 (16388 read on 16316 cached) | 0.3 | +1.7 tok/s | +1.9 % | gain |
|  | nodraft | nodraft1: 148.5 (16388 read on 16316 cached); nodraft2: 148.7 (16388 read on 16316 cached) | 0.1 | +58.1 tok/s | +64.2 % | gain |
|  | dub512 | dub512_1: 93.8 (16388 read on 16316 cached); dub512_2: 93.6 (16388 read on 16316 cached) | 0.2 | +3.2 tok/s | +3.6 % | gain |
| 49,152 | draft | draft1: 70.0 (16900 read on 32188 cached); draft2: 70.1 (16900 read on 32188 cached) | 0.1 | control | | |
|  | dub256 | dub256_1: 71.1 (16900 read on 32188 cached); dub256_2: 70.9 (16900 read on 32188 cached) | 0.2 | +0.9 tok/s | +1.4 % | gain |
|  | nodraft | nodraft1: 119.5 (16900 read on 32188 cached); nodraft2: 119.6 (16900 read on 32188 cached) | 0.1 | +49.5 tok/s | +70.7 % | gain |
|  | dub512 | dub512_1: 72.4 (16900 read on 32188 cached); dub512_2: 72.2 (16900 read on 32188 cached) | 0.2 | +2.3 tok/s | +3.2 % | gain |

### The draft and the step

Per load, pass 1: draft tokens accepted of drafted (the share), tokens a step (generated tokens over steps, a step being one pass of the main model: generated minus accepted), and the time of a step. Pass 2 is listed where its counters differ.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | draft | draft1: 7 of 23 (0.304), 1.78 a step, 245.7 ms a step; draft2: 7 of 23 (0.304), 1.78 a step, 250.5 ms a step |
|  | dub256 | dub256_1: 7 of 23 (0.304), 1.78 a step, 250.1 ms a step; dub256_2: 7 of 23 (0.304), 1.78 a step, 250.5 ms a step |
|  | nodraft | nodraft1: no draft, 1.00 a step, 50.9 ms a step; nodraft2: no draft, 1.00 a step, 50.9 ms a step |
|  | dub512 | dub512_1: 7 of 23 (0.304), 1.78 a step, 250.6 ms a step; dub512_2: 7 of 23 (0.304), 1.78 a step, 250.6 ms a step |
| 8,192 | draft | draft1: 7 of 22 (0.318), 1.78 a step, 427.4 ms a step (pass 2: 8 of 19 (0.421), 2.00 a step, 145.6 ms a step); draft2: 10 of 13 (0.769), 2.67 a step, 538.4 ms a step |
|  | dub256 | dub256_1: 6 of 25 (0.240), 1.60 a step, 418.1 ms a step (pass 2: 7 of 22 (0.318), 1.78 a step, 147.0 ms a step); dub256_2: 7 of 22 (0.318), 1.78 a step, 578.3 ms a step (pass 2: 8 of 19 (0.421), 2.00 a step, 132.0 ms a step) |
|  | nodraft | nodraft1: no draft, 1.00 a step, 115.3 ms a step; nodraft2: no draft, 1.00 a step, 115.2 ms a step |
|  | dub512 | dub512_1: 6 of 26 (0.231), 1.60 a step, 313.1 ms a step (pass 2: 7 of 23 (0.304), 1.78 a step, 175.9 ms a step); dub512_2: 8 of 18 (0.444), 2.00 a step, 462.7 ms a step |
| 16,384 | draft | draft1: 9 of 15 (0.600), 2.29 a step, 369.8 ms a step; draft2: 9 of 15 (0.600), 2.29 a step, 437.7 ms a step |
|  | dub256 | dub256_1: 10 of 14 (0.714), 2.67 a step, 550.4 ms a step; dub256_2: 10 of 14 (0.714), 2.67 a step, 200.7 ms a step |
|  | nodraft | nodraft1: no draft, 1.00 a step, 59.8 ms a step; nodraft2: no draft, 1.00 a step, 59.9 ms a step |
|  | dub512 | dub512_1: 10 of 14 (0.714), 2.67 a step, 192.0 ms a step; dub512_2: 10 of 14 (0.714), 2.67 a step, 220.4 ms a step |
| 32,768 | draft | draft1: 9 of 17 (0.529), 2.29 a step, 151.3 ms a step; draft2: 9 of 15 (0.600), 2.29 a step, 192.6 ms a step (pass 2: 9 of 13 (0.692), 2.29 a step, 166.3 ms a step) |
|  | dub256 | dub256_1: 9 of 13 (0.692), 2.29 a step, 395.5 ms a step; dub256_2: 9 of 13 (0.692), 2.29 a step, 394.5 ms a step |
|  | nodraft | nodraft1: no draft, 1.00 a step, 64.8 ms a step; nodraft2: no draft, 1.00 a step, 64.5 ms a step |
|  | dub512 | dub512_1: 8 of 16 (0.500), 2.00 a step, 305.2 ms a step; dub512_2: 8 of 16 (0.500), 2.00 a step, 141.1 ms a step |
| 49,152 | draft | draft1: 9 of 16 (0.562), 2.29 a step, 160.9 ms a step; draft2: 8 of 15 (0.533), 2.00 a step, 170.0 ms a step |
|  | dub256 | dub256_1: 10 of 14 (0.714), 2.67 a step, 162.5 ms a step; dub256_2: 10 of 14 (0.714), 2.67 a step, 166.5 ms a step |
|  | nodraft | nodraft1: no draft, 1.00 a step, 69.3 ms a step; nodraft2: no draft, 1.00 a step, 69.1 ms a step |
|  | dub512 | dub512_1: 10 of 14 (0.714), 2.67 a step, 161.9 ms a step; dub512_2: 10 of 14 (0.714), 2.67 a step, 160.3 ms a step |

### Checks

- Times: every load started on a machine checked quiet (valid=1).
- Prompts: 64 ids at depth 0, 8192 ids at depth 8,192, 16384 ids at depth 16,384, 32768 ids at depth 32,768, 49152 ids at depth 49,152; the same ids on every load and pass.
- Tokens out: 16 on every timed request.
- Greedy reply at depth 0: the same on every pass and load of every arm.
- Greedy reply at depth 8,192: 6 different replies over the 8 loads (`draft`: 2 over its 2; `dub256`: 2 over its 2; `nodraft`: 1 over its 2; `dub512`: 2 over its 2); every load wrote its own reply again on pass 2.
- Greedy reply at depth 16,384: **differs from `draft` in dub256, nodraft, dub512.**
- Greedy reply at depth 32,768: 5 different replies over the 8 loads (`draft`: 2 over its 2; `dub256`: 2 over its 2; `nodraft`: 2 over its 2; `dub512`: 1 over its 2); every load wrote its own reply again on pass 2.
- Greedy reply at depth 49,152: 3 different replies over the 8 loads (`draft`: 2 over its 2; `dub256`: 1 over its 2; `nodraft`: 1 over its 2; `dub512`: 1 over its 2); every load wrote its own reply again on pass 2.
- Primes (read on cached, the cache's source): 8,192: 8128 on 0 (none); 16,384: 8708 on 7612 (checkpoint); 32,768: 16388 on 16316 (checkpoint); 49,152: 16900 on 32188 (checkpoint).
