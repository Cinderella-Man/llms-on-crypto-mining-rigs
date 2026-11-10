## The deep lane: r150, r100, r050, r025, m2

Model file `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf`, llama-server version: 0.8.0-dev (build 12358, commit 1551f5fe1).

256 tokens per request at temperature 0, each request twice per load (pass 1 is the first request at a new depth, pass 2 the same request again). Timed loads, in their order: r1501, r1001, r0501, r0251, m21, r1502, r1002, r0502, r0252, m22.

| Arm | Loads | Draft | Context | Its own variables and flags |
|---|---|---|---:|---|
| r150 (control) | r1501, r1502 | on | 65536 | `GGML_CUDA_FATTN_SPARSE_PASCAL=1 GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=1.5` |
| r100 | r1001, r1002 | on | 65536 | `GGML_CUDA_FATTN_SPARSE_PASCAL=1 GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=1.0` |
| r050 | r0501, r0502 | on | 65536 | `GGML_CUDA_FATTN_SPARSE_PASCAL=1 GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=0.5` |
| r025 | r0251, r0252 | on | 65536 | `GGML_CUDA_FATTN_SPARSE_PASCAL=1 GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=0.25` |
| m2 | m21, m22 | on | 65536 | `GGML_CUDA_FATTN_SPARSE_PASCAL=2` |

### Generation, tok/s

Per load: pass 1, pass 2 → the load, with the load's time per token. A load's speed is its two passes' tokens over their time. Spread: the difference between the arm's fastest and slowest load. Class against `r150`: gain when the mean is above the control's by more than the larger of the two spreads, loss when below by more than it, else negligible.

| Depth (tokens) | Arm | Per load | Spread | Against `r150` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | r150 | r1501: 26.27, 27.62 → **26.93** (37.14 ms a token); r1502: 26.28, 27.61 → **26.93** (37.13 ms a token) | 0.00 | control | | |
|  | r100 | r1001: 25.76, 27.54 → **26.62** (37.57 ms a token); r1002: 25.72, 27.55 → **26.60** (37.59 ms a token) | 0.02 | -0.32 tok/s | -1.2 % | loss |
|  | r050 | r0501: 25.66, 27.58 → **26.59** (37.61 ms a token); r0502: 25.86, 27.58 → **26.69** (37.47 ms a token) | 0.10 | -0.29 tok/s | -1.1 % | loss |
|  | r025 | r0251: 23.51, 27.56 → **25.37** (39.41 ms a token); r0252: 25.64, 27.55 → **26.56** (37.65 ms a token) | 1.19 | -0.96 tok/s | -3.6 % | negligible |
|  | m2 | m21: 25.31, 26.53 → **25.90** (38.61 ms a token); m22: 23.21, 26.42 → **24.71** (40.47 ms a token) | 1.19 | -1.62 tok/s | -6.0 % | loss |
| 4,096 | r150 | r1501: 17.01, 23.53 → **19.75** (50.63 ms a token); r1502: 15.95, 23.35 → **18.95** (52.76 ms a token) | 0.80 | control | | |
|  | r100 | r1001: 15.34, 25.26 → **19.09** (52.38 ms a token); r1002: 15.27, 25.18 → **19.01** (52.60 ms a token) | 0.08 | -0.30 tok/s | -1.6 % | negligible |
|  | r050 | r0501: 14.73, 20.23 → **17.05** (58.66 ms a token); r0502: 14.96, 24.86 → **18.68** (53.53 ms a token) | 1.63 | -1.49 tok/s | -7.7 % | negligible |
|  | r025 | r0251: 17.93, 24.69 → **20.77** (48.14 ms a token); r0252: 17.44, 24.70 → **20.44** (48.91 ms a token) | 0.33 | +1.26 tok/s | +6.5 % | gain |
|  | m2 | m21: 16.10, 20.73 → **18.13** (55.17 ms a token); m22: 15.65, 25.08 → **19.27** (51.88 ms a token) | 1.15 | -0.65 tok/s | -3.4 % | negligible |
| 8,192 | r150 | r1501: 16.88, 20.13 → **18.36** (54.46 ms a token); r1502: 18.16, 21.29 → **19.60** (51.01 ms a token) | 1.24 | control | | |
|  | r100 | r1001: 24.37, 26.64 → **25.45** (39.29 ms a token); r1002: 20.24, 23.07 → **21.56** (46.38 ms a token) | 3.89 | +4.53 tok/s | +23.8 % | gain |
|  | r050 | r0501: 21.10, 24.38 → **22.62** (44.21 ms a token); r0502: 19.94, 25.04 → **22.20** (45.04 ms a token) | 0.42 | +3.43 tok/s | +18.1 % | gain |
|  | r025 | r0251: 21.55, 25.04 → **23.17** (43.16 ms a token); r0252: 21.30, 24.85 → **22.94** (43.59 ms a token) | 0.23 | +4.07 tok/s | +21.5 % | gain |
|  | m2 | m21: 23.73, 24.77 → **24.24** (41.25 ms a token); m22: 23.79, 24.37 → **24.08** (41.53 ms a token) | 0.16 | +5.18 tok/s | +27.3 % | gain |
| 12,288 | r150 | r1501: 14.59, 19.69 → **16.76** (59.66 ms a token); r1502: 15.45, 20.40 → **17.58** (56.87 ms a token) | 0.82 | control | | |
|  | r100 | r1001: 13.33, 19.41 → **15.81** (63.26 ms a token); r1002: 16.79, 20.86 → **18.60** (53.76 ms a token) | 2.79 | +0.03 tok/s | +0.2 % | negligible |
|  | r050 | r0501: 15.99, 21.59 → **18.37** (54.43 ms a token); r0502: 13.96, 20.65 → **16.66** (60.03 ms a token) | 1.71 | +0.34 tok/s | +2.0 % | negligible |
|  | r025 | r0251: 16.97, 20.37 → **18.52** (54.01 ms a token); r0252: 15.15, 19.15 → **16.92** (59.12 ms a token) | 1.60 | +0.54 tok/s | +3.2 % | negligible |
|  | m2 | m21: 13.94, 18.05 → **15.73** (63.56 ms a token); m22: 19.40, 21.67 → **20.48** (48.84 ms a token) | 4.74 | +0.93 tok/s | +5.4 % | negligible |

### Generation over all depths, tok/s

Every depth's tokens over their time, per load.

| Depth (tokens) | Arm | Per load | Spread | Against `r150` | | Class |
|---:|---|---|---:|---:|---:|---|
| all | r150 | r1501: **19.81**; r1502: **20.22** | 0.41 | control | | |
|  | r100 | r1001: **20.78**; r1002: **21.02** | 0.24 | +0.88 tok/s | +4.4 % | gain |
|  | r050 | r0501: **20.52**; r0502: **20.40** | 0.12 | +0.44 tok/s | +2.2 % | gain |
|  | r025 | r0251: **21.65**; r0252: **21.13** | 0.52 | +1.38 tok/s | +6.9 % | gain |
|  | m2 | m21: **20.14**; m22: **21.89** | 1.75 | +1.00 tok/s | +5.0 % | negligible |

### The step on pass 1 (the first request at a new depth), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `r150`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `r150` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | r150 | r1501: 124.95 (78 steps); r1502: 124.89 (78 steps) | 0.07 | control | | |
|  | r100 | r1001: 127.43 (78 steps); r1002: 127.62 (78 steps) | 0.19 | +2.61 ms | +2.1 % | loss |
|  | r050 | r0501: 127.92 (78 steps); r0502: 126.93 (78 steps) | 0.99 | +2.51 ms | +2.0 % | loss |
|  | r025 | r0251: 139.62 (78 steps); r0252: 127.99 (78 steps) | 11.62 | +8.89 ms | +7.1 % | negligible |
|  | m2 | m21: 128.06 (79 steps); m22: 139.63 (79 steps) | 11.57 | +8.93 ms | +7.1 % | negligible |
| 4,096 | r150 | r1501: 190.47 (79 steps); r1502: 200.63 (80 steps) | 10.16 | control | | |
|  | r100 | r1001: 225.48 (74 steps); r1002: 226.54 (74 steps) | 1.06 | +30.46 ms | +15.6 % | loss |
|  | r050 | r0501: 190.97 (91 steps); r0502: 231.25 (74 steps) | 40.29 | +15.56 ms | +8.0 % | negligible |
|  | r025 | r0251: 192.94 (74 steps); r0252: 198.37 (74 steps) | 5.43 | +0.10 ms | +0.1 % | negligible |
|  | m2 | m21: 176.62 (90 steps); m22: 215.21 (76 steps) | 38.59 | +0.36 ms | +0.2 % | negligible |
| 8,192 | r150 | r1501: 170.44 (89 steps); r1502: 171.88 (82 steps) | 1.44 | control | | |
|  | r100 | r1001: 156.80 (67 steps); r1002: 162.13 (78 steps) | 5.33 | -11.70 ms | -6.8 % | gain |
|  | r050 | r0501: 163.98 (74 steps); r0502: 175.84 (73 steps) | 11.86 | -1.25 ms | -0.7 % | negligible |
|  | r025 | r0251: 162.70 (73 steps); r0252: 164.61 (73 steps) | 1.90 | -7.51 ms | -4.4 % | gain |
|  | m2 | m21: 147.76 (73 steps); m22: 145.39 (74 steps) | 2.36 | -24.58 ms | -14.4 % | gain |
| 12,288 | r150 | r1501: 194.95 (90 steps); r1502: 194.89 (85 steps) | 0.05 | control | | |
|  | r100 | r1001: 208.68 (92 steps); r1002: 181.53 (84 steps) | 27.16 | +0.18 ms | +0.1 % | negligible |
|  | r050 | r0501: 200.13 (80 steps); r0502: 218.36 (84 steps) | 18.22 | +14.32 ms | +7.3 % | negligible |
|  | r025 | r0251: 175.39 (86 steps); r0252: 185.72 (91 steps) | 10.33 | -14.37 ms | -7.4 % | gain |
|  | m2 | m21: 191.28 (96 steps); m22: 164.91 (80 steps) | 26.37 | -16.82 ms | -8.6 % | negligible |

### The step on pass 2 (the same request again), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `r150`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `r150` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | r150 | r1501: 123.56 (75 steps); r1502: 123.62 (75 steps) | 0.05 | control | | |
|  | r100 | r1001: 123.92 (75 steps); r1002: 123.91 (75 steps) | 0.01 | +0.33 ms | +0.3 % | loss |
|  | r050 | r0501: 123.74 (75 steps); r0502: 123.78 (75 steps) | 0.04 | +0.17 ms | +0.1 % | loss |
|  | r025 | r0251: 123.84 (75 steps); r0252: 123.91 (75 steps) | 0.07 | +0.29 ms | +0.2 % | loss |
|  | m2 | m21: 123.71 (78 steps); m22: 124.24 (78 steps) | 0.53 | +0.39 ms | +0.3 % | negligible |
| 4,096 | r150 | r1501: 137.70 (79 steps); r1502: 137.06 (80 steps) | 0.64 | control | | |
|  | r100 | r1001: 136.96 (74 steps); r1002: 137.41 (74 steps) | 0.45 | -0.19 ms | -0.1 % | negligible |
|  | r050 | r0501: 140.60 (90 steps); r0502: 139.14 (74 steps) | 1.46 | +2.50 ms | +1.8 % | loss |
|  | r025 | r0251: 140.12 (74 steps); r0252: 140.06 (74 steps) | 0.07 | +2.71 ms | +2.0 % | loss |
|  | m2 | m21: 140.33 (88 steps); m22: 139.83 (73 steps) | 0.51 | +2.70 ms | +2.0 % | loss |
| 8,192 | r150 | r1501: 151.37 (84 steps); r1502: 150.30 (80 steps) | 1.07 | control | | |
|  | r100 | r1001: 143.43 (67 steps); r1002: 144.13 (77 steps) | 0.71 | -7.05 ms | -4.7 % | gain |
|  | r050 | r0501: 143.85 (73 steps); r0502: 144.01 (71 steps) | 0.16 | -6.90 ms | -4.6 % | gain |
|  | r025 | r0251: 143.97 (71 steps); r0252: 145.09 (71 steps) | 1.12 | -6.30 ms | -4.2 % | gain |
|  | m2 | m21: 143.55 (72 steps); m22: 143.93 (73 steps) | 0.37 | -7.09 ms | -4.7 % | gain |
| 12,288 | r150 | r1501: 147.75 (88 steps); r1502: 147.66 (85 steps) | 0.09 | control | | |
|  | r100 | r1001: 148.21 (89 steps); r1002: 147.89 (83 steps) | 0.32 | +0.35 ms | +0.2 % | loss |
|  | r050 | r0501: 148.25 (80 steps); r0502: 147.55 (84 steps) | 0.70 | +0.19 ms | +0.1 % | negligible |
|  | r025 | r0251: 147.85 (85 steps); r0252: 150.20 (89 steps) | 2.35 | +1.32 ms | +0.9 % | negligible |
|  | m2 | m21: 147.72 (96 steps); m22: 147.66 (80 steps) | 0.07 | -0.01 ms | -0.0 % | negligible |

### Prompt reading, tok/s

Per load: the tokens read at that depth over their time (a prime request reads a chunk on top of the cache; without primes each decode request reads its prompt), with the tokens read and the tokens reused. Requests that read under 32 tokens are left out.

| Depth (tokens) | Arm | Per load | Spread | Against `r150` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | r150 | r1501: 32.9 (64 read on 0 cached); r1502: 32.9 (64 read on 0 cached) | 0.0 | control | | |
|  | r100 | r1001: 32.8 (64 read on 0 cached); r1002: 31.7 (64 read on 0 cached) | 1.0 | -0.7 tok/s | -2.0 % | negligible |
|  | r050 | r0501: 31.9 (64 read on 0 cached); r0502: 32.6 (64 read on 0 cached) | 0.7 | -0.6 tok/s | -2.0 % | negligible |
|  | r025 | r0251: 31.2 (64 read on 0 cached); r0252: 32.2 (64 read on 0 cached) | 1.0 | -1.2 tok/s | -3.5 % | loss |
|  | m2 | m21: 32.2 (64 read on 0 cached); m22: 31.4 (64 read on 0 cached) | 0.8 | -1.1 tok/s | -3.2 % | loss |
| 4,096 | r150 | r1501: 114.8 (4032 read on 0 cached); r1502: 121.3 (4032 read on 0 cached) | 6.6 | control | | |
|  | r100 | r1001: 121.0 (4032 read on 0 cached); r1002: 119.7 (4032 read on 0 cached) | 1.2 | +2.3 tok/s | +1.9 % | negligible |
|  | r050 | r0501: 119.4 (4032 read on 0 cached); r0502: 121.2 (4032 read on 0 cached) | 1.8 | +2.2 tok/s | +1.9 % | negligible |
|  | r025 | r0251: 110.0 (4032 read on 0 cached); r0252: 124.2 (4032 read on 0 cached) | 14.3 | -1.0 tok/s | -0.8 % | negligible |
|  | m2 | m21: 121.3 (4032 read on 0 cached); m22: 121.1 (4032 read on 0 cached) | 0.2 | +3.1 tok/s | +2.7 % | negligible |
| 8,192 | r150 | r1501: 119.7 (4612 read on 3516 cached); r1502: 117.9 (4612 read on 3516 cached) | 1.9 | control | | |
|  | r100 | r1001: 119.5 (4612 read on 3516 cached); r1002: 121.2 (4612 read on 3516 cached) | 1.8 | +1.5 tok/s | +1.3 % | negligible |
|  | r050 | r0501: 119.0 (4612 read on 3516 cached); r0502: 118.5 (4612 read on 3516 cached) | 0.5 | -0.0 tok/s | -0.0 % | negligible |
|  | r025 | r0251: 121.9 (4612 read on 3516 cached); r0252: 117.8 (4612 read on 3516 cached) | 4.2 | +1.0 tok/s | +0.9 % | negligible |
|  | m2 | m21: 119.1 (4612 read on 3516 cached); m22: 122.0 (4612 read on 3516 cached) | 2.9 | +1.7 tok/s | +1.5 % | negligible |
| 12,288 | r150 | r1501: 120.9 (8708 read on 3516 cached); r1502: 122.0 (8708 read on 3516 cached) | 1.0 | control | | |
|  | r100 | r1001: 121.9 (8708 read on 3516 cached); r1002: 123.4 (8708 read on 3516 cached) | 1.5 | +1.2 tok/s | +1.0 % | negligible |
|  | r050 | r0501: 120.2 (8708 read on 3516 cached); r0502: 121.7 (8708 read on 3516 cached) | 1.6 | -0.5 tok/s | -0.4 % | negligible |
|  | r025 | r0251: 123.0 (8708 read on 3516 cached); r0252: 119.2 (8708 read on 3516 cached) | 3.8 | -0.3 tok/s | -0.3 % | negligible |
|  | m2 | m21: 120.6 (8708 read on 3516 cached); m22: 123.6 (8708 read on 3516 cached) | 3.0 | +0.7 tok/s | +0.5 % | negligible |

### The draft and the step

Per load, pass 1: draft tokens accepted of drafted (the share), tokens a step (generated tokens over steps, a step being one pass of the main model: generated minus accepted), and the time of a step. Pass 2 is listed where its counters differ.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | r150 | r1501: 178 of 229 (0.777), 3.28 a step, 125.0 ms a step (pass 2: 181 of 220 (0.823), 3.41 a step, 123.6 ms a step); r1502: 178 of 229 (0.777), 3.28 a step, 124.9 ms a step (pass 2: 181 of 220 (0.823), 3.41 a step, 123.6 ms a step) |
|  | r100 | r1001: 178 of 229 (0.777), 3.28 a step, 127.4 ms a step (pass 2: 181 of 220 (0.823), 3.41 a step, 123.9 ms a step); r1002: 178 of 229 (0.777), 3.28 a step, 127.6 ms a step (pass 2: 181 of 220 (0.823), 3.41 a step, 123.9 ms a step) |
|  | r050 | r0501: 178 of 229 (0.777), 3.28 a step, 127.9 ms a step (pass 2: 181 of 220 (0.823), 3.41 a step, 123.7 ms a step); r0502: 178 of 229 (0.777), 3.28 a step, 126.9 ms a step (pass 2: 181 of 220 (0.823), 3.41 a step, 123.8 ms a step) |
|  | r025 | r0251: 178 of 229 (0.777), 3.28 a step, 139.6 ms a step (pass 2: 181 of 220 (0.823), 3.41 a step, 123.8 ms a step); r0252: 178 of 229 (0.777), 3.28 a step, 128.0 ms a step (pass 2: 181 of 220 (0.823), 3.41 a step, 123.9 ms a step) |
|  | m2 | m21: 177 of 231 (0.766), 3.24 a step, 128.1 ms a step (pass 2: 178 of 228 (0.781), 3.28 a step, 123.7 ms a step); m22: 177 of 231 (0.766), 3.24 a step, 139.6 ms a step (pass 2: 178 of 228 (0.781), 3.28 a step, 124.2 ms a step) |
| 4,096 | r150 | r1501: 177 of 231 (0.766), 3.24 a step, 190.5 ms a step; r1502: 176 of 234 (0.752), 3.20 a step, 200.6 ms a step |
|  | r100 | r1001: 182 of 216 (0.843), 3.46 a step, 225.5 ms a step; r1002: 182 of 217 (0.839), 3.46 a step, 226.5 ms a step |
|  | r050 | r0501: 165 of 268 (0.616), 2.81 a step, 191.0 ms a step (pass 2: 166 of 265 (0.626), 2.84 a step, 140.6 ms a step); r0502: 182 of 216 (0.843), 3.46 a step, 231.3 ms a step |
|  | r025 | r0251: 182 of 219 (0.831), 3.46 a step, 192.9 ms a step; r0252: 182 of 219 (0.831), 3.46 a step, 198.4 ms a step |
|  | m2 | m21: 166 of 266 (0.624), 2.84 a step, 176.6 ms a step (pass 2: 168 of 260 (0.646), 2.91 a step, 140.3 ms a step); m22: 180 of 225 (0.800), 3.37 a step, 215.2 ms a step (pass 2: 183 of 216 (0.847), 3.51 a step, 139.8 ms a step) |
| 8,192 | r150 | r1501: 167 of 262 (0.637), 2.88 a step, 170.4 ms a step (pass 2: 172 of 247 (0.696), 3.05 a step, 151.4 ms a step); r1502: 174 of 240 (0.725), 3.12 a step, 171.9 ms a step (pass 2: 176 of 234 (0.752), 3.20 a step, 150.3 ms a step) |
|  | r100 | r1001: 189 of 196 (0.964), 3.82 a step, 156.8 ms a step; r1002: 178 of 230 (0.774), 3.28 a step, 162.1 ms a step (pass 2: 179 of 227 (0.789), 3.32 a step, 144.1 ms a step) |
|  | r050 | r0501: 182 of 218 (0.835), 3.46 a step, 164.0 ms a step (pass 2: 183 of 215 (0.851), 3.51 a step, 143.9 ms a step); r0502: 183 of 215 (0.851), 3.51 a step, 175.8 ms a step (pass 2: 185 of 209 (0.885), 3.61 a step, 144.0 ms a step) |
|  | r025 | r0251: 183 of 215 (0.851), 3.51 a step, 162.7 ms a step (pass 2: 185 of 209 (0.885), 3.61 a step, 144.0 ms a step); r0252: 183 of 215 (0.851), 3.51 a step, 164.6 ms a step (pass 2: 185 of 209 (0.885), 3.61 a step, 145.1 ms a step) |
|  | m2 | m21: 183 of 214 (0.855), 3.51 a step, 147.8 ms a step (pass 2: 184 of 211 (0.872), 3.56 a step, 143.6 ms a step); m22: 182 of 218 (0.835), 3.46 a step, 145.4 ms a step (pass 2: 183 of 215 (0.851), 3.51 a step, 143.9 ms a step) |
| 12,288 | r150 | r1501: 166 of 264 (0.629), 2.84 a step, 194.9 ms a step (pass 2: 168 of 258 (0.651), 2.91 a step, 147.7 ms a step); r1502: 171 of 250 (0.684), 3.01 a step, 194.9 ms a step |
|  | r100 | r1001: 164 of 270 (0.607), 2.78 a step, 208.7 ms a step (pass 2: 167 of 261 (0.640), 2.88 a step, 148.2 ms a step); r1002: 172 of 248 (0.694), 3.05 a step, 181.5 ms a step (pass 2: 173 of 245 (0.706), 3.08 a step, 147.9 ms a step) |
|  | r050 | r0501: 176 of 236 (0.746), 3.20 a step, 200.1 ms a step; r0502: 172 of 246 (0.699), 3.05 a step, 218.4 ms a step |
|  | r025 | r0251: 170 of 253 (0.672), 2.98 a step, 175.4 ms a step (pass 2: 171 of 250 (0.684), 3.01 a step, 147.9 ms a step); r0252: 165 of 267 (0.618), 2.81 a step, 185.7 ms a step (pass 2: 167 of 261 (0.640), 2.88 a step, 150.2 ms a step) |
|  | m2 | m21: 160 of 283 (0.565), 2.67 a step, 191.3 ms a step; m22: 176 of 236 (0.746), 3.20 a step, 164.9 ms a step |

### Checks

- Times: every load started on a machine checked quiet (valid=1).
- Prompts: 64 ids at depth 0, 4096 ids at depth 4,096, 8192 ids at depth 8,192, 12288 ids at depth 12,288; the same ids on every load and pass.
- Tokens out: 256 on every timed request.
- Greedy reply at depth 0: **differs from `r150` in m2.**
- Greedy reply at depth 4,096: 9 different replies over the 10 loads (`r150`: 2 over its 2; `r100`: 2 over its 2; `r050`: 2 over its 2; `r025`: 1 over its 2; `m2`: 2 over its 2); every load wrote its own reply again on pass 2.
- Greedy reply at depth 8,192: 7 different replies over the 10 loads (`r150`: 2 over its 2; `r100`: 2 over its 2; `r050`: 2 over its 2; `r025`: 1 over its 2; `m2`: 2 over its 2); every load wrote its own reply again on pass 2.
- Greedy reply at depth 12,288: 8 different replies over the 10 loads (`r150`: 2 over its 2; `r100`: 2 over its 2; `r050`: 2 over its 2; `r025`: 2 over its 2; `m2`: 2 over its 2); every load wrote its own reply again on pass 2.
- Primes (read on cached, the cache's source): 4,096: 4032 on 0 (none); 8,192: 4612 on 3516 (checkpoint); 12,288: 8708 on 3516 (checkpoint).
