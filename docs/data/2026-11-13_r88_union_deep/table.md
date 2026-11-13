## The deep lane: ub512, r100, u150

Model file `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf`, llama-server version: 1.0.0-dev (build 12832, commit eb312b7bce).

256 tokens per request at temperature 0, each request twice per load (pass 1 is the first request at a new depth, pass 2 the same request again). Timed loads, in their order: ub512_1, r100_1, u150_1, ub512_2, r100_2, u150_2.

| Arm | Loads | Draft | Context | Its own variables and flags |
|---|---|---|---:|---|
| ub512 (control) | ub512_1, ub512_2 | on | 65536 | `GGML_CUDA_FATTN_SPARSE_PASCAL=1` |
| r100 | r100_1, r100_2 | on | 65536 | `GGML_CUDA_FATTN_SPARSE_PASCAL=1 GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=1.0` |
| u150 | u150_1, u150_2 | on | 65536 | `GGML_CUDA_FATTN_SPARSE_PASCAL=1 GGML_CUDA_FATTN_SPARSE_PASCAL_UNION=1` |

### Generation, tok/s

Per load: pass 1, pass 2 → the load, with the load's time per token. A load's speed is its pass 2. Spread: the difference between the arm's fastest and slowest load. Class against `ub512`: gain when the mean is above the control's by more than the larger of the two spreads, loss when below by more than it, else negligible.

| Depth (tokens) | Arm | Per load | Spread | Against `ub512` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | ub512 | ub512_1: 26.78, 28.16 → **28.16** (35.51 ms a token); ub512_2: 26.13, 28.17 → **28.17** (35.50 ms a token) | 0.01 | control | | |
|  | r100 | r100_1: 26.06, 28.08 → **28.08** (35.61 ms a token); r100_2: 26.15, 28.17 → **28.17** (35.50 ms a token) | 0.09 | -0.04 tok/s | -0.1 % | negligible |
|  | u150 | u150_1: 26.17, 28.13 → **28.13** (35.55 ms a token); u150_2: 26.10, 28.14 → **28.14** (35.54 ms a token) | 0.01 | -0.03 tok/s | -0.1 % | loss |
| 4,096 | ub512 | ub512_1: 22.62, 25.58 → **25.58** (39.09 ms a token); ub512_2: 22.68, 25.59 → **25.59** (39.08 ms a token) | 0.01 | control | | |
|  | r100 | r100_1: 20.89, 25.57 → **25.57** (39.11 ms a token); r100_2: 20.90, 25.62 → **25.62** (39.03 ms a token) | 0.05 | +0.01 tok/s | +0.0 % | negligible |
|  | u150 | u150_1: 17.75, 23.53 → **23.53** (42.50 ms a token); u150_2: 21.20, 23.88 → **23.88** (41.87 ms a token) | 0.36 | -1.88 tok/s | -7.4 % | loss |
| 8,192 | ub512 | ub512_1: 23.79, 24.54 → **24.54** (40.75 ms a token); ub512_2: 23.55, 24.53 → **24.53** (40.77 ms a token) | 0.01 | control | | |
|  | r100 | r100_1: 25.38, 26.17 → **26.17** (38.22 ms a token); r100_2: 25.21, 26.23 → **26.23** (38.12 ms a token) | 0.07 | +1.67 tok/s | +6.8 % | gain |
|  | u150 | u150_1: 20.62, 23.85 → **23.85** (41.93 ms a token); u150_2: 22.29, 23.63 → **23.63** (42.32 ms a token) | 0.22 | -0.79 tok/s | -3.2 % | loss |
| 16,384 | ub512 | ub512_1: 18.93, 23.96 → **23.96** (41.73 ms a token); ub512_2: 21.27, 23.96 → **23.96** (41.73 ms a token) | 0.00 | control | | |
|  | r100 | r100_1: 21.23, 23.90 → **23.90** (41.84 ms a token); r100_2: 19.79, 23.96 → **23.96** (41.73 ms a token) | 0.06 | -0.03 tok/s | -0.1 % | negligible |
|  | u150 | u150_1: 20.83, 23.86 → **23.86** (41.91 ms a token); u150_2: 23.63, 23.88 → **23.88** (41.88 ms a token) | 0.02 | -0.10 tok/s | -0.4 % | loss |
| 32,768 | ub512 | ub512_1: 15.46, 17.07 → **17.07** (58.57 ms a token); ub512_2: 16.25, 17.09 → **17.09** (58.52 ms a token) | 0.01 | control | | |
|  | r100 | r100_1: 16.21, 16.85 → **16.85** (59.36 ms a token); r100_2: 16.34, 17.10 → **17.10** (58.47 ms a token) | 0.26 | -0.11 tok/s | -0.6 % | negligible |
|  | u150 | u150_1: 15.59, 16.90 → **16.90** (59.18 ms a token); u150_2: 16.20, 16.94 → **16.94** (59.03 ms a token) | 0.04 | -0.16 tok/s | -1.0 % | loss |
| 49,152 | ub512 | ub512_1: 16.48, 19.13 → **19.13** (52.27 ms a token); ub512_2: 17.12, 19.16 → **19.16** (52.19 ms a token) | 0.03 | control | | |
|  | r100 | r100_1: 17.11, 19.10 → **19.10** (52.35 ms a token); r100_2: 16.23, 19.15 → **19.15** (52.23 ms a token) | 0.05 | -0.02 tok/s | -0.1 % | negligible |
|  | u150 | u150_1: 17.34, 19.21 → **19.21** (52.05 ms a token); u150_2: 17.35, 19.27 → **19.27** (51.89 ms a token) | 0.06 | +0.10 tok/s | +0.5 % | gain |

### Generation over all depths, tok/s

Every depth's tokens over their time, per load.

| Depth (tokens) | Arm | Per load | Spread | Against `ub512` | | Class |
|---:|---|---|---:|---:|---:|---|
| all | ub512 | ub512_1: **22.39**; ub512_2: **22.41** | 0.01 | control | | |
|  | r100 | r100_1: **22.52**; r100_2: **22.64** | 0.12 | +0.18 tok/s | +0.8 % | gain |
|  | u150 | u150_1: **21.97**; u150_2: **22.02** | 0.05 | -0.41 tok/s | -1.8 % | loss |

### The step on pass 1 (the first request at a new depth), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `ub512`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `ub512` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | ub512 | ub512_1: 122.56 (78 steps); ub512_2: 125.62 (78 steps) | 3.06 | control | | |
|  | r100 | r100_1: 125.94 (78 steps); r100_2: 125.50 (78 steps) | 0.44 | +1.63 ms | +1.3 % | negligible |
|  | u150 | u150_1: 125.40 (78 steps); u150_2: 125.74 (78 steps) | 0.34 | +1.48 ms | +1.2 % | negligible |
| 4,096 | ub512 | ub512_1: 152.95 (74 steps); ub512_2: 152.54 (74 steps) | 0.42 | control | | |
|  | r100 | r100_1: 165.57 (74 steps); r100_2: 165.52 (74 steps) | 0.05 | +12.80 ms | +8.4 % | loss |
|  | u150 | u150_1: 184.88 (78 steps); u150_2: 154.82 (78 steps) | 30.06 | +17.10 ms | +11.2 % | negligible |
| 8,192 | ub512 | ub512_1: 151.59 (71 steps); ub512_2: 153.13 (71 steps) | 1.55 | control | | |
|  | r100 | r100_1: 144.08 (70 steps); r100_2: 145.06 (70 steps) | 0.98 | -7.79 ms | -5.1 % | gain |
|  | u150 | u150_1: 174.84 (71 steps); u150_2: 161.73 (71 steps) | 13.11 | +15.92 ms | +10.5 % | loss |
| 16,384 | ub512 | ub512_1: 187.85 (72 steps); ub512_2: 167.18 (72 steps) | 20.67 | control | | |
|  | r100 | r100_1: 167.51 (72 steps); r100_2: 179.69 (72 steps) | 12.18 | -3.91 ms | -2.2 % | negligible |
|  | u150 | u150_1: 173.06 (71 steps); u150_2: 152.57 (71 steps) | 20.49 | -14.70 ms | -8.3 % | negligible |
| 32,768 | ub512 | ub512_1: 179.98 (92 steps); ub512_2: 171.19 (92 steps) | 8.78 | control | | |
|  | r100 | r100_1: 171.66 (92 steps); r100_2: 170.29 (92 steps) | 1.37 | -4.61 ms | -2.6 % | negligible |
|  | u150 | u150_1: 178.54 (92 steps); u150_2: 171.79 (92 steps) | 6.75 | -0.42 ms | -0.2 % | negligible |
| 49,152 | ub512 | ub512_1: 209.92 (74 steps); ub512_2: 202.06 (74 steps) | 7.86 | control | | |
|  | r100 | r100_1: 202.14 (74 steps); r100_2: 213.18 (74 steps) | 11.04 | +1.67 ms | +0.8 % | negligible |
|  | u150 | u150_1: 202.28 (73 steps); u150_2: 202.14 (73 steps) | 0.14 | -3.78 ms | -1.8 % | negligible |

### The step on pass 2 (the same request again), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `ub512`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `ub512` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | ub512 | ub512_1: 121.22 (75 steps); ub512_2: 121.19 (75 steps) | 0.03 | control | | |
|  | r100 | r100_1: 121.54 (75 steps); r100_2: 121.17 (75 steps) | 0.37 | +0.15 ms | +0.1 % | negligible |
|  | u150 | u150_1: 121.34 (75 steps); u150_2: 121.29 (75 steps) | 0.05 | +0.11 ms | +0.1 % | loss |
| 4,096 | ub512 | ub512_1: 135.21 (74 steps); ub512_2: 135.18 (74 steps) | 0.04 | control | | |
|  | r100 | r100_1: 135.29 (74 steps); r100_2: 135.01 (74 steps) | 0.28 | -0.04 ms | -0.0 % | negligible |
|  | u150 | u150_1: 139.50 (78 steps); u150_2: 137.42 (78 steps) | 2.08 | +3.26 ms | +2.4 % | loss |
| 8,192 | ub512 | ub512_1: 149.04 (70 steps); ub512_2: 149.10 (70 steps) | 0.06 | control | | |
|  | r100 | r100_1: 141.79 (69 steps); r100_2: 141.43 (69 steps) | 0.36 | -7.46 ms | -5.0 % | gain |
|  | u150 | u150_1: 157.86 (68 steps); u150_2: 159.30 (68 steps) | 1.44 | +9.52 ms | +6.4 % | loss |
| 16,384 | ub512 | ub512_1: 148.38 (72 steps); ub512_2: 148.37 (72 steps) | 0.00 | control | | |
|  | r100 | r100_1: 148.75 (72 steps); r100_2: 148.37 (72 steps) | 0.38 | +0.19 ms | +0.1 % | negligible |
|  | u150 | u150_1: 151.12 (71 steps); u150_2: 151.02 (71 steps) | 0.11 | +2.69 ms | +1.8 % | loss |
| 32,768 | ub512 | ub512_1: 164.76 (91 steps); ub512_2: 164.62 (91 steps) | 0.14 | control | | |
|  | r100 | r100_1: 166.99 (91 steps); r100_2: 164.49 (91 steps) | 2.50 | +1.05 ms | +0.6 % | negligible |
|  | u150 | u150_1: 166.49 (91 steps); u150_2: 166.06 (91 steps) | 0.43 | +1.58 ms | +1.0 % | loss |
| 49,152 | ub512 | ub512_1: 180.83 (74 steps); ub512_2: 180.56 (74 steps) | 0.28 | control | | |
|  | r100 | r100_1: 181.11 (74 steps); r100_2: 180.67 (74 steps) | 0.44 | +0.20 ms | +0.1 % | negligible |
|  | u150 | u150_1: 182.51 (73 steps); u150_2: 181.96 (73 steps) | 0.56 | +1.54 ms | +0.9 % | loss |

### Prompt reading, tok/s

Per load: the tokens read at that depth over their time (a prime request reads a chunk on top of the cache; without primes each decode request reads its prompt), with the tokens read and the tokens reused. Requests that read under 32 tokens are left out.

| Depth (tokens) | Arm | Per load | Spread | Against `ub512` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | ub512 | ub512_1: 33.6 (64 read on 0 cached); ub512_2: 33.0 (64 read on 0 cached) | 0.6 | control | | |
|  | r100 | r100_1: 32.5 (64 read on 0 cached); r100_2: 32.7 (64 read on 0 cached) | 0.3 | -0.7 tok/s | -2.2 % | loss |
|  | u150 | u150_1: 32.8 (64 read on 0 cached); u150_2: 33.3 (64 read on 0 cached) | 0.5 | -0.2 tok/s | -0.7 % | negligible |
| 4,096 | ub512 | ub512_1: 130.8 (4032 read on 0 cached); ub512_2: 132.1 (4032 read on 0 cached) | 1.3 | control | | |
|  | r100 | r100_1: 131.2 (4032 read on 0 cached); r100_2: 131.4 (4032 read on 0 cached) | 0.2 | -0.1 tok/s | -0.1 % | negligible |
|  | u150 | u150_1: 126.1 (4032 read on 0 cached); u150_2: 132.0 (4032 read on 0 cached) | 5.9 | -2.5 tok/s | -1.9 % | negligible |
| 8,192 | ub512 | ub512_1: 126.4 (4612 read on 3516 cached); ub512_2: 126.2 (4612 read on 3516 cached) | 0.2 | control | | |
|  | r100 | r100_1: 126.6 (4612 read on 3516 cached); r100_2: 126.1 (4612 read on 3516 cached) | 0.5 | +0.0 tok/s | +0.0 % | negligible |
|  | u150 | u150_1: 127.9 (4612 read on 3516 cached); u150_2: 128.5 (4612 read on 3516 cached) | 0.6 | +1.9 tok/s | +1.5 % | gain |
| 16,384 | ub512 | ub512_1: 125.3 (12804 read on 3516 cached); ub512_2: 124.1 (12804 read on 3516 cached) | 1.1 | control | | |
|  | r100 | r100_1: 125.6 (12804 read on 3516 cached); r100_2: 125.5 (12804 read on 3516 cached) | 0.1 | +0.9 tok/s | +0.7 % | negligible |
|  | u150 | u150_1: 125.9 (12804 read on 3516 cached); u150_2: 126.3 (12804 read on 3516 cached) | 0.4 | +1.4 tok/s | +1.1 % | gain |
| 32,768 | ub512 | ub512_1: 94.4 (16900 read on 15804 cached); ub512_2: 93.4 (16900 read on 15804 cached) | 0.9 | control | | |
|  | r100 | r100_1: 93.5 (16900 read on 15804 cached); r100_2: 93.7 (16900 read on 15804 cached) | 0.2 | -0.3 tok/s | -0.3 % | negligible |
|  | u150 | u150_1: 94.3 (16900 read on 15804 cached); u150_2: 94.3 (16900 read on 15804 cached) | 0.0 | +0.4 tok/s | +0.4 % | negligible |
| 49,152 | ub512 | ub512_1: 72.6 (16900 read on 32188 cached); ub512_2: 72.1 (16900 read on 32188 cached) | 0.4 | control | | |
|  | r100 | r100_1: 71.9 (16900 read on 32188 cached); r100_2: 72.7 (16900 read on 32188 cached) | 0.7 | -0.1 tok/s | -0.1 % | negligible |
|  | u150 | u150_1: 72.6 (16900 read on 32188 cached); u150_2: 72.8 (16900 read on 32188 cached) | 0.1 | +0.3 tok/s | +0.5 % | negligible |

### The draft and the step

Per load, pass 1: draft tokens accepted of drafted (the share), tokens a step (generated tokens over steps, a step being one pass of the main model: generated minus accepted), and the time of a step. Pass 2 is listed where its counters differ.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | ub512 | ub512_1: 178 of 229 (0.777), 3.28 a step, 122.6 ms a step (pass 2: 181 of 220 (0.823), 3.41 a step, 121.2 ms a step); ub512_2: 178 of 229 (0.777), 3.28 a step, 125.6 ms a step (pass 2: 181 of 220 (0.823), 3.41 a step, 121.2 ms a step) |
|  | r100 | r100_1: 178 of 229 (0.777), 3.28 a step, 125.9 ms a step (pass 2: 181 of 220 (0.823), 3.41 a step, 121.5 ms a step); r100_2: 178 of 229 (0.777), 3.28 a step, 125.5 ms a step (pass 2: 181 of 220 (0.823), 3.41 a step, 121.2 ms a step) |
|  | u150 | u150_1: 178 of 229 (0.777), 3.28 a step, 125.4 ms a step (pass 2: 181 of 220 (0.823), 3.41 a step, 121.3 ms a step); u150_2: 178 of 229 (0.777), 3.28 a step, 125.7 ms a step (pass 2: 181 of 220 (0.823), 3.41 a step, 121.3 ms a step) |
| 4,096 | ub512 | ub512_1: 182 of 217 (0.839), 3.46 a step, 153.0 ms a step; ub512_2: 182 of 217 (0.839), 3.46 a step, 152.5 ms a step |
|  | r100 | r100_1: 182 of 217 (0.839), 3.46 a step, 165.6 ms a step; r100_2: 182 of 217 (0.839), 3.46 a step, 165.5 ms a step |
|  | u150 | u150_1: 178 of 231 (0.771), 3.28 a step, 184.9 ms a step; u150_2: 178 of 231 (0.771), 3.28 a step, 154.8 ms a step |
| 8,192 | ub512 | ub512_1: 185 of 210 (0.881), 3.61 a step, 151.6 ms a step (pass 2: 186 of 207 (0.899), 3.66 a step, 149.0 ms a step); ub512_2: 185 of 210 (0.881), 3.61 a step, 153.1 ms a step (pass 2: 186 of 207 (0.899), 3.66 a step, 149.1 ms a step) |
|  | r100 | r100_1: 186 of 205 (0.907), 3.66 a step, 144.1 ms a step (pass 2: 187 of 202 (0.926), 3.71 a step, 141.8 ms a step); r100_2: 186 of 205 (0.907), 3.66 a step, 145.1 ms a step (pass 2: 187 of 202 (0.926), 3.71 a step, 141.4 ms a step) |
|  | u150 | u150_1: 185 of 209 (0.885), 3.61 a step, 174.8 ms a step (pass 2: 188 of 199 (0.945), 3.76 a step, 157.9 ms a step); u150_2: 185 of 209 (0.885), 3.61 a step, 161.7 ms a step (pass 2: 188 of 199 (0.945), 3.76 a step, 159.3 ms a step) |
| 16,384 | ub512 | ub512_1: 184 of 210 (0.876), 3.56 a step, 187.8 ms a step; ub512_2: 184 of 210 (0.876), 3.56 a step, 167.2 ms a step |
|  | r100 | r100_1: 184 of 210 (0.876), 3.56 a step, 167.5 ms a step; r100_2: 184 of 210 (0.876), 3.56 a step, 179.7 ms a step |
|  | u150 | u150_1: 185 of 209 (0.885), 3.61 a step, 173.1 ms a step; u150_2: 185 of 209 (0.885), 3.61 a step, 152.6 ms a step |
| 32,768 | ub512 | ub512_1: 164 of 271 (0.605), 2.78 a step, 180.0 ms a step (pass 2: 165 of 268 (0.616), 2.81 a step, 164.8 ms a step); ub512_2: 164 of 271 (0.605), 2.78 a step, 171.2 ms a step (pass 2: 165 of 268 (0.616), 2.81 a step, 164.6 ms a step) |
|  | r100 | r100_1: 164 of 271 (0.605), 2.78 a step, 171.7 ms a step (pass 2: 165 of 268 (0.616), 2.81 a step, 167.0 ms a step); r100_2: 164 of 271 (0.605), 2.78 a step, 170.3 ms a step (pass 2: 165 of 268 (0.616), 2.81 a step, 164.5 ms a step) |
|  | u150 | u150_1: 164 of 270 (0.607), 2.78 a step, 178.5 ms a step (pass 2: 165 of 267 (0.618), 2.81 a step, 166.5 ms a step); u150_2: 164 of 270 (0.607), 2.78 a step, 171.8 ms a step (pass 2: 165 of 267 (0.618), 2.81 a step, 166.1 ms a step) |
| 49,152 | ub512 | ub512_1: 182 of 218 (0.835), 3.46 a step, 209.9 ms a step; ub512_2: 182 of 218 (0.835), 3.46 a step, 202.1 ms a step |
|  | r100 | r100_1: 182 of 218 (0.835), 3.46 a step, 202.1 ms a step; r100_2: 182 of 218 (0.835), 3.46 a step, 213.2 ms a step |
|  | u150 | u150_1: 183 of 213 (0.859), 3.51 a step, 202.3 ms a step; u150_2: 183 of 213 (0.859), 3.51 a step, 202.1 ms a step |

### Checks

- Times: every load started on a machine checked quiet (valid=1).
- Prompts: 64 ids at depth 0, 4096 ids at depth 4,096, 8192 ids at depth 8,192, 16384 ids at depth 16,384, 32768 ids at depth 32,768, 49152 ids at depth 49,152; the same ids on every load and pass.
- Tokens out: 256 on every timed request.
- Greedy reply at depth 0: the same on every pass and load of every arm.
- Greedy reply at depth 4,096: **differs from `ub512` in u150.**
- Greedy reply at depth 8,192: **differs from `ub512` in r100, u150.**
- Greedy reply at depth 16,384: **differs from `ub512` in u150.**
- Greedy reply at depth 32,768: **differs from `ub512` in u150.**
- Greedy reply at depth 49,152: **differs from `ub512` in u150.**
- Primes (read on cached, the cache's source): 4,096: 4032 on 0 (none); 8,192: 4612 on 3516 (checkpoint); 16,384: 12804 on 3516 (checkpoint); 32,768: 16900 on 15804 (checkpoint); 49,152: 16900 on 32188 (checkpoint).
- Graph reallocations of the target, requests with one per load: ub512_1: 0 of 18 requests; r100_1: 0 of 18 requests; u150_1: 0 of 18 requests; ub512_2: 0 of 18 requests; r100_2: 0 of 18 requests; u150_2: 0 of 18 requests.
