## The deep lane: c10, r075, r0625

Model file `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf`, llama-server version: 1.0.0-dev (build 12967, commit 6026911533).

256 tokens per request at temperature 0, each request twice per load (pass 1 is the first request at a new depth, pass 2 the same request again). Timed loads, in their order: c10a, r075a, r0625a, r0625b, r075b, c10b.

| Arm | Loads | Draft | Context | Its own variables and flags |
|---|---|---|---:|---|
| c10 (control) | c10a, c10b | on | 65536 | `GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=1.0` |
| r075 | r075a, r075b | on | 65536 | `GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=0.75` |
| r0625 | r0625a, r0625b | on | 65536 | `GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=0.625` |

### Generation, tok/s

Per load: pass 1, pass 2 → the load, with the load's time per token. A load's speed is its two passes' tokens over their time. Spread: the difference between the arm's fastest and slowest load. Class against `c10`: gain when the mean is above the control's by more than the larger of the two spreads, loss when below by more than it, else negligible.

| Depth (tokens) | Arm | Per load | Spread | Against `c10` | | Class |
|---:|---|---|---:|---:|---:|---|
| 4,096 | c10 | c10a: 31.49, 32.86 → **32.16** (31.09 ms a token); c10b: 31.49, 32.86 → **32.16** (31.09 ms a token) | 0.00 | control | | |
|  | r075 | r075a: 31.54, 32.87 → **32.19** (31.06 ms a token); r075b: 31.57, 32.88 → **32.21** (31.05 ms a token) | 0.02 | +0.04 tok/s | +0.1 % | gain |
|  | r0625 | r0625a: 31.61, 32.85 → **32.22** (31.04 ms a token); r0625b: 31.48, 32.88 → **32.16** (31.09 ms a token) | 0.05 | +0.03 tok/s | +0.1 % | negligible |
| 5,120 | c10 | c10a: 34.32, 35.67 → **34.98** (28.58 ms a token); c10b: 34.36, 35.67 → **35.00** (28.57 ms a token) | 0.02 | control | | |
|  | r075 | r075a: 34.25, 35.55 → **34.89** (28.67 ms a token); r075b: 34.29, 35.68 → **34.97** (28.59 ms a token) | 0.09 | -0.06 tok/s | -0.2 % | negligible |
|  | r0625 | r0625a: 34.10, 35.56 → **34.81** (28.72 ms a token); r0625b: 34.02, 35.56 → **34.77** (28.76 ms a token) | 0.04 | -0.20 tok/s | -0.6 % | loss |
| 6,144 | c10 | c10a: 30.18, 35.52 → **32.64** (30.64 ms a token); c10b: 33.50, 35.52 → **34.48** (29.00 ms a token) | 1.84 | control | | |
|  | r075 | r075a: 32.96, 35.09 → **33.99** (29.42 ms a token); r075b: 33.12, 35.10 → **34.08** (29.34 ms a token) | 0.09 | +0.48 tok/s | +1.4 % | negligible |
|  | r0625 | r0625a: 34.36, 36.11 → **35.21** (28.40 ms a token); r0625b: 34.21, 36.43 → **35.29** (28.34 ms a token) | 0.07 | +1.69 tok/s | +5.0 % | negligible |
| 7,168 | c10 | c10a: 29.43, 33.86 → **31.49** (31.76 ms a token); c10b: 31.52, 33.82 → **32.63** (30.65 ms a token) | 1.14 | control | | |
|  | r075 | r075a: 32.62, 34.49 → **33.53** (29.82 ms a token); r075b: 32.75, 34.51 → **33.61** (29.75 ms a token) | 0.08 | +1.51 tok/s | +4.7 % | gain |
|  | r0625 | r0625a: 32.63, 34.49 → **33.53** (29.82 ms a token); r0625b: 32.67, 34.50 → **33.56** (29.80 ms a token) | 0.03 | +1.48 tok/s | +4.6 % | gain |
| 8,192 | c10 | c10a: 31.74, 35.83 → **33.66** (29.71 ms a token); c10b: 34.50, 35.78 → **35.13** (28.47 ms a token) | 1.47 | control | | |
|  | r075 | r075a: 32.07, 33.31 → **32.68** (30.60 ms a token); r075b: 32.10, 33.35 → **32.71** (30.57 ms a token) | 0.03 | -1.70 tok/s | -4.9 % | loss |
|  | r0625 | r0625a: 31.89, 32.25 → **32.07** (31.18 ms a token); r0625b: 31.98, 33.38 → **32.66** (30.62 ms a token) | 0.59 | -2.03 tok/s | -5.9 % | loss |

### Generation over all depths, tok/s

Every depth's tokens over their time, per load.

| Depth (tokens) | Arm | Per load | Spread | Against `c10` | | Class |
|---:|---|---|---:|---:|---:|---|
| all | c10 | c10a: **32.94**; c10b: **33.83** | 0.89 | control | | |
|  | r075 | r075a: **33.43**; r075b: **33.49** | 0.06 | +0.07 tok/s | +0.2 % | negligible |
|  | r0625 | r0625a: **33.52**; r0625b: **33.65** | 0.13 | +0.20 tok/s | +0.6 % | negligible |

### The step on pass 1 (the first request at a new depth), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `c10`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `c10` | | Class |
|---:|---|---|---:|---:|---:|---|
| 4,096 | c10 | c10a: 104.22 (78 steps); c10b: 104.24 (78 steps) | 0.02 | control | | |
|  | r075 | r075a: 104.05 (78 steps); r075b: 103.97 (78 steps) | 0.08 | -0.22 ms | -0.2 % | gain |
|  | r0625 | r0625a: 103.84 (78 steps); r0625b: 104.27 (78 steps) | 0.44 | -0.17 ms | -0.2 % | negligible |
| 5,120 | c10 | c10a: 106.55 (70 steps); c10b: 106.45 (70 steps) | 0.10 | control | | |
|  | r075 | r075a: 106.78 (70 steps); r075b: 106.65 (70 steps) | 0.14 | +0.21 ms | +0.2 % | loss |
|  | r0625 | r0625a: 107.25 (70 steps); r0625b: 107.50 (70 steps) | 0.25 | +0.88 ms | +0.8 % | loss |
| 6,144 | c10 | c10a: 121.16 (70 steps); c10b: 109.18 (70 steps) | 11.98 | control | | |
|  | r075 | r075a: 107.86 (72 steps); r075b: 107.36 (72 steps) | 0.50 | -7.56 ms | -6.6 % | negligible |
|  | r0625 | r0625a: 107.99 (69 steps); r0625b: 108.45 (69 steps) | 0.46 | -6.95 ms | -6.0 % | negligible |
| 7,168 | c10 | c10a: 120.81 (72 steps); c10b: 112.81 (72 steps) | 8.00 | control | | |
|  | r075 | r075a: 107.50 (73 steps); r075b: 107.08 (73 steps) | 0.41 | -9.52 ms | -8.1 % | gain |
|  | r0625 | r0625a: 107.49 (73 steps); r0625b: 107.34 (73 steps) | 0.15 | -9.40 ms | -8.0 % | gain |
| 8,192 | c10 | c10a: 115.22 (70 steps); c10b: 106.01 (70 steps) | 9.22 | control | | |
|  | r075 | r075a: 107.86 (74 steps); r075b: 107.76 (74 steps) | 0.09 | -2.80 ms | -2.5 % | negligible |
|  | r0625 | r0625a: 108.48 (74 steps); r0625b: 108.19 (74 steps) | 0.29 | -2.28 ms | -2.1 % | negligible |

### The step on pass 2 (the same request again), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `c10`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `c10` | | Class |
|---:|---|---|---:|---:|---:|---|
| 4,096 | c10 | c10a: 99.88 (78 steps); c10b: 99.87 (78 steps) | 0.01 | control | | |
|  | r075 | r075a: 99.85 (78 steps); r075b: 99.83 (78 steps) | 0.03 | -0.03 ms | -0.0 % | gain |
|  | r0625 | r0625a: 99.90 (78 steps); r0625b: 99.81 (78 steps) | 0.09 | -0.02 ms | -0.0 % | negligible |
| 5,120 | c10 | c10a: 102.53 (70 steps); c10b: 102.54 (70 steps) | 0.01 | control | | |
|  | r075 | r075a: 102.89 (70 steps); r075b: 102.49 (70 steps) | 0.40 | +0.15 ms | +0.2 % | negligible |
|  | r0625 | r0625a: 102.85 (70 steps); r0625b: 102.83 (70 steps) | 0.02 | +0.31 ms | +0.3 % | loss |
| 6,144 | c10 | c10a: 104.44 (69 steps); c10b: 104.44 (69 steps) | 0.00 | control | | |
|  | r075 | r075a: 102.75 (71 steps); r075b: 102.71 (71 steps) | 0.04 | -1.71 ms | -1.6 % | gain |
|  | r0625 | r0625a: 104.25 (68 steps); r0625b: 103.34 (68 steps) | 0.91 | -0.64 ms | -0.6 % | negligible |
| 7,168 | c10 | c10a: 106.50 (71 steps); c10b: 106.60 (71 steps) | 0.10 | control | | |
|  | r075 | r075a: 103.09 (72 steps); r075b: 103.02 (72 steps) | 0.08 | -3.50 ms | -3.3 % | gain |
|  | r0625 | r0625a: 103.10 (72 steps); r0625b: 103.07 (72 steps) | 0.03 | -3.46 ms | -3.2 % | gain |
| 8,192 | c10 | c10a: 103.54 (69 steps); c10b: 103.69 (69 steps) | 0.15 | control | | |
|  | r075 | r075a: 103.85 (74 steps); r075b: 103.74 (74 steps) | 0.11 | +0.19 ms | +0.2 % | loss |
|  | r0625 | r0625a: 107.27 (74 steps); r0625b: 103.64 (74 steps) | 3.63 | +1.84 ms | +1.8 % | negligible |

### Prompt reading, tok/s

Per load: the tokens read at that depth over their time (a prime request reads a chunk on top of the cache; without primes each decode request reads its prompt), with the tokens read and the tokens reused. Requests that read under 32 tokens are left out.

| Depth (tokens) | Arm | Per load | Spread | Against `c10` | | Class |
|---:|---|---|---:|---:|---:|---|
| 4,096 | c10 | c10a: 148.9 (3972 read on 60 cached); c10b: 161.0 (3972 read on 60 cached) | 12.1 | control | | |
|  | r075 | r075a: 168.5 (3972 read on 60 cached); r075b: 164.4 (3972 read on 60 cached) | 4.1 | +11.5 tok/s | +7.4 % | negligible |
|  | r0625 | r0625a: 165.3 (3972 read on 60 cached); r0625b: 168.5 (3972 read on 60 cached) | 3.2 | +12.0 tok/s | +7.7 % | negligible |
| 5,120 | c10 | c10a: 241.4 (4996 read on 60 cached); c10b: 242.3 (4996 read on 60 cached) | 0.9 | control | | |
|  | r075 | r075a: 243.2 (4996 read on 60 cached); r075b: 247.1 (4996 read on 60 cached) | 3.9 | +3.3 tok/s | +1.4 % | negligible |
|  | r0625 | r0625a: 245.5 (4996 read on 60 cached); r0625b: 239.5 (4996 read on 60 cached) | 6.1 | +0.6 tok/s | +0.3 % | negligible |
| 6,144 | c10 | c10a: 318.9 (6020 read on 60 cached); c10b: 263.6 (6020 read on 60 cached) | 55.3 | control | | |
|  | r075 | r075a: 299.4 (6020 read on 60 cached); r075b: 309.8 (6020 read on 60 cached) | 10.3 | +13.4 tok/s | +4.6 % | negligible |
|  | r0625 | r0625a: 305.0 (6020 read on 60 cached); r0625b: 307.3 (6020 read on 60 cached) | 2.3 | +14.9 tok/s | +5.1 % | negligible |
| 7,168 | c10 | c10a: 341.5 (7044 read on 60 cached); c10b: 320.4 (7044 read on 60 cached) | 21.1 | control | | |
|  | r075 | r075a: 295.7 (7044 read on 60 cached); r075b: 329.2 (7044 read on 60 cached) | 33.5 | -18.5 tok/s | -5.6 % | negligible |
|  | r0625 | r0625a: 267.1 (7044 read on 60 cached); r0625b: 308.8 (7044 read on 60 cached) | 41.7 | -43.0 tok/s | -13.0 % | loss |
| 8,192 | c10 | c10a: 328.7 (8068 read on 60 cached); c10b: 330.1 (8068 read on 60 cached) | 1.4 | control | | |
|  | r075 | r075a: 316.2 (8068 read on 60 cached); r075b: 314.3 (8068 read on 60 cached) | 2.0 | -14.2 tok/s | -4.3 % | loss |
|  | r0625 | r0625a: 305.8 (8068 read on 60 cached); r0625b: 285.6 (8068 read on 60 cached) | 20.2 | -33.7 tok/s | -10.2 % | loss |

### The draft and the step

Per load, pass 1: draft tokens accepted of drafted (the share), tokens a step (generated tokens over steps, a step being one pass of the main model: generated minus accepted), and the time of a step. Pass 2 is listed where its counters differ.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 4,096 | c10 | c10a: 178 of 229 (0.777), 3.28 a step, 104.2 ms a step; c10b: 178 of 229 (0.777), 3.28 a step, 104.2 ms a step |
|  | r075 | r075a: 178 of 229 (0.777), 3.28 a step, 104.0 ms a step; r075b: 178 of 229 (0.777), 3.28 a step, 104.0 ms a step |
|  | r0625 | r0625a: 178 of 229 (0.777), 3.28 a step, 103.8 ms a step; r0625b: 178 of 229 (0.777), 3.28 a step, 104.3 ms a step |
| 5,120 | c10 | c10a: 186 of 206 (0.903), 3.66 a step, 106.6 ms a step; c10b: 186 of 206 (0.903), 3.66 a step, 106.4 ms a step |
|  | r075 | r075a: 186 of 206 (0.903), 3.66 a step, 106.8 ms a step; r075b: 186 of 206 (0.903), 3.66 a step, 106.6 ms a step |
|  | r0625 | r0625a: 186 of 206 (0.903), 3.66 a step, 107.2 ms a step; r0625b: 186 of 206 (0.903), 3.66 a step, 107.5 ms a step |
| 6,144 | c10 | c10a: 186 of 204 (0.912), 3.66 a step, 121.2 ms a step (pass 2: 187 of 201 (0.930), 3.71 a step, 104.4 ms a step); c10b: 186 of 204 (0.912), 3.66 a step, 109.2 ms a step (pass 2: 187 of 201 (0.930), 3.71 a step, 104.4 ms a step) |
|  | r075 | r075a: 184 of 210 (0.876), 3.56 a step, 107.9 ms a step (pass 2: 185 of 207 (0.894), 3.61 a step, 102.8 ms a step); r075b: 184 of 210 (0.876), 3.56 a step, 107.4 ms a step (pass 2: 185 of 207 (0.894), 3.61 a step, 102.7 ms a step) |
|  | r0625 | r0625a: 187 of 203 (0.921), 3.71 a step, 108.0 ms a step (pass 2: 188 of 200 (0.940), 3.76 a step, 104.3 ms a step); r0625b: 187 of 203 (0.921), 3.71 a step, 108.4 ms a step (pass 2: 188 of 200 (0.940), 3.76 a step, 103.3 ms a step) |
| 7,168 | c10 | c10a: 184 of 210 (0.876), 3.56 a step, 120.8 ms a step (pass 2: 185 of 207 (0.894), 3.61 a step, 106.5 ms a step); c10b: 184 of 210 (0.876), 3.56 a step, 112.8 ms a step (pass 2: 185 of 207 (0.894), 3.61 a step, 106.6 ms a step) |
|  | r075 | r075a: 183 of 213 (0.859), 3.51 a step, 107.5 ms a step (pass 2: 184 of 210 (0.876), 3.56 a step, 103.1 ms a step); r075b: 183 of 213 (0.859), 3.51 a step, 107.1 ms a step (pass 2: 184 of 210 (0.876), 3.56 a step, 103.0 ms a step) |
|  | r0625 | r0625a: 183 of 213 (0.859), 3.51 a step, 107.5 ms a step (pass 2: 184 of 210 (0.876), 3.56 a step, 103.1 ms a step); r0625b: 183 of 213 (0.859), 3.51 a step, 107.3 ms a step (pass 2: 184 of 210 (0.876), 3.56 a step, 103.1 ms a step) |
| 8,192 | c10 | c10a: 186 of 207 (0.899), 3.66 a step, 115.2 ms a step (pass 2: 187 of 204 (0.917), 3.71 a step, 103.5 ms a step); c10b: 186 of 207 (0.899), 3.66 a step, 106.0 ms a step (pass 2: 187 of 204 (0.917), 3.71 a step, 103.7 ms a step) |
|  | r075 | r075a: 182 of 218 (0.835), 3.46 a step, 107.9 ms a step; r075b: 182 of 218 (0.835), 3.46 a step, 107.8 ms a step |
|  | r0625 | r0625a: 182 of 218 (0.835), 3.46 a step, 108.5 ms a step; r0625b: 182 of 218 (0.835), 3.46 a step, 108.2 ms a step |

### The sampler pass

One request per depth and load with the server line's own sampler and a fixed seed: tok/s, the draft counters and the step.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 4,096 | c10 | c10a: 30.66 tok/s, 175 of 237 (0.738), 3.16 a step, 103.1 ms a step; c10b: 30.84 tok/s, 175 of 237 (0.738), 3.16 a step, 102.5 ms a step |
|  | r075 | r075a: 30.71 tok/s, 175 of 237 (0.738), 3.16 a step, 102.9 ms a step; r075b: 30.72 tok/s, 175 of 237 (0.738), 3.16 a step, 102.9 ms a step |
|  | r0625 | r0625a: 30.69 tok/s, 175 of 237 (0.738), 3.16 a step, 103.0 ms a step; r0625b: 30.71 tok/s, 175 of 237 (0.738), 3.16 a step, 102.9 ms a step |
| 5,120 | c10 | c10a: 32.00 tok/s, 180 of 223 (0.807), 3.37 a step, 105.3 ms a step; c10b: 31.87 tok/s, 180 of 223 (0.807), 3.37 a step, 105.7 ms a step |
|  | r075 | r075a: 31.90 tok/s, 180 of 223 (0.807), 3.37 a step, 105.6 ms a step; r075b: 31.98 tok/s, 180 of 223 (0.807), 3.37 a step, 105.3 ms a step |
|  | r0625 | r0625a: 31.46 tok/s, 179 of 226 (0.792), 3.32 a step, 105.7 ms a step; r0625b: 31.46 tok/s, 179 of 226 (0.792), 3.32 a step, 105.7 ms a step |
| 6,144 | c10 | c10a: 34.57 tok/s, 187 of 202 (0.926), 3.71 a step, 107.3 ms a step; c10b: 34.46 tok/s, 187 of 202 (0.926), 3.71 a step, 107.7 ms a step |
|  | r075 | r075a: 35.08 tok/s, 186 of 204 (0.912), 3.66 a step, 104.2 ms a step; r075b: 35.02 tok/s, 186 of 204 (0.912), 3.66 a step, 104.4 ms a step |
|  | r0625 | r0625a: 36.21 tok/s, 189 of 195 (0.969), 3.82 a step, 105.5 ms a step; r0625b: 36.49 tok/s, 189 of 197 (0.959), 3.82 a step, 104.7 ms a step |
| 7,168 | c10 | c10a: 31.47 tok/s, 181 of 222 (0.815), 3.41 a step, 108.5 ms a step; c10b: 31.47 tok/s, 181 of 222 (0.815), 3.41 a step, 108.5 ms a step |
|  | r075 | r075a: 32.15 tok/s, 181 of 220 (0.823), 3.41 a step, 106.2 ms a step; r075b: 32.18 tok/s, 181 of 220 (0.823), 3.41 a step, 106.1 ms a step |
|  | r0625 | r0625a: 32.08 tok/s, 181 of 220 (0.823), 3.41 a step, 106.4 ms a step; r0625b: 32.13 tok/s, 181 of 220 (0.823), 3.41 a step, 106.2 ms a step |
| 8,192 | c10 | c10a: 31.81 tok/s, 180 of 222 (0.811), 3.37 a step, 105.9 ms a step; c10b: 31.79 tok/s, 180 of 222 (0.811), 3.37 a step, 106.0 ms a step |
|  | r075 | r075a: 30.61 tok/s, 177 of 233 (0.760), 3.24 a step, 105.9 ms a step; r075b: 30.69 tok/s, 177 of 233 (0.760), 3.24 a step, 105.6 ms a step |
|  | r0625 | r0625a: 30.36 tok/s, 177 of 233 (0.760), 3.24 a step, 106.7 ms a step; r0625b: 30.63 tok/s, 177 of 233 (0.760), 3.24 a step, 105.8 ms a step |

### Checks

- Times: every load started on a machine checked quiet (valid=1).
- Prompts: 4096 ids at depth 4,096, 5120 ids at depth 5,120, 6144 ids at depth 6,144, 7168 ids at depth 7,168, 8192 ids at depth 8,192; the same ids on every load and pass.
- Tokens out: 256 on every timed request.
- Greedy reply at depth 4,096: the same on every pass and load of every arm.
- Greedy reply at depth 5,120: the same on every pass and load of every arm.
- Greedy reply at depth 6,144: **differs from `c10` in r075, r0625.**
- Greedy reply at depth 7,168: **differs from `c10` in r075, r0625.**
- Greedy reply at depth 8,192: **differs from `c10` in r075, r0625.**
- Primes (read on cached, the cache's source): 4,096: 3972 on 60 (checkpoint); 5,120: 4996 on 60 (checkpoint); 6,144: 6020 on 60 (checkpoint); 7,168: 7044 on 60 (checkpoint); 8,192: 8068 on 60 (checkpoint).
- Graph reallocations of the target, requests with one per load: c10a: 0 of 21 requests; r075a: 0 of 21 requests; r0625a: 0 of 21 requests; r0625b: 0 of 21 requests; r075b: 0 of 21 requests; c10b: 0 of 21 requests.
