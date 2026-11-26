## The deep lane: c, r375, r25

Model file `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf`, llama-server version: 1.0.0-dev (build 13218, commit eee615cdb9).

256 tokens per request at temperature 0, each request twice per load (pass 1 is the first request at a new depth, pass 2 the same request again). Timed loads, in their order: ca, r375a, r25a, r25b, r375b, cb.

| Arm | Loads | Draft | Context | Its own variables and flags |
|---|---|---|---:|---|
| c (control) | ca, cb | on | 65536 | `GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=10 GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=0.5` |
| r375 | r375a, r375b | on | 65536 | `GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=10 GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=0.375` |
| r25 | r25a, r25b | on | 65536 | `GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=10 GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=0.25` |

### Generation, tok/s

Per load: pass 1, pass 2 → the load, with the load's time per token. A load's speed is its two passes' tokens over their time. Spread: the difference between the arm's fastest and slowest load. Class against `c`: gain when the mean is above the control's by more than the larger of the two spreads, loss when below by more than it, else negligible.

| Depth (tokens) | Arm | Per load | Spread | Against `c` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | c | ca: 24.72, 26.22 → **25.45** (39.30 ms a token); cb: 24.76, 26.17 → **25.44** (39.30 ms a token) | 0.00 | control | | |
|  | r375 | r375a: 24.78, 26.15 → **25.44** (39.30 ms a token); r375b: 21.15, 26.17 → **23.39** (42.75 ms a token) | 2.05 | -1.03 tok/s | -4.0 % | negligible |
|  | r25 | r25a: 24.69, 26.08 → **25.37** (39.42 ms a token); r25b: 24.72, 26.23 → **25.45** (39.29 ms a token) | 0.08 | -0.03 tok/s | -0.1 % | negligible |
| 2,048 | c | ca: 33.52, 35.61 → **34.53** (28.96 ms a token); cb: 33.49, 35.60 → **34.51** (28.98 ms a token) | 0.02 | control | | |
|  | r375 | r375a: 33.59, 35.61 → **34.57** (28.93 ms a token); r375b: 33.48, 35.19 → **34.31** (29.14 ms a token) | 0.26 | -0.08 tok/s | -0.2 % | negligible |
|  | r25 | r25a: 33.20, 35.75 → **34.43** (29.04 ms a token); r25b: 33.17, 35.73 → **34.40** (29.07 ms a token) | 0.03 | -0.11 tok/s | -0.3 % | loss |
| 3,072 | c | ca: 34.66, 35.90 → **35.27** (28.36 ms a token); cb: 34.44, 35.85 → **35.13** (28.47 ms a token) | 0.14 | control | | |
|  | r375 | r375a: 34.85, 35.61 → **35.23** (28.39 ms a token); r375b: 34.83, 35.60 → **35.21** (28.40 ms a token) | 0.01 | +0.02 tok/s | +0.1 % | negligible |
|  | r25 | r25a: 34.96, 36.28 → **35.61** (28.08 ms a token); r25b: 34.84, 36.26 → **35.54** (28.14 ms a token) | 0.07 | +0.37 tok/s | +1.1 % | gain |
| 4,096 | c | ca: 30.87, 32.01 → **31.43** (31.82 ms a token); cb: 30.83, 31.99 → **31.40** (31.85 ms a token) | 0.03 | control | | |
|  | r375 | r375a: 32.76, 34.17 → **33.45** (29.89 ms a token); r375b: 32.78, 34.12 → **33.44** (29.91 ms a token) | 0.01 | +2.03 tok/s | +6.5 % | gain |
|  | r25 | r25a: 32.31, 34.11 → **33.19** (30.13 ms a token); r25b: 32.32, 34.10 → **33.19** (30.13 ms a token) | 0.00 | +1.77 tok/s | +5.6 % | gain |

### Generation over all depths, tok/s

Every depth's tokens over their time, per load.

| Depth (tokens) | Arm | Per load | Spread | Against `c` | | Class |
|---:|---|---|---:|---:|---:|---|
| all | c | ca: **31.15**; cb: **31.10** | 0.04 | control | | |
|  | r375 | r375a: **31.62**; r375b: **30.72** | 0.90 | +0.04 tok/s | +0.1 % | negligible |
|  | r25 | r25a: **31.58**; r25b: **31.59** | 0.01 | +0.46 tok/s | +1.5 % | gain |

### The step on pass 1 (the first request at a new depth), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `c`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `c` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | c | ca: 95.01 (109 steps); cb: 94.87 (109 steps) | 0.14 | control | | |
|  | r375 | r375a: 94.79 (109 steps); r375b: 111.07 (109 steps) | 16.28 | +7.99 ms | +8.4 % | negligible |
|  | r25 | r25a: 95.11 (109 steps); r25b: 95.01 (109 steps) | 0.11 | +0.12 ms | +0.1 % | negligible |
| 2,048 | c | ca: 100.49 (76 steps); cb: 100.58 (76 steps) | 0.09 | control | | |
|  | r375 | r375a: 100.27 (76 steps); r375b: 100.61 (76 steps) | 0.34 | -0.09 ms | -0.1 % | negligible |
|  | r25 | r25a: 102.80 (75 steps); r25b: 102.91 (75 steps) | 0.11 | +2.32 ms | +2.3 % | loss |
| 3,072 | c | ca: 101.18 (73 steps); cb: 101.84 (73 steps) | 0.66 | control | | |
|  | r375 | r375a: 100.63 (73 steps); r375b: 100.67 (73 steps) | 0.04 | -0.86 ms | -0.8 % | gain |
|  | r25 | r25a: 101.71 (72 steps); r25b: 102.06 (72 steps) | 0.36 | +0.37 ms | +0.4 % | negligible |
| 4,096 | c | ca: 102.38 (81 steps); cb: 102.52 (81 steps) | 0.14 | control | | |
|  | r375 | r375a: 102.82 (76 steps); r375b: 102.77 (76 steps) | 0.05 | +0.34 ms | +0.3 % | loss |
|  | r25 | r25a: 102.91 (77 steps); r25b: 102.85 (77 steps) | 0.05 | +0.43 ms | +0.4 % | loss |

### The step on pass 2 (the same request again), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `c`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `c` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | c | ca: 90.42 (108 steps); cb: 90.59 (108 steps) | 0.17 | control | | |
|  | r375 | r375a: 90.66 (108 steps); r375b: 90.59 (108 steps) | 0.06 | +0.12 ms | +0.1 % | negligible |
|  | r25 | r25a: 90.88 (108 steps); r25b: 90.37 (108 steps) | 0.51 | +0.12 ms | +0.1 % | negligible |
| 2,048 | c | ca: 95.85 (75 steps); cb: 95.89 (75 steps) | 0.04 | control | | |
|  | r375 | r375a: 95.86 (75 steps); r375b: 97.00 (75 steps) | 1.14 | +0.56 ms | +0.6 % | negligible |
|  | r25 | r25a: 98.09 (73 steps); r25b: 98.15 (73 steps) | 0.06 | +2.25 ms | +2.3 % | loss |
| 3,072 | c | ca: 97.69 (73 steps); cb: 97.82 (73 steps) | 0.13 | control | | |
|  | r375 | r375a: 98.48 (73 steps); r375b: 98.51 (73 steps) | 0.04 | +0.74 ms | +0.8 % | loss |
|  | r25 | r25a: 98.01 (72 steps); r25b: 98.05 (72 steps) | 0.04 | +0.27 ms | +0.3 % | loss |
| 4,096 | c | ca: 98.73 (81 steps); cb: 98.79 (81 steps) | 0.06 | control | | |
|  | r375 | r375a: 98.58 (76 steps); r375b: 98.71 (76 steps) | 0.13 | -0.12 ms | -0.1 % | negligible |
|  | r25 | r25a: 98.74 (76 steps); r25b: 98.77 (76 steps) | 0.03 | -0.00 ms | -0.0 % | negligible |

### Prompt reading, tok/s

Per load: the tokens read at that depth over their time (a prime request reads a chunk on top of the cache; without primes each decode request reads its prompt), with the tokens read and the tokens reused. Requests that read under 32 tokens are left out.

| Depth (tokens) | Arm | Per load | Spread | Against `c` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | c | ca: 54.6 (64 read on 0 cached); cb: 55.6 (64 read on 0 cached) | 1.0 | control | | |
|  | r375 | r375a: 54.8 (64 read on 0 cached); r375b: 58.4 (64 read on 0 cached) | 3.6 | +1.5 tok/s | +2.8 % | negligible |
|  | r25 | r25a: 55.7 (64 read on 0 cached); r25b: 55.6 (64 read on 0 cached) | 0.1 | +0.5 tok/s | +1.0 % | negligible |
| 2,048 | c | ca: 161.8 (1984 read on 0 cached); cb: 159.7 (1984 read on 0 cached) | 2.1 | control | | |
|  | r375 | r375a: 153.7 (1984 read on 0 cached); r375b: 157.9 (1984 read on 0 cached) | 4.2 | -4.9 tok/s | -3.1 % | loss |
|  | r25 | r25a: 162.6 (1984 read on 0 cached); r25b: 167.5 (1984 read on 0 cached) | 5.0 | +4.3 tok/s | +2.7 % | negligible |
| 3,072 | c | ca: 130.7 (1540 read on 1468 cached); cb: 135.0 (1540 read on 1468 cached) | 4.3 | control | | |
|  | r375 | r375a: 129.4 (1540 read on 1468 cached); r375b: 136.6 (1540 read on 1468 cached) | 7.2 | +0.1 tok/s | +0.1 % | negligible |
|  | r25 | r25a: 135.6 (1540 read on 1468 cached); r25b: 130.3 (1540 read on 1468 cached) | 5.3 | +0.1 tok/s | +0.1 % | negligible |
| 4,096 | c | ca: 292.4 (2564 read on 1468 cached); cb: 294.1 (2564 read on 1468 cached) | 1.6 | control | | |
|  | r375 | r375a: 295.0 (2564 read on 1468 cached); r375b: 294.0 (2564 read on 1468 cached) | 1.0 | +1.2 tok/s | +0.4 % | negligible |
|  | r25 | r25a: 289.0 (2564 read on 1468 cached); r25b: 295.4 (2564 read on 1468 cached) | 6.4 | -1.0 tok/s | -0.4 % | negligible |

### The draft and the step

Per load, pass 1: draft tokens accepted of drafted (the share), tokens a step (generated tokens over steps, a step being one pass of the main model: generated minus accepted), and the time of a step. Pass 2 is listed where its counters differ.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | c | ca: 147 of 321 (0.458), 2.35 a step, 95.0 ms a step (pass 2: 148 of 318 (0.465), 2.37 a step, 90.4 ms a step); cb: 147 of 321 (0.458), 2.35 a step, 94.9 ms a step (pass 2: 148 of 318 (0.465), 2.37 a step, 90.6 ms a step) |
|  | r375 | r375a: 147 of 321 (0.458), 2.35 a step, 94.8 ms a step (pass 2: 148 of 318 (0.465), 2.37 a step, 90.7 ms a step); r375b: 147 of 321 (0.458), 2.35 a step, 111.1 ms a step (pass 2: 148 of 318 (0.465), 2.37 a step, 90.6 ms a step) |
|  | r25 | r25a: 147 of 321 (0.458), 2.35 a step, 95.1 ms a step (pass 2: 148 of 318 (0.465), 2.37 a step, 90.9 ms a step); r25b: 147 of 321 (0.458), 2.35 a step, 95.0 ms a step (pass 2: 148 of 318 (0.465), 2.37 a step, 90.4 ms a step) |
| 2,048 | c | ca: 180 of 224 (0.804), 3.37 a step, 100.5 ms a step (pass 2: 181 of 221 (0.819), 3.41 a step, 95.9 ms a step); cb: 180 of 224 (0.804), 3.37 a step, 100.6 ms a step (pass 2: 181 of 221 (0.819), 3.41 a step, 95.9 ms a step) |
|  | r375 | r375a: 180 of 224 (0.804), 3.37 a step, 100.3 ms a step (pass 2: 181 of 221 (0.819), 3.41 a step, 95.9 ms a step); r375b: 180 of 224 (0.804), 3.37 a step, 100.6 ms a step (pass 2: 181 of 221 (0.819), 3.41 a step, 97.0 ms a step) |
|  | r25 | r25a: 181 of 221 (0.819), 3.41 a step, 102.8 ms a step (pass 2: 183 of 215 (0.851), 3.51 a step, 98.1 ms a step); r25b: 181 of 221 (0.819), 3.41 a step, 102.9 ms a step (pass 2: 183 of 215 (0.851), 3.51 a step, 98.2 ms a step) |
| 3,072 | c | ca: 183 of 213 (0.859), 3.51 a step, 101.2 ms a step; cb: 183 of 213 (0.859), 3.51 a step, 101.8 ms a step |
|  | r375 | r375a: 183 of 216 (0.847), 3.51 a step, 100.6 ms a step; r375b: 183 of 216 (0.847), 3.51 a step, 100.7 ms a step |
|  | r25 | r25a: 184 of 210 (0.876), 3.56 a step, 101.7 ms a step; r25b: 184 of 210 (0.876), 3.56 a step, 102.1 ms a step |
| 4,096 | c | ca: 175 of 237 (0.738), 3.16 a step, 102.4 ms a step; cb: 175 of 237 (0.738), 3.16 a step, 102.5 ms a step |
|  | r375 | r375a: 180 of 223 (0.807), 3.37 a step, 102.8 ms a step; r375b: 180 of 223 (0.807), 3.37 a step, 102.8 ms a step |
|  | r25 | r25a: 179 of 226 (0.792), 3.32 a step, 102.9 ms a step (pass 2: 180 of 223 (0.807), 3.37 a step, 98.7 ms a step); r25b: 179 of 226 (0.792), 3.32 a step, 102.9 ms a step (pass 2: 180 of 223 (0.807), 3.37 a step, 98.8 ms a step) |

### The sampler pass

One request per depth and load with the server line's own sampler and a fixed seed: tok/s, the draft counters and the step.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | c | ca: 28.26 tok/s, 159 of 285 (0.558), 2.64 a step, 93.4 ms a step; cb: 28.42 tok/s, 159 of 285 (0.558), 2.64 a step, 92.9 ms a step |
|  | r375 | r375a: 28.49 tok/s, 159 of 285 (0.558), 2.64 a step, 92.6 ms a step; r375b: 25.78 tok/s, 159 of 285 (0.558), 2.64 a step, 102.4 ms a step |
|  | r25 | r25a: 25.99 tok/s, 154 of 302 (0.510), 2.51 a step, 96.6 ms a step; r25b: 25.89 tok/s, 154 of 302 (0.510), 2.51 a step, 96.9 ms a step |
| 2,048 | c | ca: 30.77 tok/s, 172 of 247 (0.696), 3.05 a step, 99.0 ms a step; cb: 30.81 tok/s, 172 of 247 (0.696), 3.05 a step, 98.9 ms a step |
|  | r375 | r375a: 30.29 tok/s, 172 of 247 (0.696), 3.05 a step, 100.6 ms a step; r375b: 30.68 tok/s, 172 of 247 (0.696), 3.05 a step, 99.3 ms a step |
|  | r25 | r25a: 33.09 tok/s, 178 of 231 (0.771), 3.28 a step, 99.2 ms a step; r25b: 32.90 tok/s, 178 of 231 (0.771), 3.28 a step, 99.8 ms a step |
| 3,072 | c | ca: 35.71 tok/s, 185 of 209 (0.885), 3.61 a step, 101.0 ms a step; cb: 35.73 tok/s, 185 of 209 (0.885), 3.61 a step, 100.9 ms a step |
|  | r375 | r375a: 36.95 tok/s, 187 of 201 (0.930), 3.71 a step, 100.4 ms a step; r375b: 36.96 tok/s, 187 of 201 (0.930), 3.71 a step, 100.4 ms a step |
|  | r25 | r25a: 35.31 tok/s, 185 of 207 (0.894), 3.61 a step, 102.1 ms a step; r25b: 35.29 tok/s, 185 of 207 (0.894), 3.61 a step, 102.2 ms a step |
| 4,096 | c | ca: 29.17 tok/s, 179 of 226 (0.792), 3.32 a step, 114.0 ms a step; cb: 31.93 tok/s, 179 of 226 (0.792), 3.32 a step, 104.1 ms a step |
|  | r375 | r375a: 33.74 tok/s, 180 of 223 (0.807), 3.37 a step, 99.8 ms a step; r375b: 33.70 tok/s, 180 of 223 (0.807), 3.37 a step, 100.0 ms a step |
|  | r25 | r25a: 30.22 tok/s, 173 of 243 (0.712), 3.08 a step, 102.1 ms a step; r25b: 30.29 tok/s, 173 of 243 (0.712), 3.08 a step, 101.8 ms a step |

### Checks

- Times: every load started on a machine checked quiet (valid=1).
- Prompts: 64 ids at depth 0, 2048 ids at depth 2,048, 3072 ids at depth 3,072, 4096 ids at depth 4,096; the same ids on every load and pass.
- Tokens out: 256 on every timed request.
- Greedy reply at depth 0: the same on every pass and load of every arm.
- Greedy reply at depth 2,048: the same on every pass and load of every arm.
- Greedy reply at depth 3,072: **differs from `c` in r375.**
- Greedy reply at depth 4,096: **differs from `c` in r375, r25.**
- Primes (read on cached, the cache's source): 2,048: 1984 on 0 (none); 3,072: 1540 on 1468 (checkpoint); 4,096: 2564 on 1468 (checkpoint).
- Graph reallocations of the target, requests with one per load: ca: 0 of 16 requests; r375a: 0 of 16 requests; r25a: 0 of 16 requests; r25b: 0 of 16 requests; r375b: 0 of 16 requests; cb: 0 of 16 requests.
