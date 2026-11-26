## The deep lane: c, r42, r45, r49

Model file `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf`, llama-server version: 1.0.0-dev (build 13218, commit eee615cdb9).

256 tokens per request at temperature 0, each request twice per load (pass 1 is the first request at a new depth, pass 2 the same request again). Timed loads, in their order: ca, r42a, r45a, r49a, r49b, r45b, r42b, cb.

| Arm | Loads | Draft | Context | Its own variables and flags |
|---|---|---|---:|---|
| c (control) | ca, cb | on | 65536 | `GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=10 GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=0.5` |
| r42 | r42a, r42b | on | 65536 | `GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=10 GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=0.42` |
| r45 | r45a, r45b | on | 65536 | `GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=10 GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=0.45` |
| r49 | r49a, r49b | on | 65536 | `GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=10 GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=0.49` |

### Generation, tok/s

Per load: pass 1, pass 2 → the load, with the load's time per token. A load's speed is its two passes' tokens over their time. Spread: the difference between the arm's fastest and slowest load. Class against `c`: gain when the mean is above the control's by more than the larger of the two spreads, loss when below by more than it, else negligible.

| Depth (tokens) | Arm | Per load | Spread | Against `c` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | c | ca: 24.99, 26.21 → **25.58** (39.09 ms a token); cb: 24.70, 26.16 → **25.41** (39.36 ms a token) | 0.18 | control | | |
|  | r42 | r42a: 24.72, 26.22 → **25.45** (39.29 ms a token); r42b: 24.72, 26.20 → **25.43** (39.32 ms a token) | 0.01 | -0.06 tok/s | -0.2 % | negligible |
|  | r45 | r45a: 24.75, 26.21 → **25.46** (39.28 ms a token); r45b: 24.87, 26.21 → **25.52** (39.19 ms a token) | 0.06 | -0.01 tok/s | -0.0 % | negligible |
|  | r49 | r49a: 24.60, 26.16 → **25.36** (39.44 ms a token); r49b: 24.71, 26.21 → **25.44** (39.30 ms a token) | 0.09 | -0.10 tok/s | -0.4 % | negligible |
| 3,328 | c | ca: 31.34, 34.42 → **32.81** (30.48 ms a token); cb: 32.51, 34.47 → **33.46** (29.88 ms a token) | 0.66 | control | | |
|  | r42 | r42a: 32.45, 34.54 → **33.46** (29.89 ms a token); r42b: 31.85, 34.65 → **33.19** (30.13 ms a token) | 0.27 | +0.19 tok/s | +0.6 % | negligible |
|  | r45 | r45a: 32.48, 34.52 → **33.47** (29.88 ms a token); r45b: 32.42, 34.55 → **33.45** (29.89 ms a token) | 0.02 | +0.32 tok/s | +1.0 % | negligible |
|  | r49 | r49a: 32.53, 34.43 → **33.45** (29.89 ms a token); r49b: 32.46, 34.50 → **33.45** (29.90 ms a token) | 0.00 | +0.31 tok/s | +0.9 % | negligible |
| 3,584 | c | ca: 27.30, 29.57 → **28.39** (35.22 ms a token); cb: 27.58, 29.57 → **28.54** (35.04 ms a token) | 0.15 | control | | |
|  | r42 | r42a: 32.69, 35.23 → **33.91** (29.49 ms a token); r42b: 32.70, 35.13 → **33.87** (29.52 ms a token) | 0.04 | +5.43 tok/s | +19.1 % | gain |
|  | r45 | r45a: 30.38, 32.21 → **31.27** (31.98 ms a token); r45b: 30.37, 32.15 → **31.24** (32.02 ms a token) | 0.04 | +2.79 tok/s | +9.8 % | gain |
|  | r49 | r49a: 27.60, 29.23 → **28.40** (35.22 ms a token); r49b: 27.51, 29.56 → **28.50** (35.09 ms a token) | 0.11 | -0.02 tok/s | -0.1 % | negligible |
| 3,840 | c | ca: 34.23, 35.58 → **34.89** (28.66 ms a token); cb: 34.28, 35.55 → **34.90** (28.65 ms a token) | 0.01 | control | | |
|  | r42 | r42a: 34.66, 35.92 → **35.28** (28.35 ms a token); r42b: 34.70, 35.98 → **35.33** (28.30 ms a token) | 0.05 | +0.41 tok/s | +1.2 % | gain |
|  | r45 | r45a: 34.53, 35.99 → **35.24** (28.37 ms a token); r45b: 34.58, 35.73 → **35.14** (28.46 ms a token) | 0.10 | +0.30 tok/s | +0.9 % | gain |
|  | r49 | r49a: 34.52, 36.00 → **35.24** (28.37 ms a token); r49b: 34.74, 35.90 → **35.31** (28.32 ms a token) | 0.07 | +0.38 tok/s | +1.1 % | gain |

### Generation over all depths, tok/s

Every depth's tokens over their time, per load.

| Depth (tokens) | Arm | Per load | Spread | Against `c` | | Class |
|---:|---|---|---:|---:|---:|---|
| all | c | ca: **29.97**; cb: **30.09** | 0.12 | control | | |
|  | r42 | r42a: **31.49**; r42b: **31.43** | 0.06 | +1.43 tok/s | +4.8 % | gain |
|  | r45 | r45a: **30.89**; r45b: **30.88** | 0.01 | +0.85 tok/s | +2.8 % | gain |
|  | r49 | r49a: **30.09**; r49b: **30.16** | 0.07 | +0.10 tok/s | +0.3 % | negligible |

### The step on pass 1 (the first request at a new depth), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `c`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `c` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | c | ca: 94.00 (109 steps); cb: 95.10 (109 steps) | 1.10 | control | | |
|  | r42 | r42a: 95.00 (109 steps); r42b: 95.03 (109 steps) | 0.03 | +0.47 ms | +0.5 % | negligible |
|  | r45 | r45a: 94.88 (109 steps); r45b: 94.44 (109 steps) | 0.44 | +0.12 ms | +0.1 % | negligible |
|  | r49 | r49a: 95.46 (109 steps); r49b: 95.03 (109 steps) | 0.43 | +0.70 ms | +0.7 % | negligible |
| 3,328 | c | ca: 107.47 (76 steps); cb: 103.60 (76 steps) | 3.87 | control | | |
|  | r42 | r42a: 103.81 (76 steps); r42b: 105.77 (76 steps) | 1.96 | -0.75 ms | -0.7 % | negligible |
|  | r45 | r45a: 103.71 (76 steps); r45b: 103.91 (76 steps) | 0.20 | -1.73 ms | -1.6 % | negligible |
|  | r49 | r49a: 103.56 (76 steps); r49b: 103.79 (76 steps) | 0.23 | -1.87 ms | -1.8 % | negligible |
| 3,584 | c | ca: 106.56 (88 steps); cb: 105.47 (88 steps) | 1.09 | control | | |
|  | r42 | r42a: 103.05 (76 steps); r42b: 103.00 (76 steps) | 0.05 | -2.99 ms | -2.8 % | gain |
|  | r45 | r45a: 102.76 (82 steps); r45b: 102.79 (82 steps) | 0.03 | -3.24 ms | -3.1 % | gain |
|  | r49 | r49a: 105.39 (88 steps); r49b: 105.73 (88 steps) | 0.34 | -0.45 ms | -0.4 % | negligible |
| 3,840 | c | ca: 103.88 (72 steps); cb: 103.73 (72 steps) | 0.15 | control | | |
|  | r42 | r42a: 102.58 (72 steps); r42b: 102.47 (72 steps) | 0.12 | -1.28 ms | -1.2 % | gain |
|  | r45 | r45a: 102.97 (72 steps); r45b: 102.83 (72 steps) | 0.14 | -0.90 ms | -0.9 % | gain |
|  | r49 | r49a: 103.00 (72 steps); r49b: 102.33 (72 steps) | 0.67 | -1.14 ms | -1.1 % | gain |

### The step on pass 2 (the same request again), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `c`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `c` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | c | ca: 90.43 (108 steps); cb: 90.60 (108 steps) | 0.17 | control | | |
|  | r42 | r42a: 90.40 (108 steps); r42b: 90.48 (108 steps) | 0.08 | -0.07 ms | -0.1 % | negligible |
|  | r45 | r45a: 90.44 (108 steps); r45b: 90.45 (108 steps) | 0.01 | -0.07 ms | -0.1 % | negligible |
|  | r49 | r49a: 90.62 (108 steps); r49b: 90.42 (108 steps) | 0.20 | +0.01 ms | +0.0 % | negligible |
| 3,328 | c | ca: 99.17 (75 steps); cb: 99.02 (75 steps) | 0.15 | control | | |
|  | r42 | r42a: 98.83 (75 steps); r42b: 98.50 (75 steps) | 0.33 | -0.43 ms | -0.4 % | gain |
|  | r45 | r45a: 98.89 (75 steps); r45b: 98.78 (75 steps) | 0.11 | -0.26 ms | -0.3 % | gain |
|  | r49 | r49a: 99.14 (75 steps); r49b: 98.92 (75 steps) | 0.21 | -0.06 ms | -0.1 % | negligible |
| 3,584 | c | ca: 99.51 (87 steps); cb: 99.50 (87 steps) | 0.01 | control | | |
|  | r42 | r42a: 98.19 (74 steps); r42b: 98.47 (74 steps) | 0.28 | -1.18 ms | -1.2 % | gain |
|  | r45 | r45a: 98.11 (81 steps); r45b: 98.31 (81 steps) | 0.20 | -1.30 ms | -1.3 % | gain |
|  | r49 | r49a: 100.65 (87 steps); r49b: 99.54 (87 steps) | 1.12 | +0.59 ms | +0.6 % | negligible |
| 3,840 | c | ca: 99.94 (72 steps); cb: 100.02 (72 steps) | 0.08 | control | | |
|  | r42 | r42a: 98.98 (72 steps); r42b: 98.81 (72 steps) | 0.17 | -1.08 ms | -1.1 % | gain |
|  | r45 | r45a: 98.80 (72 steps); r45b: 99.52 (72 steps) | 0.72 | -0.82 ms | -0.8 % | gain |
|  | r49 | r49a: 98.77 (72 steps); r49b: 99.04 (72 steps) | 0.27 | -1.07 ms | -1.1 % | gain |

### Prompt reading, tok/s

Per load: the tokens read at that depth over their time (a prime request reads a chunk on top of the cache; without primes each decode request reads its prompt), with the tokens read and the tokens reused. Requests that read under 32 tokens are left out.

| Depth (tokens) | Arm | Per load | Spread | Against `c` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | c | ca: 58.7 (64 read on 0 cached); cb: 58.4 (64 read on 0 cached) | 0.3 | control | | |
|  | r42 | r42a: 55.6 (64 read on 0 cached); r42b: 55.5 (64 read on 0 cached) | 0.1 | -3.0 tok/s | -5.1 % | loss |
|  | r45 | r45a: 58.2 (64 read on 0 cached); r45b: 55.3 (64 read on 0 cached) | 3.0 | -1.8 tok/s | -3.1 % | negligible |
|  | r49 | r49a: 56.8 (64 read on 0 cached); r49b: 58.6 (64 read on 0 cached) | 1.8 | -0.8 tok/s | -1.4 % | negligible |
| 3,328 | c | ca: 135.8 (3264 read on 0 cached); cb: 137.6 (3264 read on 0 cached) | 1.8 | control | | |
|  | r42 | r42a: 136.9 (3264 read on 0 cached); r42b: 131.7 (3264 read on 0 cached) | 5.2 | -2.4 tok/s | -1.8 % | negligible |
|  | r45 | r45a: 135.1 (3264 read on 0 cached); r45b: 138.0 (3264 read on 0 cached) | 2.8 | -0.2 tok/s | -0.1 % | negligible |
|  | r49 | r49a: 135.9 (3264 read on 0 cached); r49b: 128.7 (3264 read on 0 cached) | 7.2 | -4.4 tok/s | -3.2 % | negligible |
| 3,584 | c | ca: 155.1 (772 read on 2748 cached); cb: 148.3 (772 read on 2748 cached) | 6.7 | control | | |
|  | r42 | r42a: 150.2 (772 read on 2748 cached); r42b: 154.4 (772 read on 2748 cached) | 4.1 | +0.6 tok/s | +0.4 % | negligible |
|  | r45 | r45a: 152.1 (772 read on 2748 cached); r45b: 155.0 (772 read on 2748 cached) | 2.9 | +1.9 tok/s | +1.2 % | negligible |
|  | r49 | r49a: 153.2 (772 read on 2748 cached); r49b: 155.3 (772 read on 2748 cached) | 2.1 | +2.6 tok/s | +1.7 % | negligible |
| 3,840 | c | ca: 184.1 (1028 read on 2748 cached); cb: 182.7 (1028 read on 2748 cached) | 1.3 | control | | |
|  | r42 | r42a: 183.6 (1028 read on 2748 cached); r42b: 184.3 (1028 read on 2748 cached) | 0.8 | +0.6 tok/s | +0.3 % | negligible |
|  | r45 | r45a: 181.0 (1028 read on 2748 cached); r45b: 181.4 (1028 read on 2748 cached) | 0.4 | -2.2 tok/s | -1.2 % | loss |
|  | r49 | r49a: 179.9 (1028 read on 2748 cached); r49b: 186.1 (1028 read on 2748 cached) | 6.2 | -0.4 tok/s | -0.2 % | negligible |

### The draft and the step

Per load, pass 1: draft tokens accepted of drafted (the share), tokens a step (generated tokens over steps, a step being one pass of the main model: generated minus accepted), and the time of a step. Pass 2 is listed where its counters differ.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | c | ca: 147 of 321 (0.458), 2.35 a step, 94.0 ms a step (pass 2: 148 of 318 (0.465), 2.37 a step, 90.4 ms a step); cb: 147 of 321 (0.458), 2.35 a step, 95.1 ms a step (pass 2: 148 of 318 (0.465), 2.37 a step, 90.6 ms a step) |
|  | r42 | r42a: 147 of 321 (0.458), 2.35 a step, 95.0 ms a step (pass 2: 148 of 318 (0.465), 2.37 a step, 90.4 ms a step); r42b: 147 of 321 (0.458), 2.35 a step, 95.0 ms a step (pass 2: 148 of 318 (0.465), 2.37 a step, 90.5 ms a step) |
|  | r45 | r45a: 147 of 321 (0.458), 2.35 a step, 94.9 ms a step (pass 2: 148 of 318 (0.465), 2.37 a step, 90.4 ms a step); r45b: 147 of 321 (0.458), 2.35 a step, 94.4 ms a step (pass 2: 148 of 318 (0.465), 2.37 a step, 90.5 ms a step) |
|  | r49 | r49a: 147 of 321 (0.458), 2.35 a step, 95.5 ms a step (pass 2: 148 of 318 (0.465), 2.37 a step, 90.6 ms a step); r49b: 147 of 321 (0.458), 2.35 a step, 95.0 ms a step (pass 2: 148 of 318 (0.465), 2.37 a step, 90.4 ms a step) |
| 3,328 | c | ca: 180 of 224 (0.804), 3.37 a step, 107.5 ms a step (pass 2: 181 of 221 (0.819), 3.41 a step, 99.2 ms a step); cb: 180 of 224 (0.804), 3.37 a step, 103.6 ms a step (pass 2: 181 of 221 (0.819), 3.41 a step, 99.0 ms a step) |
|  | r42 | r42a: 180 of 224 (0.804), 3.37 a step, 103.8 ms a step (pass 2: 181 of 221 (0.819), 3.41 a step, 98.8 ms a step); r42b: 180 of 224 (0.804), 3.37 a step, 105.8 ms a step (pass 2: 181 of 221 (0.819), 3.41 a step, 98.5 ms a step) |
|  | r45 | r45a: 180 of 224 (0.804), 3.37 a step, 103.7 ms a step (pass 2: 181 of 221 (0.819), 3.41 a step, 98.9 ms a step); r45b: 180 of 224 (0.804), 3.37 a step, 103.9 ms a step (pass 2: 181 of 221 (0.819), 3.41 a step, 98.8 ms a step) |
|  | r49 | r49a: 180 of 224 (0.804), 3.37 a step, 103.6 ms a step (pass 2: 181 of 221 (0.819), 3.41 a step, 99.1 ms a step); r49b: 180 of 224 (0.804), 3.37 a step, 103.8 ms a step (pass 2: 181 of 221 (0.819), 3.41 a step, 98.9 ms a step) |
| 3,584 | c | ca: 168 of 260 (0.646), 2.91 a step, 106.6 ms a step (pass 2: 169 of 257 (0.658), 2.94 a step, 99.5 ms a step); cb: 168 of 260 (0.646), 2.91 a step, 105.5 ms a step (pass 2: 169 of 257 (0.658), 2.94 a step, 99.5 ms a step) |
|  | r42 | r42a: 180 of 222 (0.811), 3.37 a step, 103.0 ms a step (pass 2: 182 of 216 (0.843), 3.46 a step, 98.2 ms a step); r42b: 180 of 222 (0.811), 3.37 a step, 103.0 ms a step (pass 2: 182 of 216 (0.843), 3.46 a step, 98.5 ms a step) |
|  | r45 | r45a: 174 of 240 (0.725), 3.12 a step, 102.8 ms a step (pass 2: 175 of 237 (0.738), 3.16 a step, 98.1 ms a step); r45b: 174 of 240 (0.725), 3.12 a step, 102.8 ms a step (pass 2: 175 of 237 (0.738), 3.16 a step, 98.3 ms a step) |
|  | r49 | r49a: 168 of 260 (0.646), 2.91 a step, 105.4 ms a step (pass 2: 169 of 257 (0.658), 2.94 a step, 100.7 ms a step); r49b: 168 of 260 (0.646), 2.91 a step, 105.7 ms a step (pass 2: 169 of 257 (0.658), 2.94 a step, 99.5 ms a step) |
| 3,840 | c | ca: 184 of 212 (0.868), 3.56 a step, 103.9 ms a step; cb: 184 of 212 (0.868), 3.56 a step, 103.7 ms a step |
|  | r42 | r42a: 184 of 212 (0.868), 3.56 a step, 102.6 ms a step; r42b: 184 of 212 (0.868), 3.56 a step, 102.5 ms a step |
|  | r45 | r45a: 184 of 212 (0.868), 3.56 a step, 103.0 ms a step; r45b: 184 of 212 (0.868), 3.56 a step, 102.8 ms a step |
|  | r49 | r49a: 184 of 212 (0.868), 3.56 a step, 103.0 ms a step; r49b: 184 of 212 (0.868), 3.56 a step, 102.3 ms a step |

### The sampler pass

One request per depth and load with the server line's own sampler and a fixed seed: tok/s, the draft counters and the step.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | c | ca: 28.28 tok/s, 159 of 285 (0.558), 2.64 a step, 93.3 ms a step; cb: 28.31 tok/s, 159 of 285 (0.558), 2.64 a step, 93.2 ms a step |
|  | r42 | r42a: 28.37 tok/s, 159 of 285 (0.558), 2.64 a step, 93.0 ms a step; r42b: 28.29 tok/s, 159 of 285 (0.558), 2.64 a step, 93.3 ms a step |
|  | r45 | r45a: 28.26 tok/s, 159 of 285 (0.558), 2.64 a step, 93.4 ms a step; r45b: 28.37 tok/s, 159 of 285 (0.558), 2.64 a step, 93.0 ms a step |
|  | r49 | r49a: 28.30 tok/s, 159 of 285 (0.558), 2.64 a step, 93.2 ms a step; r49b: 25.75 tok/s, 159 of 285 (0.558), 2.64 a step, 102.5 ms a step |
| 3,328 | c | ca: 32.31 tok/s, 177 of 233 (0.760), 3.24 a step, 100.3 ms a step; cb: 32.47 tok/s, 177 of 233 (0.760), 3.24 a step, 99.8 ms a step |
|  | r42 | r42a: 34.24 tok/s, 182 of 216 (0.843), 3.46 a step, 101.0 ms a step; r42b: 30.25 tok/s, 182 of 216 (0.843), 3.46 a step, 114.4 ms a step |
|  | r45 | r45a: 32.52 tok/s, 177 of 233 (0.760), 3.24 a step, 99.7 ms a step; r45b: 32.48 tok/s, 177 of 233 (0.760), 3.24 a step, 99.8 ms a step |
|  | r49 | r49a: 31.52 tok/s, 175 of 240 (0.729), 3.16 a step, 100.3 ms a step; r49b: 30.79 tok/s, 177 of 233 (0.760), 3.24 a step, 105.2 ms a step |
| 3,584 | c | ca: 31.26 tok/s, 176 of 235 (0.749), 3.20 a step, 102.4 ms a step; cb: 31.39 tok/s, 176 of 235 (0.749), 3.20 a step, 101.9 ms a step |
|  | r42 | r42a: 36.32 tok/s, 185 of 207 (0.894), 3.61 a step, 99.3 ms a step; r42b: 36.28 tok/s, 185 of 207 (0.894), 3.61 a step, 99.4 ms a step |
|  | r45 | r45a: 35.38 tok/s, 183 of 213 (0.859), 3.51 a step, 99.1 ms a step; r45b: 35.13 tok/s, 183 of 213 (0.859), 3.51 a step, 99.8 ms a step |
|  | r49 | r49a: 30.25 tok/s, 173 of 244 (0.709), 3.08 a step, 101.9 ms a step; r49b: 31.40 tok/s, 176 of 235 (0.749), 3.20 a step, 101.9 ms a step |
| 3,840 | c | ca: 32.39 tok/s, 180 of 222 (0.811), 3.37 a step, 104.0 ms a step; cb: 32.91 tok/s, 180 of 222 (0.811), 3.37 a step, 102.4 ms a step |
|  | r42 | r42a: 32.85 tok/s, 179 of 226 (0.792), 3.32 a step, 101.2 ms a step; r42b: 32.93 tok/s, 179 of 226 (0.792), 3.32 a step, 101.0 ms a step |
|  | r45 | r45a: 32.81 tok/s, 179 of 226 (0.792), 3.32 a step, 101.3 ms a step; r45b: 32.75 tok/s, 179 of 226 (0.792), 3.32 a step, 101.5 ms a step |
|  | r49 | r49a: 34.17 tok/s, 182 of 216 (0.843), 3.46 a step, 101.2 ms a step; r49b: 34.19 tok/s, 182 of 216 (0.843), 3.46 a step, 101.2 ms a step |

### Checks

- Times: every load started on a machine checked quiet (valid=1).
- Prompts: 64 ids at depth 0, 3328 ids at depth 3,328, 3584 ids at depth 3,584, 3840 ids at depth 3,840; the same ids on every load and pass.
- Tokens out: 256 on every timed request.
- Greedy reply at depth 0: the same on every pass and load of every arm.
- Greedy reply at depth 3,328: **differs from `c` in r42.**
- Greedy reply at depth 3,584: **differs from `c` in r42, r45.**
- Greedy reply at depth 3,840: the same on every pass and load of every arm.
- Primes (read on cached, the cache's source): 3,328: 3264 on 0 (none); 3,584: 772 on 2748 (checkpoint); 3,840: 1028 on 2748 (checkpoint).
- Graph reallocations of the target, requests with one per load: ca: 0 of 16 requests; r42a: 0 of 16 requests; r45a: 0 of 16 requests; r49a: 0 of 16 requests; r49b: 0 of 16 requests; r45b: 0 of 16 requests; r42b: 0 of 16 requests; cb: 0 of 16 requests.
