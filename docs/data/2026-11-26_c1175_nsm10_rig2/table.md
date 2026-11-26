## The deep lane: c, n10, r05

Model file `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf`, llama-server version: 1.0.0-dev (build 13218, commit eee615cdb9).

256 tokens per request at temperature 0, each request twice per load (pass 1 is the first request at a new depth, pass 2 the same request again). Timed loads, in their order: ca, n10a, r05a, r05b, n10b, cb.

| Arm | Loads | Draft | Context | Its own variables and flags |
|---|---|---|---:|---|
| c (control) | ca, cb | on | 65536 | `GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=5 GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=0.75` |
| n10 | n10a, n10b | on | 65536 | `GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=10 GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=0.75` |
| r05 | r05a, r05b | on | 65536 | `GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=10 GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=0.5` |

### Generation, tok/s

Per load: pass 1, pass 2 → the load, with the load's time per token. A load's speed is its two passes' tokens over their time. Spread: the difference between the arm's fastest and slowest load. Class against `c`: gain when the mean is above the control's by more than the larger of the two spreads, loss when below by more than it, else negligible.

| Depth (tokens) | Arm | Per load | Spread | Against `c` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | c | ca: 22.78, 26.25 → **24.39** (40.99 ms a token); cb: 24.74, 26.25 → **25.47** (39.25 ms a token) | 1.08 | control | | |
|  | n10 | n10a: 24.70, 26.21 → **25.43** (39.32 ms a token); n10b: 22.80, 26.26 → **24.41** (40.97 ms a token) | 1.03 | -0.01 tok/s | -0.1 % | negligible |
|  | r05 | r05a: 24.79, 26.28 → **25.51** (39.20 ms a token); r05b: 24.90, 26.28 → **25.57** (39.10 ms a token) | 0.06 | +0.61 tok/s | +2.4 % | negligible |
| 4,096 | c | ca: 31.07, 32.48 → **31.76** (31.49 ms a token); cb: 31.08, 32.54 → **31.79** (31.45 ms a token) | 0.03 | control | | |
|  | n10 | n10a: 31.17, 32.51 → **31.82** (31.43 ms a token); n10b: 30.96, 32.44 → **31.68** (31.57 ms a token) | 0.14 | -0.02 tok/s | -0.1 % | negligible |
|  | r05 | r05a: 30.79, 32.17 → **31.46** (31.78 ms a token); r05b: 30.82, 32.15 → **31.47** (31.78 ms a token) | 0.01 | -0.31 tok/s | -1.0 % | loss |
| 6,144 | c | ca: 33.93, 36.08 → **34.97** (28.59 ms a token); cb: 34.03, 36.05 → **35.01** (28.56 ms a token) | 0.04 | control | | |
|  | n10 | n10a: 34.63, 37.51 → **36.01** (27.77 ms a token); n10b: 35.51, 37.46 → **36.46** (27.43 ms a token) | 0.45 | +1.24 tok/s | +3.6 % | gain |
|  | r05 | r05a: 35.35, 37.55 → **36.42** (27.46 ms a token); r05b: 35.32, 37.51 → **36.38** (27.48 ms a token) | 0.03 | +1.41 tok/s | +4.0 % | gain |
| 8,192 | c | ca: 27.56, 31.56 → **29.42** (33.99 ms a token); cb: 27.81, 31.59 → **29.58** (33.80 ms a token) | 0.16 | control | | |
|  | n10 | n10a: 35.22, 36.82 → **36.00** (27.77 ms a token); n10b: 35.42, 36.77 → **36.08** (27.71 ms a token) | 0.08 | +6.54 tok/s | +22.2 % | gain |
|  | r05 | r05a: 35.22, 36.79 → **35.99** (27.79 ms a token); r05b: 35.19, 36.77 → **35.96** (27.81 ms a token) | 0.03 | +6.47 tok/s | +21.9 % | gain |

### Generation over all depths, tok/s

Every depth's tokens over their time, per load.

| Depth (tokens) | Arm | Per load | Spread | Against `c` | | Class |
|---:|---|---|---:|---:|---:|---|
| all | c | ca: **29.62**; cb: **30.06** | 0.44 | control | | |
|  | n10 | n10a: **31.67**; n10b: **31.33** | 0.35 | +1.66 tok/s | +5.6 % | gain |
|  | r05 | r05a: **31.69**; r05b: **31.70** | 0.01 | +1.86 tok/s | +6.2 % | gain |

### The step on pass 1 (the first request at a new depth), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `c`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `c` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | c | ca: 103.08 (109 steps); cb: 94.93 (109 steps) | 8.15 | control | | |
|  | n10 | n10a: 95.07 (109 steps); n10b: 102.99 (109 steps) | 7.92 | +0.03 ms | +0.0 % | negligible |
|  | r05 | r05a: 94.75 (109 steps); r05b: 94.30 (109 steps) | 0.44 | -4.48 ms | -4.5 % | negligible |
| 4,096 | c | ca: 104.31 (79 steps); cb: 104.26 (79 steps) | 0.05 | control | | |
|  | n10 | n10a: 103.98 (79 steps); n10b: 104.68 (79 steps) | 0.70 | +0.04 ms | +0.0 % | negligible |
|  | r05 | r05a: 102.66 (81 steps); r05b: 102.54 (81 steps) | 0.12 | -1.69 ms | -1.6 % | gain |
| 6,144 | c | ca: 107.78 (70 steps); cb: 107.48 (70 steps) | 0.29 | control | | |
|  | n10 | n10a: 105.59 (70 steps); n10b: 102.98 (70 steps) | 2.61 | -3.34 ms | -3.1 % | gain |
|  | r05 | r05a: 103.46 (70 steps); r05b: 103.54 (70 steps) | 0.08 | -4.13 ms | -3.8 % | gain |
| 8,192 | c | ca: 111.92 (83 steps); cb: 110.89 (83 steps) | 1.02 | control | | |
|  | n10 | n10a: 102.37 (71 steps); n10b: 101.80 (71 steps) | 0.57 | -9.32 ms | -8.4 % | gain |
|  | r05 | r05a: 102.38 (71 steps); r05b: 102.45 (71 steps) | 0.07 | -8.99 ms | -8.1 % | gain |

### The step on pass 2 (the same request again), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `c`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `c` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | c | ca: 90.31 (108 steps); cb: 90.29 (108 steps) | 0.02 | control | | |
|  | n10 | n10a: 90.44 (108 steps); n10b: 90.28 (108 steps) | 0.16 | +0.06 ms | +0.1 % | negligible |
|  | r05 | r05a: 90.19 (108 steps); r05b: 90.20 (108 steps) | 0.00 | -0.10 ms | -0.1 % | gain |
| 4,096 | c | ca: 99.75 (79 steps); cb: 99.60 (79 steps) | 0.15 | control | | |
|  | n10 | n10a: 99.69 (79 steps); n10b: 99.89 (79 steps) | 0.21 | +0.11 ms | +0.1 % | negligible |
|  | r05 | r05a: 98.23 (81 steps); r05b: 98.32 (81 steps) | 0.08 | -1.40 ms | -1.4 % | gain |
| 6,144 | c | ca: 102.83 (69 steps); cb: 102.91 (69 steps) | 0.08 | control | | |
|  | n10 | n10a: 98.92 (69 steps); n10b: 99.05 (69 steps) | 0.13 | -3.88 ms | -3.8 % | gain |
|  | r05 | r05a: 98.80 (69 steps); r05b: 98.90 (69 steps) | 0.11 | -4.02 ms | -3.9 % | gain |
| 8,192 | c | ca: 103.99 (78 steps); cb: 103.89 (78 steps) | 0.10 | control | | |
|  | n10 | n10a: 99.32 (70 steps); n10b: 99.46 (70 steps) | 0.14 | -4.55 ms | -4.4 % | gain |
|  | r05 | r05a: 99.39 (70 steps); r05b: 99.47 (70 steps) | 0.07 | -4.51 ms | -4.3 % | gain |

### Prompt reading, tok/s

Per load: the tokens read at that depth over their time (a prime request reads a chunk on top of the cache; without primes each decode request reads its prompt), with the tokens read and the tokens reused. Requests that read under 32 tokens are left out.

| Depth (tokens) | Arm | Per load | Spread | Against `c` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | c | ca: 50.9 (64 read on 0 cached); cb: 55.2 (64 read on 0 cached) | 4.4 | control | | |
|  | n10 | n10a: 57.9 (64 read on 0 cached); n10b: 58.4 (64 read on 0 cached) | 0.4 | +5.1 tok/s | +9.7 % | gain |
|  | r05 | r05a: 58.9 (64 read on 0 cached); r05b: 57.9 (64 read on 0 cached) | 1.0 | +5.3 tok/s | +10.0 % | gain |
| 4,096 | c | ca: 205.2 (4032 read on 0 cached); cb: 197.2 (4032 read on 0 cached) | 8.0 | control | | |
|  | n10 | n10a: 208.4 (4032 read on 0 cached); n10b: 196.9 (4032 read on 0 cached) | 11.5 | +1.5 tok/s | +0.7 % | negligible |
|  | r05 | r05a: 186.0 (4032 read on 0 cached); r05b: 207.7 (4032 read on 0 cached) | 21.7 | -4.3 tok/s | -2.1 % | negligible |
| 6,144 | c | ca: 278.0 (2564 read on 3516 cached); cb: 271.8 (2564 read on 3516 cached) | 6.1 | control | | |
|  | n10 | n10a: 280.8 (2564 read on 3516 cached); n10b: 281.4 (2564 read on 3516 cached) | 0.6 | +6.2 tok/s | +2.3 % | gain |
|  | r05 | r05a: 218.6 (2564 read on 3516 cached); r05b: 271.5 (2564 read on 3516 cached) | 53.0 | -29.9 tok/s | -10.9 % | negligible |
| 8,192 | c | ca: 234.2 (4612 read on 3516 cached); cb: 237.5 (4612 read on 3516 cached) | 3.3 | control | | |
|  | n10 | n10a: 233.1 (4612 read on 3516 cached); n10b: 241.6 (4612 read on 3516 cached) | 8.5 | +1.5 tok/s | +0.6 % | negligible |
|  | r05 | r05a: 226.2 (4612 read on 3516 cached); r05b: 231.3 (4612 read on 3516 cached) | 5.2 | -7.1 tok/s | -3.0 % | loss |

### The draft and the step

Per load, pass 1: draft tokens accepted of drafted (the share), tokens a step (generated tokens over steps, a step being one pass of the main model: generated minus accepted), and the time of a step. Pass 2 is listed where its counters differ.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | c | ca: 147 of 321 (0.458), 2.35 a step, 103.1 ms a step (pass 2: 148 of 318 (0.465), 2.37 a step, 90.3 ms a step); cb: 147 of 321 (0.458), 2.35 a step, 94.9 ms a step (pass 2: 148 of 318 (0.465), 2.37 a step, 90.3 ms a step) |
|  | n10 | n10a: 147 of 321 (0.458), 2.35 a step, 95.1 ms a step (pass 2: 148 of 318 (0.465), 2.37 a step, 90.4 ms a step); n10b: 147 of 321 (0.458), 2.35 a step, 103.0 ms a step (pass 2: 148 of 318 (0.465), 2.37 a step, 90.3 ms a step) |
|  | r05 | r05a: 147 of 321 (0.458), 2.35 a step, 94.7 ms a step (pass 2: 148 of 318 (0.465), 2.37 a step, 90.2 ms a step); r05b: 147 of 321 (0.458), 2.35 a step, 94.3 ms a step (pass 2: 148 of 318 (0.465), 2.37 a step, 90.2 ms a step) |
| 4,096 | c | ca: 177 of 231 (0.766), 3.24 a step, 104.3 ms a step; cb: 177 of 231 (0.766), 3.24 a step, 104.3 ms a step |
|  | n10 | n10a: 177 of 231 (0.766), 3.24 a step, 104.0 ms a step; n10b: 177 of 231 (0.766), 3.24 a step, 104.7 ms a step |
|  | r05 | r05a: 175 of 237 (0.738), 3.16 a step, 102.7 ms a step; r05b: 175 of 237 (0.738), 3.16 a step, 102.5 ms a step |
| 6,144 | c | ca: 186 of 205 (0.907), 3.66 a step, 107.8 ms a step (pass 2: 187 of 202 (0.926), 3.71 a step, 102.8 ms a step); cb: 186 of 205 (0.907), 3.66 a step, 107.5 ms a step (pass 2: 187 of 202 (0.926), 3.71 a step, 102.9 ms a step) |
|  | n10 | n10a: 186 of 205 (0.907), 3.66 a step, 105.6 ms a step (pass 2: 187 of 202 (0.926), 3.71 a step, 98.9 ms a step); n10b: 186 of 205 (0.907), 3.66 a step, 103.0 ms a step (pass 2: 187 of 202 (0.926), 3.71 a step, 99.1 ms a step) |
|  | r05 | r05a: 186 of 205 (0.907), 3.66 a step, 103.5 ms a step (pass 2: 187 of 202 (0.926), 3.71 a step, 98.8 ms a step); r05b: 186 of 205 (0.907), 3.66 a step, 103.5 ms a step (pass 2: 187 of 202 (0.926), 3.71 a step, 98.9 ms a step) |
| 8,192 | c | ca: 173 of 245 (0.706), 3.08 a step, 111.9 ms a step (pass 2: 178 of 230 (0.774), 3.28 a step, 104.0 ms a step); cb: 173 of 245 (0.706), 3.08 a step, 110.9 ms a step (pass 2: 178 of 230 (0.774), 3.28 a step, 103.9 ms a step) |
|  | n10 | n10a: 185 of 210 (0.881), 3.61 a step, 102.4 ms a step (pass 2: 186 of 207 (0.899), 3.66 a step, 99.3 ms a step); n10b: 185 of 210 (0.881), 3.61 a step, 101.8 ms a step (pass 2: 186 of 207 (0.899), 3.66 a step, 99.5 ms a step) |
|  | r05 | r05a: 185 of 210 (0.881), 3.61 a step, 102.4 ms a step (pass 2: 186 of 207 (0.899), 3.66 a step, 99.4 ms a step); r05b: 185 of 210 (0.881), 3.61 a step, 102.5 ms a step (pass 2: 186 of 207 (0.899), 3.66 a step, 99.5 ms a step) |

### The sampler pass

One request per depth and load with the server line's own sampler and a fixed seed: tok/s, the draft counters and the step.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | c | ca: 25.69 tok/s, 159 of 285 (0.558), 2.64 a step, 102.7 ms a step; cb: 28.49 tok/s, 159 of 285 (0.558), 2.64 a step, 92.6 ms a step |
|  | n10 | n10a: 28.41 tok/s, 159 of 285 (0.558), 2.64 a step, 92.9 ms a step; n10b: 25.48 tok/s, 159 of 285 (0.558), 2.64 a step, 103.6 ms a step |
|  | r05 | r05a: 28.41 tok/s, 159 of 285 (0.558), 2.64 a step, 92.9 ms a step; r05b: 28.45 tok/s, 159 of 285 (0.558), 2.64 a step, 92.8 ms a step |
| 4,096 | c | ca: 31.43 tok/s, 177 of 232 (0.763), 3.24 a step, 103.1 ms a step; cb: 28.17 tok/s, 177 of 232 (0.763), 3.24 a step, 115.0 ms a step |
|  | n10 | n10a: 31.42 tok/s, 177 of 232 (0.763), 3.24 a step, 103.1 ms a step; n10b: 31.45 tok/s, 177 of 232 (0.763), 3.24 a step, 103.0 ms a step |
|  | r05 | r05a: 32.47 tok/s, 178 of 229 (0.777), 3.28 a step, 101.1 ms a step; r05b: 32.29 tok/s, 178 of 229 (0.777), 3.28 a step, 101.6 ms a step |
| 6,144 | c | ca: 31.77 tok/s, 179 of 226 (0.792), 3.32 a step, 104.6 ms a step; cb: 31.90 tok/s, 179 of 226 (0.792), 3.32 a step, 104.2 ms a step |
|  | n10 | n10a: 33.40 tok/s, 181 of 219 (0.826), 3.41 a step, 102.2 ms a step; n10b: 33.42 tok/s, 181 of 219 (0.826), 3.41 a step, 102.1 ms a step |
|  | r05 | r05a: 36.05 tok/s, 185 of 210 (0.881), 3.61 a step, 100.0 ms a step; r05b: 35.78 tok/s, 185 of 210 (0.881), 3.61 a step, 100.8 ms a step |
| 8,192 | c | ca: 35.02 tok/s, 187 of 202 (0.926), 3.71 a step, 105.9 ms a step; cb: 35.09 tok/s, 187 of 202 (0.926), 3.71 a step, 105.7 ms a step |
|  | n10 | n10a: 31.63 tok/s, 176 of 237 (0.743), 3.20 a step, 101.2 ms a step; n10b: 31.71 tok/s, 176 of 237 (0.743), 3.20 a step, 100.9 ms a step |
|  | r05 | r05a: 31.71 tok/s, 176 of 237 (0.743), 3.20 a step, 100.9 ms a step; r05b: 31.65 tok/s, 176 of 237 (0.743), 3.20 a step, 101.1 ms a step |

### Checks

- Times: every load started on a machine checked quiet (valid=1).
- Prompts: 64 ids at depth 0, 4096 ids at depth 4,096, 6144 ids at depth 6,144, 8192 ids at depth 8,192; the same ids on every load and pass.
- Tokens out: 256 on every timed request.
- Greedy reply at depth 0: the same on every pass and load of every arm.
- Greedy reply at depth 4,096: **differs from `c` in r05.**
- Greedy reply at depth 6,144: the same on every pass and load of every arm.
- Greedy reply at depth 8,192: **differs from `c` in n10, r05.**
- Primes (read on cached, the cache's source): 4,096: 4032 on 0 (none); 6,144: 2564 on 3516 (checkpoint); 8,192: 4612 on 3516 (checkpoint).
- Graph reallocations of the target, requests with one per load: ca: 0 of 16 requests; n10a: 0 of 16 requests; r05a: 0 of 16 requests; r05b: 0 of 16 requests; n10b: 0 of 16 requests; cb: 0 of 16 requests.
