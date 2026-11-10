## The deep lane: dense, sparse

Model file `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf`, llama-server version: 0.8.0-dev (build 12358, commit 1551f5fe1).

256 tokens per request at temperature 0, each request twice per load (pass 1 is the first request at a new depth, pass 2 the same request again). Timed loads, in their order: dense1, sparse1, dense2, sparse2.

| Arm | Loads | Draft | Context | Its own variables and flags |
|---|---|---|---:|---|
| dense (control) | dense1, dense2 | on | 65536 | `GGML_CUDA_FATTN_SPARSE_PASCAL=0` |
| sparse | sparse1, sparse2 | on | 65536 | `GGML_CUDA_FATTN_SPARSE_PASCAL=1` |

### Generation, tok/s

Per load: pass 1, pass 2 → the load, with the load's time per token. A load's speed is its two passes' tokens over their time. Spread: the difference between the arm's fastest and slowest load. Class against `dense`: gain when the mean is above the control's by more than the larger of the two spreads, loss when below by more than it, else negligible.

| Depth (tokens) | Arm | Per load | Spread | Against `dense` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | dense | dense1: 26.38, 27.70 → **27.03** (37.00 ms a token); dense2: 25.77, 27.70 → **26.70** (37.46 ms a token) | 0.33 | control | | |
|  | sparse | sparse1: 25.81, 27.67 → **26.71** (37.44 ms a token); sparse2: 25.78, 27.67 → **26.69** (37.47 ms a token) | 0.02 | -0.16 tok/s | -0.6 % | negligible |
| 8,192 | dense | dense1: 14.69, 19.68 → **16.82** (59.44 ms a token); dense2: 16.63, 24.98 → **19.97** (50.08 ms a token) | 3.14 | control | | |
|  | sparse | sparse1: 17.29, 25.40 → **20.58** (48.60 ms a token); sparse2: 17.50, 25.42 → **20.73** (48.25 ms a token) | 0.15 | +2.25 tok/s | +12.3 % | negligible |
| 16,384 | dense | dense1: 15.33, 19.51 → **17.17** (58.25 ms a token); dense2: 17.45, 19.70 → **18.51** (54.03 ms a token) | 1.34 | control | | |
|  | sparse | sparse1: 18.19, 23.75 → **20.60** (48.54 ms a token); sparse2: 18.51, 24.25 → **20.99** (47.64 ms a token) | 0.39 | +2.96 tok/s | +16.6 % | gain |
| 32,768 | dense | dense1: 9.26, 10.39 → **9.79** (102.12 ms a token); dense2: 10.44, 11.35 → **10.88** (91.94 ms a token) | 1.08 | control | | |
|  | sparse | sparse1: 14.79, 19.29 → **16.75** (59.71 ms a token); sparse2: 17.07, 19.31 → **18.12** (55.18 ms a token) | 1.38 | +7.10 tok/s | +68.7 % | gain |
| 49,152 | dense | dense1: 10.45, 10.73 → **10.59** (94.45 ms a token); dense2: 11.15, 11.96 → **11.54** (86.64 ms a token) | 0.95 | control | | |
|  | sparse | sparse1: 17.06, 18.63 → **17.81** (56.13 ms a token); sparse2: 13.91, 15.55 → **14.68** (68.11 ms a token) | 3.13 | +5.18 tok/s | +46.8 % | gain |

### Generation over all depths, tok/s

Every depth's tokens over their time, per load.

| Depth (tokens) | Arm | Per load | Spread | Against `dense` | | Class |
|---:|---|---|---:|---:|---:|---|
| all | dense | dense1: **14.23**; dense2: **15.62** | 1.38 | control | | |
|  | sparse | sparse1: **19.97**; sparse2: **19.48** | 0.48 | +4.80 tok/s | +32.1 % | gain |

### The step on pass 1 (the first request at a new depth), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `dense`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `dense` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | dense | dense1: 124.41 (78 steps); dense2: 127.38 (78 steps) | 2.97 | control | | |
|  | sparse | sparse1: 127.14 (78 steps); sparse2: 127.32 (78 steps) | 0.18 | +1.34 ms | +1.1 % | negligible |
| 8,192 | dense | dense1: 193.62 (90 steps); dense2: 229.73 (67 steps) | 36.10 | control | | |
|  | sparse | sparse1: 221.00 (67 steps); sparse2: 218.37 (67 steps) | 2.63 | +8.01 ms | +3.8 % | negligible |
| 16,384 | dense | dense1: 228.81 (73 steps); dense2: 206.68 (71 steps) | 22.14 | control | | |
|  | sparse | sparse1: 195.46 (72 steps); sparse2: 194.84 (71 steps) | 0.62 | -22.60 ms | -10.4 % | gain |
| 32,768 | dense | dense1: 268.32 (103 steps); dense2: 258.10 (95 steps) | 10.22 | control | | |
|  | sparse | sparse1: 216.30 (80 steps); sparse2: 185.10 (81 steps) | 31.20 | -62.51 ms | -23.7 % | gain |
| 49,152 | dense | dense1: 306.33 (80 steps); dense2: 323.28 (71 steps) | 16.95 | control | | |
|  | sparse | sparse1: 197.40 (76 steps); sparse2: 202.27 (91 steps) | 4.87 | -114.96 ms | -36.5 % | gain |

### The step on pass 2 (the same request again), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `dense`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `dense` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | dense | dense1: 123.21 (75 steps); dense2: 123.23 (75 steps) | 0.02 | control | | |
|  | sparse | sparse1: 123.36 (75 steps); sparse2: 123.38 (75 steps) | 0.02 | +0.15 ms | +0.1 % | loss |
| 8,192 | dense | dense1: 153.00 (85 steps); dense2: 152.97 (67 steps) | 0.03 | control | | |
|  | sparse | sparse1: 150.40 (67 steps); sparse2: 150.34 (67 steps) | 0.06 | -2.62 ms | -1.7 % | gain |
| 16,384 | dense | dense1: 182.25 (72 steps); dense2: 182.98 (71 steps) | 0.74 | control | | |
|  | sparse | sparse1: 149.73 (72 steps); sparse2: 150.80 (70 steps) | 1.07 | -32.35 ms | -17.7 % | gain |
| 32,768 | dense | dense1: 241.65 (102 steps); dense2: 242.49 (93 steps) | 0.84 | control | | |
|  | sparse | sparse1: 165.86 (80 steps); sparse2: 165.74 (80 steps) | 0.13 | -76.27 ms | -31.5 % | gain |
| 49,152 | dense | dense1: 301.90 (79 steps); dense2: 301.48 (71 steps) | 0.42 | control | | |
|  | sparse | sparse1: 180.77 (76 steps); sparse2: 180.96 (91 steps) | 0.19 | -120.82 ms | -40.0 % | gain |

### Prompt reading, tok/s

Per load: the tokens read at that depth over their time (a prime request reads a chunk on top of the cache; without primes each decode request reads its prompt), with the tokens read and the tokens reused. Requests that read under 32 tokens are left out.

| Depth (tokens) | Arm | Per load | Spread | Against `dense` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | dense | dense1: 32.5 (64 read on 0 cached); dense2: 32.2 (64 read on 0 cached) | 0.3 | control | | |
|  | sparse | sparse1: 32.1 (64 read on 0 cached); sparse2: 32.4 (64 read on 0 cached) | 0.3 | -0.1 tok/s | -0.2 % | negligible |
| 8,192 | dense | dense1: 129.7 (8128 read on 0 cached); dense2: 127.4 (8128 read on 0 cached) | 2.3 | control | | |
|  | sparse | sparse1: 128.7 (8128 read on 0 cached); sparse2: 130.1 (8128 read on 0 cached) | 1.4 | +0.9 tok/s | +0.7 % | negligible |
| 16,384 | dense | dense1: 109.8 (8708 read on 7612 cached); dense2: 110.2 (8708 read on 7612 cached) | 0.4 | control | | |
|  | sparse | sparse1: 109.6 (8708 read on 7612 cached); sparse2: 109.6 (8708 read on 7612 cached) | 0.1 | -0.4 tok/s | -0.4 % | negligible |
| 32,768 | dense | dense1: 91.2 (16388 read on 16316 cached); dense2: 91.2 (16388 read on 16316 cached) | 0.0 | control | | |
|  | sparse | sparse1: 90.6 (16388 read on 16316 cached); sparse2: 90.5 (16388 read on 16316 cached) | 0.1 | -0.7 tok/s | -0.8 % | loss |
| 49,152 | dense | dense1: 70.5 (16900 read on 32188 cached); dense2: 70.3 (16900 read on 32188 cached) | 0.2 | control | | |
|  | sparse | sparse1: 70.4 (16900 read on 32188 cached); sparse2: 70.2 (16900 read on 32188 cached) | 0.3 | -0.1 tok/s | -0.2 % | negligible |

### The draft and the step

Per load, pass 1: draft tokens accepted of drafted (the share), tokens a step (generated tokens over steps, a step being one pass of the main model: generated minus accepted), and the time of a step. Pass 2 is listed where its counters differ.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | dense | dense1: 178 of 229 (0.777), 3.28 a step, 124.4 ms a step (pass 2: 181 of 220 (0.823), 3.41 a step, 123.2 ms a step); dense2: 178 of 229 (0.777), 3.28 a step, 127.4 ms a step (pass 2: 181 of 220 (0.823), 3.41 a step, 123.2 ms a step) |
|  | sparse | sparse1: 178 of 229 (0.777), 3.28 a step, 127.1 ms a step (pass 2: 181 of 220 (0.823), 3.41 a step, 123.4 ms a step); sparse2: 178 of 229 (0.777), 3.28 a step, 127.3 ms a step (pass 2: 181 of 220 (0.823), 3.41 a step, 123.4 ms a step) |
| 8,192 | dense | dense1: 166 of 265 (0.626), 2.84 a step, 193.6 ms a step (pass 2: 171 of 250 (0.684), 3.01 a step, 153.0 ms a step); dense2: 189 of 196 (0.964), 3.82 a step, 229.7 ms a step |
|  | sparse | sparse1: 189 of 196 (0.964), 3.82 a step, 221.0 ms a step; sparse2: 189 of 196 (0.964), 3.82 a step, 218.4 ms a step |
| 16,384 | dense | dense1: 183 of 214 (0.855), 3.51 a step, 228.8 ms a step (pass 2: 184 of 211 (0.872), 3.56 a step, 182.2 ms a step); dense2: 185 of 210 (0.881), 3.61 a step, 206.7 ms a step |
|  | sparse | sparse1: 184 of 210 (0.876), 3.56 a step, 195.5 ms a step; sparse2: 185 of 210 (0.881), 3.61 a step, 194.8 ms a step (pass 2: 186 of 207 (0.899), 3.66 a step, 150.8 ms a step) |
| 32,768 | dense | dense1: 153 of 301 (0.508), 2.49 a step, 268.3 ms a step (pass 2: 154 of 298 (0.517), 2.51 a step, 241.7 ms a step); dense2: 161 of 280 (0.575), 2.69 a step, 258.1 ms a step (pass 2: 163 of 274 (0.595), 2.75 a step, 242.5 ms a step) |
|  | sparse | sparse1: 176 of 236 (0.746), 3.20 a step, 216.3 ms a step; sparse2: 175 of 239 (0.732), 3.16 a step, 185.1 ms a step (pass 2: 176 of 236 (0.746), 3.20 a step, 165.7 ms a step) |
| 49,152 | dense | dense1: 176 of 237 (0.743), 3.20 a step, 306.3 ms a step (pass 2: 177 of 234 (0.756), 3.24 a step, 301.9 ms a step); dense2: 185 of 209 (0.885), 3.61 a step, 323.3 ms a step |
|  | sparse | sparse1: 180 of 223 (0.807), 3.37 a step, 197.4 ms a step; sparse2: 165 of 268 (0.616), 2.81 a step, 202.3 ms a step |

### Checks

- Times: every load started on a machine checked quiet (valid=1).
- Prompts: 64 ids at depth 0, 8192 ids at depth 8,192, 16384 ids at depth 16,384, 32768 ids at depth 32,768, 49152 ids at depth 49,152; the same ids on every load and pass.
- Tokens out: 256 on every timed request.
- Greedy reply at depth 0: the same on every pass and load of every arm.
- Greedy reply at depth 8,192: 3 different replies over the 4 loads (`dense`: 2 over its 2; `sparse`: 1 over its 2); every load wrote its own reply again on pass 2.
- Greedy reply at depth 16,384: 4 different replies over the 4 loads (`dense`: 2 over its 2; `sparse`: 2 over its 2); every load wrote its own reply again on pass 2.
- Greedy reply at depth 32,768: 4 different replies over the 4 loads (`dense`: 2 over its 2; `sparse`: 2 over its 2); every load wrote its own reply again on pass 2.
- Greedy reply at depth 49,152: 4 different replies over the 4 loads (`dense`: 2 over its 2; `sparse`: 2 over its 2); every load wrote its own reply again on pass 2.
- Primes (read on cached, the cache's source): 8,192: 8128 on 0 (none); 16,384: 8708 on 7612 (checkpoint); 32,768: 16388 on 16316 (checkpoint); 49,152: 16900 on 32188 (checkpoint).
