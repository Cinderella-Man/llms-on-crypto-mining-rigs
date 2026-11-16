## The deep lane: v4r, v4nk, v4ne

Model file `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf`, llama-server version: 1.0.0-dev (build 12832, commit eb312b7bce).

256 tokens per request at temperature 0, each request twice per load (pass 1 is the first request at a new depth, pass 2 the same request again). Timed loads, in their order: r1, n1, e1, r2, n2, e2.

| Arm | Loads | Draft | Context | Its own variables and flags |
|---|---|---|---:|---|
| v4r (control) | r1, r2 | on | 65536 | `GGML_CUDA_KEEPALIVE=decode GGML_CUDA_KEEPALIVE_EAGER=1` |
| v4nk | n1, n2 | on | 65536 | — |
| v4ne | e1, e2 | on | 65536 | `GGML_CUDA_KEEPALIVE=decode` |

### Generation, tok/s

Per load: pass 1, pass 2 → the load, with the load's time per token. A load's speed is its two passes' tokens over their time. Spread: the difference between the arm's fastest and slowest load. Class against `v4r`: gain when the mean is above the control's by more than the larger of the two spreads, loss when below by more than it, else negligible.

| Depth (tokens) | Arm | Per load | Spread | Against `v4r` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | v4r | r1: 23.06, 24.10 → **23.57** (42.42 ms a token); r2: 22.87, 24.09 → **23.46** (42.62 ms a token) | 0.11 | control | | |
|  | v4nk | n1: 23.02, 23.73 → **23.37** (42.79 ms a token); n2: 23.40, 24.16 → **23.78** (42.06 ms a token) | 0.41 | +0.06 tok/s | +0.2 % | negligible |
|  | v4ne | e1: 23.30, 24.10 → **23.70** (42.20 ms a token); e2: 23.34, 24.10 → **23.72** (42.16 ms a token) | 0.02 | +0.19 tok/s | +0.8 % | gain |
| 4,096 | v4r | r1: 28.42, 29.54 → **28.97** (34.52 ms a token); r2: 28.45, 29.54 → **28.99** (34.50 ms a token) | 0.02 | control | | |
|  | v4nk | n1: 30.83, 30.45 → **30.64** (32.64 ms a token); n2: 30.85, 30.83 → **30.84** (32.43 ms a token) | 0.20 | +1.76 tok/s | +6.1 % | gain |
|  | v4ne | e1: 23.66, 29.56 → **26.28** (38.05 ms a token); e2: 23.73, 29.46 → **26.29** (38.04 ms a token) | 0.01 | -2.69 tok/s | -9.3 % | loss |
| 8,192 | v4r | r1: 30.00, 31.59 → **30.77** (32.50 ms a token); r2: 29.51, 31.60 → **30.52** (32.77 ms a token) | 0.25 | control | | |
|  | v4nk | n1: 31.63, 30.87 → **31.24** (32.01 ms a token); n2: 32.34, 32.31 → **32.33** (30.94 ms a token) | 1.08 | +1.14 tok/s | +3.7 % | gain |
|  | v4ne | e1: 30.86, 31.61 → **31.23** (32.02 ms a token); e2: 30.87, 31.61 → **31.24** (32.01 ms a token) | 0.01 | +0.59 tok/s | +1.9 % | gain |
| 16,384 | v4r | r1: 30.33, 31.37 → **30.84** (32.43 ms a token); r2: 30.39, 31.39 → **30.88** (32.38 ms a token) | 0.04 | control | | |
|  | v4nk | n1: 29.37, 31.83 → **30.55** (32.74 ms a token); n2: 28.55, 29.43 → **28.98** (34.50 ms a token) | 1.56 | -1.09 tok/s | -3.5 % | negligible |
|  | v4ne | e1: 31.14, 31.24 → **31.19** (32.06 ms a token); e2: 31.15, 31.24 → **31.20** (32.06 ms a token) | 0.01 | +0.33 tok/s | +1.1 % | gain |
| 32,768 | v4r | r1: 21.23, 22.05 → **21.63** (46.23 ms a token); r2: 21.20, 22.05 → **21.61** (46.27 ms a token) | 0.02 | control | | |
|  | v4nk | n1: 20.39, 22.40 → **21.35** (46.84 ms a token); n2: 21.32, 21.63 → **21.47** (46.57 ms a token) | 0.12 | -0.21 tok/s | -1.0 % | loss |
|  | v4ne | e1: 21.43, 22.02 → **21.72** (46.04 ms a token); e2: 21.00, 22.02 → **21.50** (46.51 ms a token) | 0.22 | -0.01 tok/s | -0.1 % | negligible |
| 49,152 | v4r | r1: 23.89, 24.45 → **24.17** (41.37 ms a token); r2: 23.94, 24.49 → **24.21** (41.30 ms a token) | 0.04 | control | | |
|  | v4nk | n1: 22.36, 20.96 → **21.64** (46.22 ms a token); n2: 22.53, 21.43 → **21.96** (45.53 ms a token) | 0.33 | -2.39 tok/s | -9.9 % | loss |
|  | v4ne | e1: 24.14, 24.26 → **24.20** (41.32 ms a token); e2: 24.28, 24.48 → **24.38** (41.02 ms a token) | 0.18 | +0.10 tok/s | +0.4 % | negligible |

### Generation over all depths, tok/s

Every depth's tokens over their time, per load.

| Depth (tokens) | Arm | Per load | Spread | Against `v4r` | | Class |
|---:|---|---|---:|---:|---:|---|
| all | v4r | r1: **26.15**; r2: **26.11** | 0.04 | control | | |
|  | v4nk | n1: **25.72**; n2: **25.86** | 0.13 | -0.33 tok/s | -1.3 % | loss |
|  | v4ne | e1: **25.90**; e2: **25.88** | 0.01 | -0.24 tok/s | -0.9 % | loss |

### The step on pass 1 (the first request at a new depth), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `v4r`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `v4r` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | v4r | r1: 101.84 (109 steps); r2: 102.71 (109 steps) | 0.87 | control | | |
|  | v4nk | n1: 102.02 (109 steps); n2: 100.35 (109 steps) | 1.66 | -1.09 ms | -1.1 % | negligible |
|  | v4ne | e1: 100.79 (109 steps); e2: 100.61 (109 steps) | 0.18 | -1.58 ms | -1.5 % | gain |
| 4,096 | v4r | r1: 114.02 (79 steps); r2: 113.91 (79 steps) | 0.11 | control | | |
|  | v4nk | n1: 105.10 (79 steps); n2: 105.04 (79 steps) | 0.05 | -8.89 ms | -7.8 % | gain |
|  | v4ne | e1: 136.97 (79 steps); e2: 136.57 (79 steps) | 0.40 | +22.81 ms | +20.0 % | loss |
| 8,192 | v4r | r1: 118.53 (72 steps); r2: 120.48 (72 steps) | 1.95 | control | | |
|  | v4nk | n1: 112.42 (72 steps); n2: 109.95 (72 steps) | 2.47 | -8.32 ms | -7.0 % | gain |
|  | v4ne | e1: 115.21 (72 steps); e2: 115.17 (72 steps) | 0.04 | -4.32 ms | -3.6 % | gain |
| 16,384 | v4r | r1: 122.34 (69 steps); r2: 122.09 (69 steps) | 0.25 | control | | |
|  | v4nk | n1: 126.34 (69 steps); n2: 129.94 (69 steps) | 3.61 | +5.92 ms | +4.8 % | loss |
|  | v4ne | e1: 119.16 (69 steps); e2: 119.09 (69 steps) | 0.07 | -3.09 ms | -2.5 % | gain |
| 32,768 | v4r | r1: 131.08 (92 steps); r2: 131.28 (92 steps) | 0.20 | control | | |
|  | v4nk | n1: 136.47 (92 steps); n2: 130.52 (92 steps) | 5.96 | +2.32 ms | +1.8 % | negligible |
|  | v4ne | e1: 129.84 (92 steps); e2: 132.48 (92 steps) | 2.64 | -0.02 ms | -0.0 % | negligible |
| 49,152 | v4r | r1: 140.99 (76 steps); r2: 140.73 (76 steps) | 0.26 | control | | |
|  | v4nk | n1: 150.62 (76 steps); n2: 149.50 (76 steps) | 1.12 | +9.20 ms | +6.5 % | loss |
|  | v4ne | e1: 139.53 (76 steps); e2: 138.73 (76 steps) | 0.79 | -1.73 ms | -1.2 % | gain |

### The step on pass 2 (the same request again), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `v4r`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `v4r` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | v4r | r1: 98.34 (108 steps); r2: 98.40 (108 steps) | 0.06 | control | | |
|  | v4nk | n1: 99.91 (108 steps); n2: 98.10 (108 steps) | 1.80 | +0.63 ms | +0.6 % | negligible |
|  | v4ne | e1: 98.34 (108 steps); e2: 98.34 (108 steps) | 0.00 | -0.03 ms | -0.0 % | negligible |
| 4,096 | v4r | r1: 109.72 (79 steps); r2: 109.69 (79 steps) | 0.02 | control | | |
|  | v4nk | n1: 106.44 (79 steps); n2: 105.11 (79 steps) | 1.33 | -3.93 ms | -3.6 % | gain |
|  | v4ne | e1: 109.63 (79 steps); e2: 109.99 (79 steps) | 0.35 | +0.11 ms | +0.1 % | negligible |
| 8,192 | v4r | r1: 114.15 (71 steps); r2: 114.10 (71 steps) | 0.05 | control | | |
|  | v4nk | n1: 116.82 (71 steps); n2: 111.58 (71 steps) | 5.23 | +0.07 ms | +0.1 % | negligible |
|  | v4ne | e1: 114.07 (71 steps); e2: 114.05 (71 steps) | 0.02 | -0.07 ms | -0.1 % | gain |
| 16,384 | v4r | r1: 118.27 (69 steps); r2: 118.20 (69 steps) | 0.08 | control | | |
|  | v4nk | n1: 116.58 (69 steps); n2: 126.07 (69 steps) | 9.50 | +3.09 ms | +2.6 % | negligible |
|  | v4ne | e1: 118.76 (69 steps); e2: 118.77 (69 steps) | 0.02 | +0.53 ms | +0.4 % | loss |
| 32,768 | v4r | r1: 127.60 (91 steps); r2: 127.59 (91 steps) | 0.01 | control | | |
|  | v4nk | n1: 125.59 (91 steps); n2: 130.09 (91 steps) | 4.50 | +0.25 ms | +0.2 % | negligible |
|  | v4ne | e1: 127.77 (91 steps); e2: 127.77 (91 steps) | 0.00 | +0.18 ms | +0.1 % | loss |
| 49,152 | v4r | r1: 137.74 (76 steps); r2: 137.53 (76 steps) | 0.22 | control | | |
|  | v4nk | n1: 160.73 (76 steps); n2: 157.22 (76 steps) | 3.52 | +21.34 ms | +15.5 % | loss |
|  | v4ne | e1: 138.84 (76 steps); e2: 137.60 (76 steps) | 1.24 | +0.59 ms | +0.4 % | negligible |

### Prompt reading, tok/s

Per load: the tokens read at that depth over their time (a prime request reads a chunk on top of the cache; without primes each decode request reads its prompt), with the tokens read and the tokens reused. Requests that read under 32 tokens are left out.

| Depth (tokens) | Arm | Per load | Spread | Against `v4r` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | v4r | r1: 53.5 (64 read on 0 cached); r2: 51.7 (64 read on 0 cached) | 1.8 | control | | |
|  | v4nk | n1: 47.2 (64 read on 0 cached); n2: 48.5 (64 read on 0 cached) | 1.3 | -4.7 tok/s | -9.0 % | loss |
|  | v4ne | e1: 53.3 (64 read on 0 cached); e2: 53.9 (64 read on 0 cached) | 0.5 | +1.0 tok/s | +1.9 % | negligible |
| 4,096 | v4r | r1: 198.0 (4032 read on 0 cached); r2: 197.3 (4032 read on 0 cached) | 0.7 | control | | |
|  | v4nk | n1: 198.1 (4032 read on 0 cached); n2: 200.1 (4032 read on 0 cached) | 2.0 | +1.4 tok/s | +0.7 % | negligible |
|  | v4ne | e1: 198.8 (4032 read on 0 cached); e2: 197.6 (4032 read on 0 cached) | 1.2 | +0.5 tok/s | +0.3 % | negligible |
| 8,192 | v4r | r1: 207.9 (4612 read on 3516 cached); r2: 220.0 (4612 read on 3516 cached) | 12.1 | control | | |
|  | v4nk | n1: 221.9 (4612 read on 3516 cached); n2: 218.9 (4612 read on 3516 cached) | 3.0 | +6.4 tok/s | +3.0 % | negligible |
|  | v4ne | e1: 221.6 (4612 read on 3516 cached); e2: 223.4 (4612 read on 3516 cached) | 1.8 | +8.5 tok/s | +4.0 % | negligible |
| 16,384 | v4r | r1: 275.7 (12804 read on 3516 cached); r2: 281.1 (12804 read on 3516 cached) | 5.3 | control | | |
|  | v4nk | n1: 290.7 (12804 read on 3516 cached); n2: 282.6 (12804 read on 3516 cached) | 8.1 | +8.3 tok/s | +3.0 % | gain |
|  | v4ne | e1: 284.7 (12804 read on 3516 cached); e2: 279.5 (12804 read on 3516 cached) | 5.1 | +3.7 tok/s | +1.3 % | negligible |
| 32,768 | v4r | r1: 221.3 (16900 read on 15804 cached); r2: 224.6 (16900 read on 15804 cached) | 3.4 | control | | |
|  | v4nk | n1: 230.1 (16900 read on 15804 cached); n2: 230.0 (16900 read on 15804 cached) | 0.1 | +7.1 tok/s | +3.2 % | gain |
|  | v4ne | e1: 225.7 (16900 read on 15804 cached); e2: 225.4 (16900 read on 15804 cached) | 0.2 | +2.6 tok/s | +1.2 % | negligible |
| 49,152 | v4r | r1: 178.7 (16900 read on 32188 cached); r2: 179.0 (16900 read on 32188 cached) | 0.3 | control | | |
|  | v4nk | n1: 181.5 (16900 read on 32188 cached); n2: 181.4 (16900 read on 32188 cached) | 0.0 | +2.6 tok/s | +1.4 % | gain |
|  | v4ne | e1: 174.7 (16900 read on 32188 cached); e2: 180.4 (16900 read on 32188 cached) | 5.8 | -1.3 tok/s | -0.7 % | negligible |

### The draft and the step

Per load, pass 1: draft tokens accepted of drafted (the share), tokens a step (generated tokens over steps, a step being one pass of the main model: generated minus accepted), and the time of a step. Pass 2 is listed where its counters differ.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | v4r | r1: 147 of 321 (0.458), 2.35 a step, 101.8 ms a step (pass 2: 148 of 318 (0.465), 2.37 a step, 98.3 ms a step); r2: 147 of 321 (0.458), 2.35 a step, 102.7 ms a step (pass 2: 148 of 318 (0.465), 2.37 a step, 98.4 ms a step) |
|  | v4nk | n1: 147 of 321 (0.458), 2.35 a step, 102.0 ms a step (pass 2: 148 of 318 (0.465), 2.37 a step, 99.9 ms a step); n2: 147 of 321 (0.458), 2.35 a step, 100.4 ms a step (pass 2: 148 of 318 (0.465), 2.37 a step, 98.1 ms a step) |
|  | v4ne | e1: 147 of 321 (0.458), 2.35 a step, 100.8 ms a step (pass 2: 148 of 318 (0.465), 2.37 a step, 98.3 ms a step); e2: 147 of 321 (0.458), 2.35 a step, 100.6 ms a step (pass 2: 148 of 318 (0.465), 2.37 a step, 98.3 ms a step) |
| 4,096 | v4r | r1: 177 of 231 (0.766), 3.24 a step, 114.0 ms a step; r2: 177 of 231 (0.766), 3.24 a step, 113.9 ms a step |
|  | v4nk | n1: 177 of 231 (0.766), 3.24 a step, 105.1 ms a step; n2: 177 of 231 (0.766), 3.24 a step, 105.0 ms a step |
|  | v4ne | e1: 177 of 231 (0.766), 3.24 a step, 137.0 ms a step; e2: 177 of 231 (0.766), 3.24 a step, 136.6 ms a step |
| 8,192 | v4r | r1: 184 of 212 (0.868), 3.56 a step, 118.5 ms a step (pass 2: 185 of 209 (0.885), 3.61 a step, 114.1 ms a step); r2: 184 of 212 (0.868), 3.56 a step, 120.5 ms a step (pass 2: 185 of 209 (0.885), 3.61 a step, 114.1 ms a step) |
|  | v4nk | n1: 184 of 212 (0.868), 3.56 a step, 112.4 ms a step (pass 2: 185 of 209 (0.885), 3.61 a step, 116.8 ms a step); n2: 184 of 212 (0.868), 3.56 a step, 110.0 ms a step (pass 2: 185 of 209 (0.885), 3.61 a step, 111.6 ms a step) |
|  | v4ne | e1: 184 of 212 (0.868), 3.56 a step, 115.2 ms a step (pass 2: 185 of 209 (0.885), 3.61 a step, 114.1 ms a step); e2: 184 of 212 (0.868), 3.56 a step, 115.2 ms a step (pass 2: 185 of 209 (0.885), 3.61 a step, 114.0 ms a step) |
| 16,384 | v4r | r1: 187 of 203 (0.921), 3.71 a step, 122.3 ms a step; r2: 187 of 203 (0.921), 3.71 a step, 122.1 ms a step |
|  | v4nk | n1: 187 of 203 (0.921), 3.71 a step, 126.3 ms a step; n2: 187 of 203 (0.921), 3.71 a step, 129.9 ms a step |
|  | v4ne | e1: 187 of 203 (0.921), 3.71 a step, 119.2 ms a step; e2: 187 of 203 (0.921), 3.71 a step, 119.1 ms a step |
| 32,768 | v4r | r1: 164 of 273 (0.601), 2.78 a step, 131.1 ms a step (pass 2: 165 of 270 (0.611), 2.81 a step, 127.6 ms a step); r2: 164 of 273 (0.601), 2.78 a step, 131.3 ms a step (pass 2: 165 of 270 (0.611), 2.81 a step, 127.6 ms a step) |
|  | v4nk | n1: 164 of 273 (0.601), 2.78 a step, 136.5 ms a step (pass 2: 165 of 270 (0.611), 2.81 a step, 125.6 ms a step); n2: 164 of 273 (0.601), 2.78 a step, 130.5 ms a step (pass 2: 165 of 270 (0.611), 2.81 a step, 130.1 ms a step) |
|  | v4ne | e1: 164 of 273 (0.601), 2.78 a step, 129.8 ms a step (pass 2: 165 of 270 (0.611), 2.81 a step, 127.8 ms a step); e2: 164 of 273 (0.601), 2.78 a step, 132.5 ms a step (pass 2: 165 of 270 (0.611), 2.81 a step, 127.8 ms a step) |
| 49,152 | v4r | r1: 180 of 225 (0.800), 3.37 a step, 141.0 ms a step; r2: 180 of 225 (0.800), 3.37 a step, 140.7 ms a step |
|  | v4nk | n1: 180 of 225 (0.800), 3.37 a step, 150.6 ms a step; n2: 180 of 225 (0.800), 3.37 a step, 149.5 ms a step |
|  | v4ne | e1: 180 of 225 (0.800), 3.37 a step, 139.5 ms a step; e2: 180 of 225 (0.800), 3.37 a step, 138.7 ms a step |

### The sampler pass

One request per depth and load with the server line's own sampler and a fixed seed: tok/s, the draft counters and the step.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | v4r | r1: 24.25 tok/s, 154 of 302 (0.510), 2.51 a step, 103.5 ms a step; r2: 24.07 tok/s, 154 of 302 (0.510), 2.51 a step, 104.3 ms a step |
|  | v4nk | n1: 23.34 tok/s, 154 of 302 (0.510), 2.51 a step, 107.5 ms a step; n2: 23.89 tok/s, 154 of 302 (0.510), 2.51 a step, 105.0 ms a step |
|  | v4ne | e1: 24.70 tok/s, 154 of 302 (0.510), 2.51 a step, 101.6 ms a step; e2: 24.66 tok/s, 154 of 302 (0.510), 2.51 a step, 101.8 ms a step |
| 4,096 | v4r | r1: 29.86 tok/s, 179 of 228 (0.785), 3.32 a step, 111.3 ms a step; r2: 29.89 tok/s, 179 of 228 (0.785), 3.32 a step, 111.2 ms a step |
|  | v4nk | n1: 30.43 tok/s, 179 of 228 (0.785), 3.32 a step, 109.3 ms a step; n2: 30.45 tok/s, 179 of 228 (0.785), 3.32 a step, 109.2 ms a step |
|  | v4ne | e1: 29.87 tok/s, 179 of 228 (0.785), 3.32 a step, 111.3 ms a step; e2: 29.89 tok/s, 179 of 228 (0.785), 3.32 a step, 111.2 ms a step |
| 8,192 | v4r | r1: 29.59 tok/s, 182 of 216 (0.843), 3.46 a step, 116.9 ms a step; r2: 29.60 tok/s, 182 of 216 (0.843), 3.46 a step, 116.9 ms a step |
|  | v4nk | n1: 28.99 tok/s, 182 of 216 (0.843), 3.46 a step, 119.3 ms a step; n2: 30.14 tok/s, 182 of 216 (0.843), 3.46 a step, 114.8 ms a step |
|  | v4ne | e1: 30.03 tok/s, 182 of 216 (0.843), 3.46 a step, 115.2 ms a step; e2: 30.08 tok/s, 182 of 216 (0.843), 3.46 a step, 115.0 ms a step |
| 16,384 | v4r | r1: 27.14 tok/s, 177 of 233 (0.760), 3.24 a step, 119.4 ms a step; r2: 27.19 tok/s, 177 of 233 (0.760), 3.24 a step, 119.2 ms a step |
|  | v4nk | n1: 24.05 tok/s, 177 of 233 (0.760), 3.24 a step, 134.7 ms a step; n2: 24.92 tok/s, 177 of 233 (0.760), 3.24 a step, 130.0 ms a step |
|  | v4ne | e1: 27.25 tok/s, 177 of 233 (0.760), 3.24 a step, 118.9 ms a step; e2: 27.17 tok/s, 177 of 233 (0.760), 3.24 a step, 119.3 ms a step |
| 32,768 | v4r | r1: 22.26 tok/s, 168 of 261 (0.644), 2.91 a step, 130.7 ms a step; r2: 22.24 tok/s, 168 of 261 (0.644), 2.91 a step, 130.8 ms a step |
|  | v4nk | n1: 22.48 tok/s, 168 of 261 (0.644), 2.91 a step, 129.4 ms a step; n2: 20.19 tok/s, 168 of 261 (0.644), 2.91 a step, 144.1 ms a step |
|  | v4ne | e1: 22.26 tok/s, 168 of 261 (0.644), 2.91 a step, 130.7 ms a step; e2: 20.53 tok/s, 168 of 261 (0.644), 2.91 a step, 141.7 ms a step |
| 49,152 | v4r | r1: 23.53 tok/s, 179 of 225 (0.796), 3.32 a step, 141.3 ms a step; r2: 23.59 tok/s, 179 of 225 (0.796), 3.32 a step, 140.9 ms a step |
|  | v4nk | n1: 20.20 tok/s, 179 of 225 (0.796), 3.32 a step, 164.6 ms a step; n2: 18.90 tok/s, 179 of 225 (0.796), 3.32 a step, 175.9 ms a step |
|  | v4ne | e1: 23.82 tok/s, 179 of 225 (0.796), 3.32 a step, 139.6 ms a step; e2: 22.11 tok/s, 179 of 225 (0.796), 3.32 a step, 150.3 ms a step |

### Checks

- Times: every load started on a machine checked quiet (valid=1).
- Prompts: 64 ids at depth 0, 4096 ids at depth 4,096, 8192 ids at depth 8,192, 16384 ids at depth 16,384, 32768 ids at depth 32,768, 49152 ids at depth 49,152; the same ids on every load and pass.
- Tokens out: 256 on every timed request.
- Greedy reply at depth 0: the same on every pass and load of every arm.
- Greedy reply at depth 4,096: the same on every pass and load of every arm.
- Greedy reply at depth 8,192: the same on every pass and load of every arm.
- Greedy reply at depth 16,384: the same on every pass and load of every arm.
- Greedy reply at depth 32,768: the same on every pass and load of every arm.
- Greedy reply at depth 49,152: the same on every pass and load of every arm.
- Primes (read on cached, the cache's source): 4,096: 4032 on 0 (none); 8,192: 4612 on 3516 (checkpoint); 16,384: 12804 on 3516 (checkpoint); 32,768: 16900 on 15804 (checkpoint); 49,152: 16900 on 32188 (checkpoint).
- Graph reallocations of the target, requests with one per load: r1: 0 of 24 requests; n1: 0 of 24 requests; e1: 0 of 24 requests; r2: 0 of 24 requests; n2: 0 of 24 requests; e2: 0 of 24 requests.
