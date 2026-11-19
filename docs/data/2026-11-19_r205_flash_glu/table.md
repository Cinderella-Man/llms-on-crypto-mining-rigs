## The deep lane: off, u1, u2

Model file `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf`, llama-server version: 1.0.0-dev (build 12967, commit 6026911533).

256 tokens per request at temperature 0, each request twice per load (pass 1 is the first request at a new depth, pass 2 the same request again). Timed loads, in their order: f1, a1, b1, f2, a2, b2.

| Arm | Loads | Draft | Context | Its own variables and flags |
|---|---|---|---:|---|
| off (control) | f1, f2 | on | 65536 | — |
| u1 | a1, a2 | on | 65536 | `GGML_CUDA_MMID_GLU_UNFUSE=1` |
| u2 | b1, b2 | on | 65536 | `GGML_CUDA_MMID_GLU_UNFUSE=2` |

### Generation, tok/s

Per load: pass 1, pass 2 → the load, with the load's time per token. A load's speed is its two passes' tokens over their time. Spread: the difference between the arm's fastest and slowest load. Class against `off`: gain when the mean is above the control's by more than the larger of the two spreads, loss when below by more than it, else negligible.

| Depth (tokens) | Arm | Per load | Spread | Against `off` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | off | f1: 23.05, 24.09 → **23.56** (42.45 ms a token); f2: 22.88, 24.04 → **23.44** (42.65 ms a token) | 0.11 | control | | |
|  | u1 | a1: 22.35, 23.50 → **22.91** (43.65 ms a token); a2: 22.37, 23.50 → **22.92** (43.63 ms a token) | 0.01 | -0.59 tok/s | -2.5 % | loss |
|  | u2 | b1: 22.38, 23.55 → **22.95** (43.58 ms a token); b2: 22.47, 23.51 → **22.98** (43.52 ms a token) | 0.03 | -0.54 tok/s | -2.3 % | loss |
| 8,192 | off | f1: 25.87, 32.37 → **28.76** (34.77 ms a token); f2: 30.74, 32.37 → **31.54** (31.71 ms a token) | 2.78 | control | | |
|  | u1 | a1: 30.18, 31.75 → **30.95** (32.32 ms a token); a2: 28.00, 31.83 → **29.79** (33.57 ms a token) | 1.15 | +0.22 tok/s | +0.7 % | negligible |
|  | u2 | b1: 30.21, 31.77 → **30.97** (32.29 ms a token); b2: 30.24, 31.71 → **30.96** (32.30 ms a token) | 0.01 | +0.81 tok/s | +2.7 % | negligible |

### Generation over all depths, tok/s

Every depth's tokens over their time, per load.

| Depth (tokens) | Arm | Per load | Spread | Against `off` | | Class |
|---:|---|---|---:|---:|---:|---|
| all | off | f1: **25.90**; f2: **26.89** | 0.99 | control | | |
|  | u1 | a1: **26.33**; a2: **25.91** | 0.42 | -0.28 tok/s | -1.1 % | negligible |
|  | u2 | b1: **26.36**; b2: **26.38** | 0.02 | -0.03 tok/s | -0.1 % | negligible |

### The step on pass 1 (the first request at a new depth), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `off`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `off` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | off | f1: 101.88 (109 steps); f2: 102.67 (109 steps) | 0.79 | control | | |
|  | u1 | a1: 105.06 (109 steps); a2: 105.00 (109 steps) | 0.06 | +2.76 ms | +2.7 % | loss |
|  | u2 | b1: 104.96 (109 steps); b2: 104.53 (109 steps) | 0.43 | +2.47 ms | +2.4 % | loss |
| 8,192 | off | f1: 139.35 (71 steps); f2: 117.28 (71 steps) | 22.08 | control | | |
|  | u1 | a1: 119.49 (71 steps); a2: 128.76 (71 steps) | 9.27 | -4.19 ms | -3.3 % | negligible |
|  | u2 | b1: 119.36 (71 steps); b2: 119.24 (71 steps) | 0.11 | -9.02 ms | -7.0 % | negligible |

### The step on pass 2 (the same request again), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `off`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `off` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | off | f1: 98.40 (108 steps); f2: 98.59 (108 steps) | 0.19 | control | | |
|  | u1 | a1: 100.88 (108 steps); a2: 100.85 (108 steps) | 0.03 | +2.37 ms | +2.4 % | loss |
|  | u2 | b1: 100.67 (108 steps); b2: 100.83 (108 steps) | 0.16 | +2.25 ms | +2.3 % | loss |
| 8,192 | off | f1: 112.97 (70 steps); f2: 112.98 (70 steps) | 0.01 | control | | |
|  | u1 | a1: 115.17 (70 steps); a2: 114.91 (70 steps) | 0.26 | +2.06 ms | +1.8 % | loss |
|  | u2 | b1: 115.12 (70 steps); b2: 115.33 (70 steps) | 0.20 | +2.25 ms | +2.0 % | loss |

### Prompt reading, tok/s

Per load: the tokens read at that depth over their time (a prime request reads a chunk on top of the cache; without primes each decode request reads its prompt), with the tokens read and the tokens reused. Requests that read under 32 tokens are left out.

| Depth (tokens) | Arm | Per load | Spread | Against `off` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | off | f1: 53.5 (64 read on 0 cached); f2: 53.1 (64 read on 0 cached) | 0.3 | control | | |
|  | u1 | a1: 53.2 (64 read on 0 cached); a2: 52.3 (64 read on 0 cached) | 0.9 | -0.6 tok/s | -1.1 % | negligible |
|  | u2 | b1: 52.1 (64 read on 0 cached); b2: 53.2 (64 read on 0 cached) | 1.1 | -0.7 tok/s | -1.3 % | negligible |
| 8,192 | off | f1: 217.7 (8128 read on 0 cached); f2: 222.1 (8128 read on 0 cached) | 4.4 | control | | |
|  | u1 | a1: 221.2 (8128 read on 0 cached); a2: 224.6 (8128 read on 0 cached) | 3.4 | +3.0 tok/s | +1.4 % | negligible |
|  | u2 | b1: 224.7 (8128 read on 0 cached); b2: 225.2 (8128 read on 0 cached) | 0.5 | +5.0 tok/s | +2.3 % | gain |

### The draft and the step

Per load, pass 1: draft tokens accepted of drafted (the share), tokens a step (generated tokens over steps, a step being one pass of the main model: generated minus accepted), and the time of a step. Pass 2 is listed where its counters differ.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | off | f1: 147 of 321 (0.458), 2.35 a step, 101.9 ms a step (pass 2: 148 of 318 (0.465), 2.37 a step, 98.4 ms a step); f2: 147 of 321 (0.458), 2.35 a step, 102.7 ms a step (pass 2: 148 of 318 (0.465), 2.37 a step, 98.6 ms a step) |
|  | u1 | a1: 147 of 321 (0.458), 2.35 a step, 105.1 ms a step (pass 2: 148 of 318 (0.465), 2.37 a step, 100.9 ms a step); a2: 147 of 321 (0.458), 2.35 a step, 105.0 ms a step (pass 2: 148 of 318 (0.465), 2.37 a step, 100.8 ms a step) |
|  | u2 | b1: 147 of 321 (0.458), 2.35 a step, 105.0 ms a step (pass 2: 148 of 318 (0.465), 2.37 a step, 100.7 ms a step); b2: 147 of 321 (0.458), 2.35 a step, 104.5 ms a step (pass 2: 148 of 318 (0.465), 2.37 a step, 100.8 ms a step) |
| 8,192 | off | f1: 185 of 208 (0.889), 3.61 a step, 139.4 ms a step (pass 2: 186 of 205 (0.907), 3.66 a step, 113.0 ms a step); f2: 185 of 208 (0.889), 3.61 a step, 117.3 ms a step (pass 2: 186 of 205 (0.907), 3.66 a step, 113.0 ms a step) |
|  | u1 | a1: 185 of 208 (0.889), 3.61 a step, 119.5 ms a step (pass 2: 186 of 205 (0.907), 3.66 a step, 115.2 ms a step); a2: 185 of 208 (0.889), 3.61 a step, 128.8 ms a step (pass 2: 186 of 205 (0.907), 3.66 a step, 114.9 ms a step) |
|  | u2 | b1: 185 of 208 (0.889), 3.61 a step, 119.4 ms a step (pass 2: 186 of 205 (0.907), 3.66 a step, 115.1 ms a step); b2: 185 of 208 (0.889), 3.61 a step, 119.2 ms a step (pass 2: 186 of 205 (0.907), 3.66 a step, 115.3 ms a step) |

### The sampler pass

One request per depth and load with the server line's own sampler and a fixed seed: tok/s, the draft counters and the step.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | off | f1: 24.21 tok/s, 154 of 302 (0.510), 2.51 a step, 103.7 ms a step; f2: 24.04 tok/s, 154 of 302 (0.510), 2.51 a step, 104.4 ms a step |
|  | u1 | a1: 23.49 tok/s, 154 of 302 (0.510), 2.51 a step, 106.9 ms a step; a2: 23.45 tok/s, 154 of 302 (0.510), 2.51 a step, 107.0 ms a step |
|  | u2 | b1: 23.44 tok/s, 154 of 302 (0.510), 2.51 a step, 107.1 ms a step; b2: 23.60 tok/s, 154 of 302 (0.510), 2.51 a step, 106.3 ms a step |
| 8,192 | off | f1: 21.93 tok/s, 165 of 268 (0.616), 2.81 a step, 128.3 ms a step; f2: 23.86 tok/s, 165 of 268 (0.616), 2.81 a step, 117.9 ms a step |
|  | u1 | a1: 23.49 tok/s, 165 of 268 (0.616), 2.81 a step, 119.7 ms a step; a2: 23.36 tok/s, 165 of 268 (0.616), 2.81 a step, 120.4 ms a step |
|  | u2 | b1: 23.44 tok/s, 165 of 268 (0.616), 2.81 a step, 120.0 ms a step; b2: 23.57 tok/s, 165 of 268 (0.616), 2.81 a step, 119.4 ms a step |

### Checks

- Times: every load started on a machine checked quiet (valid=1).
- Prompts: 64 ids at depth 0, 8192 ids at depth 8,192; the same ids on every load and pass.
- Tokens out: 256 on every timed request.
- Greedy reply at depth 0: the same on every pass and load of every arm.
- Greedy reply at depth 8,192: the same on every pass and load of every arm.
- Primes (read on cached, the cache's source): 8,192: 8128 on 0 (none).
- Graph reallocations of the target, requests with one per load: f1: 0 of 8 requests; a1: 0 of 8 requests; b1: 0 of 8 requests; f2: 0 of 8 requests; a2: 0 of 8 requests; b2: 0 of 8 requests.
