## The deep lane: u2, p0

Model file `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-pmix-fast-gmix-q2-00001-of-00003.gguf`, llama-server version: 1.0.0-dev (build 13244, commit 883f21f2c5).

256 tokens per request at temperature 0, each request twice per load (pass 1 is the first request at a new depth, pass 2 the same request again). Timed loads, in their order: u2b, p0.

| Arm | Loads | Draft | Context | Its own variables and flags |
|---|---|---|---:|---|
| u2 (control) | u2b | on | 65536 | `GGML_CUDA_MTOK_IQ_FAST=2` |
| p0 | p0 | on | 65536 | `-m <MODELS>/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-pmix-fast-00001-of-00003.gguf` |

### Generation, tok/s

Per load: pass 1, pass 2 → the load, with the load's time per token. A load's speed is its two passes' tokens over their time. Spread: the difference between the arm's fastest and slowest load. Class against `u2`: gain when the mean is above the control's by more than the larger of the two spreads, loss when below by more than it, else negligible.

| Depth (tokens) | Arm | Per load | Spread | Against `u2` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | u2 | u2b: 23.54, 24.68 → **24.09** (41.51 ms a token) | — | control | | |
|  | p0 | p0: 28.16, 30.62 → **29.34** (34.09 ms a token) | — | +5.24 tok/s | +21.8 % | no class (one load) |
| 8,192 | u2 | u2b: 20.67, 23.10 → **21.82** (45.84 ms a token) | — | control | | |
|  | p0 | p0: 25.97, 27.96 → **26.93** (37.14 ms a token) | — | +5.11 tok/s | +23.4 % | no class (one load) |

### Generation over all depths, tok/s

Every depth's tokens over their time, per load.

| Depth (tokens) | Arm | Per load | Spread | Against `u2` | | Class |
|---:|---|---|---:|---:|---:|---|
| all | u2 | u2b: **22.90** | — | control | | |
|  | p0 | p0: **28.08** | — | +5.18 tok/s | +22.6 % | no class (one load) |

### The step on pass 1 (the first request at a new depth), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `u2`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `u2` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | u2 | u2b: 103.59 (105 steps) | — | control | | |
|  | p0 | p0: 102.16 (89 steps) | — | -1.44 ms | -1.4 % | no class (one load) |
| 8,192 | u2 | u2b: 127.69 (97 steps) | — | control | | |
|  | p0 | p0: 113.32 (87 steps) | — | -14.37 ms | -11.3 % | no class (one load) |

### The step on pass 2 (the same request again), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `u2`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `u2` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | u2 | u2b: 99.75 (104 steps) | — | control | | |
|  | p0 | p0: 95.00 (88 steps) | — | -4.75 ms | -4.8 % | no class (one load) |
| 8,192 | u2 | u2b: 114.25 (97 steps) | — | control | | |
|  | p0 | p0: 109.01 (84 steps) | — | -5.24 ms | -4.6 % | no class (one load) |

### Prompt reading, tok/s

Per load: the tokens read at that depth over their time (a prime request reads a chunk on top of the cache; without primes each decode request reads its prompt), with the tokens read and the tokens reused. Requests that read under 32 tokens are left out.

| Depth (tokens) | Arm | Per load | Spread | Against `u2` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | u2 | u2b: 48.3 (64 read on 0 cached) | — | control | | |
|  | p0 | p0: 55.5 (64 read on 0 cached) | — | +7.2 tok/s | +14.9 % | no class (one load) |
| 8,192 | u2 | u2b: 147.5 (8128 read on 0 cached) | — | control | | |
|  | p0 | p0: 224.4 (8128 read on 0 cached) | — | +76.9 tok/s | +52.1 % | no class (one load) |

### The draft and the step

Per load, pass 1: draft tokens accepted of drafted (the share), tokens a step (generated tokens over steps, a step being one pass of the main model: generated minus accepted), and the time of a step. Pass 2 is listed where its counters differ.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | u2 | u2b: 151 of 312 (0.484), 2.44 a step, 103.6 ms a step (pass 2: 152 of 309 (0.492), 2.46 a step, 99.7 ms a step) |
|  | p0 | p0: 167 of 261 (0.640), 2.88 a step, 102.2 ms a step (pass 2: 168 of 258 (0.651), 2.91 a step, 95.0 ms a step) |
| 8,192 | u2 | u2b: 159 of 286 (0.556), 2.64 a step, 127.7 ms a step |
|  | p0 | p0: 169 of 257 (0.658), 2.94 a step, 113.3 ms a step (pass 2: 172 of 248 (0.694), 3.05 a step, 109.0 ms a step) |

### The sampler pass

One request per depth and load with the server line's own sampler and a fixed seed: tok/s, the draft counters and the step.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | u2 | u2b: 20.58 tok/s, 137 of 354 (0.387), 2.15 a step, 104.5 ms a step |
|  | p0 | p0: 31.44 tok/s, 171 of 251 (0.681), 3.01 a step, 95.8 ms a step |
| 8,192 | u2 | u2b: 23.17 tok/s, 170 of 252 (0.675), 2.98 a step, 128.5 ms a step |
|  | p0 | p0: 26.01 tok/s, 168 of 258 (0.651), 2.91 a step, 111.9 ms a step |

### Checks

- Times: every load started on a machine checked quiet (valid=1).
- Prompts: 64 ids at depth 0, 8192 ids at depth 8,192; the same ids on every load and pass.
- Tokens out: 256 on every timed request.
- Greedy reply at depth 0: **differs from `u2` in p0.**
- Greedy reply at depth 8,192: **differs from `u2` in p0.**
- Primes (read on cached, the cache's source): 8,192: 8128 on 0 (none).
- Graph reallocations of the target, requests with one per load: u2b: 0 of 8 requests; p0: 0 of 8 requests.
