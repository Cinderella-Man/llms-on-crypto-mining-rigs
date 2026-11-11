## The deep lane: base, host, nopf, trace

Model file `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf`, llama-server version: 0.9.0-dev (build 12585, commit 8d329fa0ed).

256 tokens per request at temperature 0, each request twice per load (pass 1 is the first request at a new depth, pass 2 the same request again). Timed loads, in their order: base1, host1, nopf1, base2, host2, nopf2, trace.

| Arm | Loads | Draft | Context | Its own variables and flags |
|---|---|---|---:|---|
| base (control) | base1, base2 | on | 65536 | — |
| host | host1, host2 | on | 65536 | `LLAMA_MTP_H_DEV=0` |
| nopf | nopf1, nopf2 | on | 65536 | `GGML_SCHED_PREFETCH=0` |
| trace | trace | on | 65536 | `LLAMA_UBATCH_TRACE=1`, `-lv 4` |

### Generation, tok/s

Per load: pass 1, pass 2 → the load, with the load's time per token. A load's speed is its two passes' tokens over their time. Spread: the difference between the arm's fastest and slowest load. Class against `base`: gain when the mean is above the control's by more than the larger of the two spreads, loss when below by more than it, else negligible.

| Depth (tokens) | Arm | Per load | Spread | Against `base` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | base | base1: 26.20, 28.14 → **27.14** (36.85 ms a token); base2: 26.34, 28.04 → **27.16** (36.81 ms a token) | 0.03 | control | | |
|  | host | host1: 25.87, 27.72 → **26.76** (37.37 ms a token); host2: 25.87, 27.77 → **26.79** (37.33 ms a token) | 0.03 | -0.38 tok/s | -1.4 % | loss |
|  | nopf | nopf1: 26.05, 28.05 → **27.01** (37.02 ms a token); nopf2: 26.18, 28.15 → **27.13** (36.86 ms a token) | 0.12 | -0.08 tok/s | -0.3 % | negligible |
|  | trace | trace: 26.13, 28.11 → **27.08** (36.92 ms a token) | — | -0.07 tok/s | -0.3 % | no class (one load) |
| 16,384 | base | base1: 17.74, 18.43 → **18.08** (55.32 ms a token); base2: 17.73, 18.38 → **18.05** (55.40 ms a token) | 0.03 | control | | |
|  | host | host1: 15.29, 18.26 → **16.64** (60.08 ms a token); host2: 14.82, 18.24 → **16.35** (61.14 ms a token) | 0.29 | -1.56 tok/s | -8.7 % | loss |
|  | nopf | nopf1: 17.77, 18.39 → **18.08** (55.32 ms a token); nopf2: 17.72, 18.40 → **18.05** (55.40 ms a token) | 0.02 | -0.00 tok/s | -0.0 % | negligible |
|  | trace | trace: 17.69, 18.36 → **18.02** (55.50 ms a token) | — | -0.05 tok/s | -0.3 % | no class (one load) |
| 49,152 | base | base1: 10.72, 11.43 → **11.06** (90.39 ms a token); base2: 9.95, 11.45 → **10.65** (93.91 ms a token) | 0.41 | control | | |
|  | host | host1: 9.84, 11.37 → **10.55** (94.77 ms a token); host2: 9.99, 11.37 → **10.64** (94.01 ms a token) | 0.08 | -0.26 tok/s | -2.4 % | negligible |
|  | nopf | nopf1: 10.75, 11.44 → **11.08** (90.23 ms a token); nopf2: 10.70, 11.44 → **11.06** (90.44 ms a token) | 0.03 | +0.21 tok/s | +2.0 % | negligible |
|  | trace | trace: 9.65, 11.44 → **10.47** (95.52 ms a token) | — | -0.39 tok/s | -3.6 % | no class (one load) |

### Generation over all depths, tok/s

Every depth's tokens over their time, per load.

| Depth (tokens) | Arm | Per load | Spread | Against `base` | | Class |
|---:|---|---|---:|---:|---:|---|
| all | base | base1: **16.43**; base2: **16.12** | 0.31 | control | | |
|  | host | host1: **15.61**; host2: **15.59** | 0.02 | -0.68 tok/s | -4.2 % | loss |
|  | nopf | nopf1: **16.43**; nopf2: **16.42** | 0.01 | +0.15 tok/s | +0.9 % | negligible |
|  | trace | trace: **15.96** | — | -0.31 tok/s | -1.9 % | no class (one load) |

### The step on pass 1 (the first request at a new depth), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `base`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `base` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | base | base1: 125.25 (78 steps); base2: 124.61 (78 steps) | 0.64 | control | | |
|  | host | host1: 126.88 (78 steps); host2: 126.86 (78 steps) | 0.02 | +1.94 ms | +1.6 % | loss |
|  | nopf | nopf1: 126.01 (78 steps); nopf2: 125.37 (78 steps) | 0.64 | +0.75 ms | +0.6 % | loss |
|  | trace | trace: 125.62 (78 steps) | — | +0.69 ms | +0.5 % | no class (one load) |
| 16,384 | base | base1: 185.03 (78 steps); base2: 185.11 (78 steps) | 0.08 | control | | |
|  | host | host1: 214.65 (78 steps); host2: 221.46 (78 steps) | 6.81 | +32.99 ms | +17.8 % | loss |
|  | nopf | nopf1: 184.69 (78 steps); nopf2: 185.24 (78 steps) | 0.55 | -0.10 ms | -0.1 % | negligible |
|  | trace | trace: 185.57 (78 steps) | — | +0.50 ms | +0.3 % | no class (one load) |
| 49,152 | base | base1: 318.49 (75 steps); base2: 343.01 (75 steps) | 24.52 | control | | |
|  | host | host1: 346.83 (75 steps); host2: 341.62 (75 steps) | 5.21 | +13.47 ms | +4.1 % | negligible |
|  | nopf | nopf1: 317.59 (75 steps); nopf2: 319.13 (75 steps) | 1.53 | -12.39 ms | -3.7 % | negligible |
|  | trace | trace: 353.82 (75 steps) | — | +23.06 ms | +7.0 % | no class (one load) |

### The step on pass 2 (the same request again), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `base`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `base` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | base | base1: 121.28 (75 steps); base2: 121.71 (75 steps) | 0.43 | control | | |
|  | host | host1: 123.15 (75 steps); host2: 122.89 (75 steps) | 0.26 | +1.52 ms | +1.3 % | loss |
|  | nopf | nopf1: 121.71 (75 steps); nopf2: 121.25 (75 steps) | 0.46 | -0.02 ms | -0.0 % | negligible |
|  | trace | trace: 121.42 (75 steps) | — | -0.08 ms | -0.1 % | no class (one load) |
| 16,384 | base | base1: 180.40 (77 steps); base2: 180.85 (77 steps) | 0.44 | control | | |
|  | host | host1: 182.06 (77 steps); host2: 182.23 (77 steps) | 0.17 | +1.52 ms | +0.8 % | loss |
|  | nopf | nopf1: 180.76 (77 steps); nopf2: 180.71 (77 steps) | 0.06 | +0.11 ms | +0.1 % | negligible |
|  | trace | trace: 181.08 (77 steps) | — | +0.45 ms | +0.3 % | no class (one load) |
| 49,152 | base | base1: 298.57 (75 steps); base2: 298.06 (75 steps) | 0.51 | control | | |
|  | host | host1: 300.14 (75 steps); host2: 300.19 (75 steps) | 0.05 | +1.84 ms | +0.6 % | loss |
|  | nopf | nopf1: 298.41 (75 steps); nopf2: 298.30 (75 steps) | 0.11 | +0.04 ms | +0.0 % | negligible |
|  | trace | trace: 298.28 (75 steps) | — | -0.04 ms | -0.0 % | no class (one load) |

### Prompt reading, tok/s

Per load: the tokens read at that depth over their time (a prime request reads a chunk on top of the cache; without primes each decode request reads its prompt), with the tokens read and the tokens reused. Requests that read under 32 tokens are left out.

| Depth (tokens) | Arm | Per load | Spread | Against `base` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | base | base1: 31.9 (64 read on 0 cached); base2: 32.8 (64 read on 0 cached) | 0.9 | control | | |
|  | host | host1: 32.9 (64 read on 0 cached); host2: 32.1 (64 read on 0 cached) | 0.8 | +0.2 tok/s | +0.6 % | negligible |
|  | nopf | nopf1: 31.4 (64 read on 0 cached); nopf2: 32.6 (64 read on 0 cached) | 1.2 | -0.3 tok/s | -1.0 % | negligible |
|  | trace | trace: 32.2 (64 read on 0 cached) | — | -0.2 tok/s | -0.5 % | no class (one load) |
| 16,384 | base | base1: 131.8 (16320 read on 0 cached); base2: 132.1 (16320 read on 0 cached) | 0.3 | control | | |
|  | host | host1: 126.3 (16320 read on 0 cached); host2: 124.5 (16320 read on 0 cached) | 1.8 | -6.6 tok/s | -5.0 % | loss |
|  | nopf | nopf1: 132.1 (16320 read on 0 cached); nopf2: 131.5 (16320 read on 0 cached) | 0.6 | -0.2 tok/s | -0.1 % | negligible |
|  | trace | trace: 132.0 (16320 read on 0 cached) | — | -0.0 tok/s | -0.0 % | no class (one load) |
| 49,152 | base | base1: 82.8 (33284 read on 15804 cached); base2: 82.8 (33284 read on 15804 cached) | 0.0 | control | | |
|  | host | host1: 80.4 (33284 read on 15804 cached); host2: 80.4 (33284 read on 15804 cached) | 0.0 | -2.4 tok/s | -2.9 % | loss |
|  | nopf | nopf1: 83.3 (33284 read on 15804 cached); nopf2: 82.6 (33284 read on 15804 cached) | 0.7 | +0.2 tok/s | +0.2 % | negligible |
|  | trace | trace: 82.6 (33284 read on 15804 cached) | — | -0.2 tok/s | -0.2 % | no class (one load) |

### The draft and the step

Per load, pass 1: draft tokens accepted of drafted (the share), tokens a step (generated tokens over steps, a step being one pass of the main model: generated minus accepted), and the time of a step. Pass 2 is listed where its counters differ.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | base | base1: 178 of 229 (0.777), 3.28 a step, 125.3 ms a step (pass 2: 181 of 220 (0.823), 3.41 a step, 121.3 ms a step); base2: 178 of 229 (0.777), 3.28 a step, 124.6 ms a step (pass 2: 181 of 220 (0.823), 3.41 a step, 121.7 ms a step) |
|  | host | host1: 178 of 229 (0.777), 3.28 a step, 126.9 ms a step (pass 2: 181 of 220 (0.823), 3.41 a step, 123.1 ms a step); host2: 178 of 229 (0.777), 3.28 a step, 126.9 ms a step (pass 2: 181 of 220 (0.823), 3.41 a step, 122.9 ms a step) |
|  | nopf | nopf1: 178 of 229 (0.777), 3.28 a step, 126.0 ms a step (pass 2: 181 of 220 (0.823), 3.41 a step, 121.7 ms a step); nopf2: 178 of 229 (0.777), 3.28 a step, 125.4 ms a step (pass 2: 181 of 220 (0.823), 3.41 a step, 121.2 ms a step) |
|  | trace | trace: 178 of 229 (0.777), 3.28 a step, 125.6 ms a step (pass 2: 181 of 220 (0.823), 3.41 a step, 121.4 ms a step) |
| 16,384 | base | base1: 178 of 231 (0.771), 3.28 a step, 185.0 ms a step (pass 2: 179 of 228 (0.785), 3.32 a step, 180.4 ms a step); base2: 178 of 231 (0.771), 3.28 a step, 185.1 ms a step (pass 2: 179 of 228 (0.785), 3.32 a step, 180.8 ms a step) |
|  | host | host1: 178 of 231 (0.771), 3.28 a step, 214.6 ms a step (pass 2: 179 of 228 (0.785), 3.32 a step, 182.1 ms a step); host2: 178 of 231 (0.771), 3.28 a step, 221.5 ms a step (pass 2: 179 of 228 (0.785), 3.32 a step, 182.2 ms a step) |
|  | nopf | nopf1: 178 of 231 (0.771), 3.28 a step, 184.7 ms a step (pass 2: 179 of 228 (0.785), 3.32 a step, 180.8 ms a step); nopf2: 178 of 231 (0.771), 3.28 a step, 185.2 ms a step (pass 2: 179 of 228 (0.785), 3.32 a step, 180.7 ms a step) |
|  | trace | trace: 178 of 231 (0.771), 3.28 a step, 185.6 ms a step (pass 2: 179 of 228 (0.785), 3.32 a step, 181.1 ms a step) |
| 49,152 | base | base1: 181 of 219 (0.826), 3.41 a step, 318.5 ms a step; base2: 181 of 219 (0.826), 3.41 a step, 343.0 ms a step |
|  | host | host1: 181 of 219 (0.826), 3.41 a step, 346.8 ms a step; host2: 181 of 219 (0.826), 3.41 a step, 341.6 ms a step |
|  | nopf | nopf1: 181 of 219 (0.826), 3.41 a step, 317.6 ms a step; nopf2: 181 of 219 (0.826), 3.41 a step, 319.1 ms a step |
|  | trace | trace: 181 of 219 (0.826), 3.41 a step, 353.8 ms a step |

### Checks

- Times: every load started on a machine checked quiet (valid=1).
- Prompts: 64 ids at depth 0, 16384 ids at depth 16,384, 49152 ids at depth 49,152; the same ids on every load and pass.
- Tokens out: 256 on every timed request.
- Greedy reply at depth 0: the same on every pass and load of every arm.
- Greedy reply at depth 16,384: the same on every pass and load of every arm.
- Greedy reply at depth 49,152: the same on every pass and load of every arm.
- Primes (read on cached, the cache's source): 16,384: 16320 on 0 (none); 49,152: 33284 on 15804 (checkpoint).
