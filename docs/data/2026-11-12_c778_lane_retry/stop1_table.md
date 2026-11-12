## The deep lane: base, host

Model file `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf`.

64 tokens per request at temperature 0, each request twice per load (pass 1 is the first request at a new depth, pass 2 the same request again). Timed loads, in their order: base1, host2, host1, base2.

| Arm | Loads | Draft | Context | Its own variables and flags |
|---|---|---|---:|---|
| base (control) | base1, base2 | on | 65536 | — |
| host | host2, host1 | on | 65536 | `--decode-ms 40` |

### Generation, tok/s

Per load: pass 1, pass 2 → the load, with the load's time per token. A load's speed is its two passes' tokens over their time. Spread: the difference between the arm's fastest and slowest load. Class against `base`: gain when the mean is above the control's by more than the larger of the two spreads, loss when below by more than it, else negligible.

| Depth (tokens) | Arm | Per load | Spread | Against `base` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | base | base1: 9540.85, 9540.85 → **9540.85** (0.10 ms a token); base2: 9540.85, 9540.85 → **9540.85** (0.10 ms a token) | 0.00 | control | | |
|  | host | host2: 49.22, 49.22 → **49.22** (20.32 ms a token); host1: 49.22, 49.22 → **49.22** (20.32 ms a token) | 0.00 | -9491.62 tok/s | -99.5 % | loss |
| 512 | base | base1: 8520.84, 8520.84 → **8520.84** (0.12 ms a token); base2: 8520.84, 8520.84 → **8520.84** (0.12 ms a token) | 0.00 | control | | |
|  | host | host2: 53.44, 53.44 → **53.44** (18.71 ms a token); host1: 53.44, 53.44 → **53.44** (18.71 ms a token) | 0.00 | -8467.39 tok/s | -99.4 % | loss |
| 1,024 | base | base1: 6512.01, 6512.01 → **6512.01** (0.15 ms a token); base2: 6512.01, 6512.01 → **6512.01** (0.15 ms a token) | 0.00 | control | | |
|  | host | host2: 49.11, 49.11 → **49.11** (20.36 ms a token); host1: 49.11, 49.11 → **49.11** (20.36 ms a token) | 0.00 | -6462.90 tok/s | -99.2 % | loss |
| 2,048 | base | base1: 5528.20, 5528.20 → **5528.20** (0.18 ms a token); base2: 5528.20, 5528.20 → **5528.20** (0.18 ms a token) | 0.00 | control | | |
|  | host | host2: 55.66, 55.66 → **55.66** (17.97 ms a token); host1: 55.66, 55.66 → **55.66** (17.97 ms a token) | 0.00 | -5472.54 tok/s | -99.0 % | loss |

### Generation over all depths, tok/s

Every depth's tokens over their time, per load.

| Depth (tokens) | Arm | Per load | Spread | Against `base` | | Class |
|---:|---|---|---:|---:|---:|---|
| all | base | base1: **7186.17**; base2: **7186.17** | 0.00 | control | | |
|  | host | host2: **51.71**; host1: **51.71** | 0.00 | -7134.46 tok/s | -99.3 % | loss |

### The step on pass 1 (the first request at a new depth), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `base`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `base` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | base | base1: 0.28 (24 steps); base2: 0.28 (24 steps) | 0.00 | control | | |
|  | host | host2: 54.18 (24 steps); host1: 54.18 (24 steps) | 0.00 | +53.90 ms | +19282.9 % | loss |
| 512 | base | base1: 0.33 (23 steps); base2: 0.33 (23 steps) | 0.00 | control | | |
|  | host | host2: 52.07 (23 steps); host1: 52.07 (23 steps) | 0.00 | +51.74 ms | +15843.7 % | loss |
| 1,024 | base | base1: 0.39 (25 steps); base2: 0.39 (25 steps) | 0.00 | control | | |
|  | host | host2: 52.13 (25 steps); host1: 52.13 (25 steps) | 0.00 | +51.74 ms | +13161.4 % | loss |
| 2,048 | base | base1: 0.53 (22 steps); base2: 0.53 (22 steps) | 0.00 | control | | |
|  | host | host2: 52.27 (22 steps); host1: 52.27 (22 steps) | 0.00 | +51.74 ms | +9832.3 % | loss |

### The step on pass 2 (the same request again), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `base`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `base` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | base | base1: 0.28 (24 steps); base2: 0.28 (24 steps) | 0.00 | control | | |
|  | host | host2: 54.18 (24 steps); host1: 54.18 (24 steps) | 0.00 | +53.90 ms | +19282.9 % | loss |
| 512 | base | base1: 0.33 (23 steps); base2: 0.33 (23 steps) | 0.00 | control | | |
|  | host | host2: 52.07 (23 steps); host1: 52.07 (23 steps) | 0.00 | +51.74 ms | +15843.7 % | loss |
| 1,024 | base | base1: 0.39 (25 steps); base2: 0.39 (25 steps) | 0.00 | control | | |
|  | host | host2: 52.13 (25 steps); host1: 52.13 (25 steps) | 0.00 | +51.74 ms | +13161.4 % | loss |
| 2,048 | base | base1: 0.53 (22 steps); base2: 0.53 (22 steps) | 0.00 | control | | |
|  | host | host2: 52.27 (22 steps); host1: 52.27 (22 steps) | 0.00 | +51.74 ms | +9832.3 % | loss |

### Prompt reading, tok/s

Per load: the tokens read at that depth over their time (a prime request reads a chunk on top of the cache; without primes each decode request reads its prompt), with the tokens read and the tokens reused. Requests that read under 32 tokens are left out.

| Depth (tokens) | Arm | Per load | Spread | Against `base` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | base | base1: 500000.0 (64 read on 0 cached); base2: 500000.0 (64 read on 0 cached) | 0.0 | control | | |
|  | host | host2: 500000.0 (64 read on 0 cached); host1: 500000.0 (64 read on 0 cached) | 0.0 | +0.0 tok/s | +0.0 % | negligible |
| 512 | base | base1: 500000.0 (448 read on 0 cached); base2: 500000.0 (448 read on 0 cached) | 0.0 | control | | |
|  | host | host2: 500000.0 (448 read on 0 cached); host1: 500000.0 (448 read on 0 cached) | 0.0 | +0.0 tok/s | +0.0 % | negligible |
| 1,024 | base | base1: 500000.0 (516 read on 444 cached); base2: 500000.0 (516 read on 444 cached) | 0.0 | control | | |
|  | host | host2: 500000.0 (516 read on 444 cached); host1: 500000.0 (516 read on 444 cached) | 0.0 | +0.0 tok/s | +0.0 % | negligible |
| 2,048 | base | base1: 500000.0 (1028 read on 956 cached); base2: 500000.0 (1028 read on 956 cached) | 0.0 | control | | |
|  | host | host2: 500000.0 (1028 read on 956 cached); host1: 500000.0 (1028 read on 956 cached) | 0.0 | +0.0 tok/s | +0.0 % | negligible |

### The draft and the step

Per load, pass 1: draft tokens accepted of drafted (the share), tokens a step (generated tokens over steps, a step being one pass of the main model: generated minus accepted), and the time of a step. Pass 2 is listed where its counters differ.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | base | base1: 40 of 75 (0.533), 2.67 a step, 0.3 ms a step; base2: 40 of 75 (0.533), 2.67 a step, 0.3 ms a step |
|  | host | host2: 40 of 75 (0.533), 2.67 a step, 54.2 ms a step; host1: 40 of 75 (0.533), 2.67 a step, 54.2 ms a step |
| 512 | base | base1: 41 of 69 (0.594), 2.78 a step, 0.3 ms a step; base2: 41 of 69 (0.594), 2.78 a step, 0.3 ms a step |
|  | host | host2: 41 of 69 (0.594), 2.78 a step, 52.1 ms a step; host1: 41 of 69 (0.594), 2.78 a step, 52.1 ms a step |
| 1,024 | base | base1: 39 of 75 (0.520), 2.56 a step, 0.4 ms a step; base2: 39 of 75 (0.520), 2.56 a step, 0.4 ms a step |
|  | host | host2: 39 of 75 (0.520), 2.56 a step, 52.1 ms a step; host1: 39 of 75 (0.520), 2.56 a step, 52.1 ms a step |
| 2,048 | base | base1: 42 of 66 (0.636), 2.91 a step, 0.5 ms a step; base2: 42 of 66 (0.636), 2.91 a step, 0.5 ms a step |
|  | host | host2: 42 of 66 (0.636), 2.91 a step, 52.3 ms a step; host1: 42 of 66 (0.636), 2.91 a step, 52.3 ms a step |

### Checks

- Times: every load started on a machine checked quiet (valid=1).
- Prompts: 64 ids at depth 0, 512 ids at depth 512, 1024 ids at depth 1,024, 2048 ids at depth 2,048; the same ids on every load and pass.
- Tokens out: 64 on every timed request.
- Greedy reply at depth 0: the same on every pass and load of every arm.
- Greedy reply at depth 512: the same on every pass and load of every arm.
- Greedy reply at depth 1,024: the same on every pass and load of every arm.
- Greedy reply at depth 2,048: the same on every pass and load of every arm.
- Primes (read on cached, the cache's source): 512: 448 on 0 (none); 1,024: 516 on 444 (checkpoint); 2,048: 1028 on 956 (checkpoint).
- Graph reallocations: no `graph reallocs` line in the server logs (no logs here, or a build that does not print the count).
- Stopped by the RAM watchdog (2 stops, each load's files set aside): `host1` (host) after decode pass 1 at depth 512 (MemAvailable 50 MiB, below 60), `stopped/host1-1`, run again; `base2` (base) while loading (MemAvailable 50 MiB, below 60), `stopped/base2-1`, run again.
