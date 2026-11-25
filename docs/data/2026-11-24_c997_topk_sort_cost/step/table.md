## The deep lane: unset, sort

Model file `flint-near-q8-p065.gguf`, llama-server version: 1.0.0-dev (build 13218, commit eee615cdb9).

256 tokens per request at temperature 0, each request twice per load (pass 1 is the first request at a new depth, pass 2 the same request again). Timed loads, in their order: unset1, sort1, unset2, sort2.

| Arm | Loads | Draft | Context | Its own variables and flags |
|---|---|---|---:|---|
| unset (control) | unset1, unset2 | off | 262144 | — |
| sort | sort1, sort2 | off | 262144 | `GGML_CUDA_TOP_K_SORT=1` |

### Generation, tok/s

Per load: pass 1, pass 2 → the load, with the load's time per token. A load's speed is its two passes' tokens over their time. Spread: the difference between the arm's fastest and slowest load. Class against `unset`: gain when the mean is above the control's by more than the larger of the two spreads, loss when below by more than it, else negligible.

| Depth (tokens) | Arm | Per load | Spread | Against `unset` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | unset | unset1: 47.59, 53.48 → **50.36** (19.86 ms a token); unset2: 47.63, 55.11 → **51.10** (19.57 ms a token) | 0.74 | control | | |
|  | sort | sort1: 47.65, 54.93 → **51.03** (19.60 ms a token); sort2: 47.48, 55.45 → **51.16** (19.55 ms a token) | 0.12 | +0.36 tok/s | +0.7 % | negligible |
| 8,192 | unset | unset1: 47.69, 52.18 → **49.83** (20.07 ms a token); unset2: 44.12, 51.94 → **47.71** (20.96 ms a token) | 2.12 | control | | |
|  | sort | sort1: 46.98, 52.14 → **49.42** (20.23 ms a token); sort2: 45.47, 52.00 → **48.51** (20.61 ms a token) | 0.91 | +0.20 tok/s | +0.4 % | negligible |

### Generation over all depths, tok/s

Every depth's tokens over their time, per load.

| Depth (tokens) | Arm | Per load | Spread | Against `unset` | | Class |
|---:|---|---|---:|---:|---:|---|
| all | unset | unset1: **50.10**; unset2: **49.35** | 0.75 | control | | |
|  | sort | sort1: **50.22**; sort2: **49.80** | 0.42 | +0.28 tok/s | +0.6 % | negligible |

### Prompt reading, tok/s

Per load: the tokens read at that depth over their time (a prime request reads a chunk on top of the cache; without primes each decode request reads its prompt), with the tokens read and the tokens reused. Requests that read under 32 tokens are left out.

| Depth (tokens) | Arm | Per load | Spread | Against `unset` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | unset | unset1: 78.8 (64 read on 0 cached); unset2: 73.3 (64 read on 0 cached) | 5.5 | control | | |
|  | sort | sort1: 75.9 (64 read on 0 cached); sort2: 87.1 (64 read on 0 cached) | 11.2 | +5.4 tok/s | +7.2 % | negligible |
| 8,192 | unset | unset1: 590.1 (8128 read on 0 cached); unset2: 572.8 (8128 read on 0 cached) | 17.3 | control | | |
|  | sort | sort1: 576.0 (8128 read on 0 cached); sort2: 535.0 (8128 read on 0 cached) | 40.9 | -26.0 tok/s | -4.5 % | negligible |

### The draft and the step

Per load, pass 1: draft tokens accepted of drafted (the share), tokens a step (generated tokens over steps, a step being one pass of the main model: generated minus accepted), and the time of a step. Pass 2 is listed where its counters differ.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | unset | unset1: no draft, 1.00 a step, 21.0 ms a step; unset2: no draft, 1.00 a step, 21.0 ms a step |
|  | sort | sort1: no draft, 1.00 a step, 21.0 ms a step; sort2: no draft, 1.00 a step, 21.1 ms a step |
| 8,192 | unset | unset1: no draft, 1.00 a step, 21.0 ms a step; unset2: no draft, 1.00 a step, 22.7 ms a step |
|  | sort | sort1: no draft, 1.00 a step, 21.3 ms a step; sort2: no draft, 1.00 a step, 22.0 ms a step |

### The sampler pass

One request per depth and load with the server line's own sampler and a fixed seed: tok/s, the draft counters and the step.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | unset | unset1: 62.77 tok/s, no draft, 1.00 a step, 15.9 ms a step; unset2: 60.64 tok/s, no draft, 1.00 a step, 16.5 ms a step |
|  | sort | sort1: 57.14 tok/s, no draft, 1.00 a step, 17.5 ms a step; sort2: 60.88 tok/s, no draft, 1.00 a step, 16.4 ms a step |
| 8,192 | unset | unset1: 43.76 tok/s, no draft, 1.00 a step, 22.9 ms a step; unset2: 44.05 tok/s, no draft, 1.00 a step, 22.7 ms a step |
|  | sort | sort1: 44.08 tok/s, no draft, 1.00 a step, 22.7 ms a step; sort2: 44.18 tok/s, no draft, 1.00 a step, 22.6 ms a step |

### Checks

- Times: every load started on a machine checked quiet (valid=1).
- Prompts: 64 ids at depth 0, 8192 ids at depth 8,192; the same ids on every load and pass.
- Tokens out: 256 on every timed request.
- Greedy reply at depth 0: the same on every pass and load of every arm.
- Greedy reply at depth 8,192: the same on every pass and load of every arm.
- Primes (read on cached, the cache's source): 8,192: 8128 on 0 (none).
- Graph reallocations of the target, requests with one per load: unset1: 0 of 8 requests; sort1: 0 of 8 requests; unset2: 0 of 8 requests; sort2: 0 of 8 requests.
