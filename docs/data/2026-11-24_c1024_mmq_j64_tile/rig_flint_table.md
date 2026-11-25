## The deep lane: unset, j64

Model file `flint-near-q8-p065.gguf`, llama-server version: 1.0.0-dev (build 13219, commit ebec6544e3).

256 tokens per request at temperature 0, each request twice per load (pass 1 is the first request at a new depth, pass 2 the same request again). Timed loads, in their order: unset1, j641, unset2, j642.

| Arm | Loads | Draft | Context | Its own variables and flags |
|---|---|---|---:|---|
| unset (control) | unset1, unset2 | off | 262144 | — |
| j64 | j641, j642 | off | 262144 | `GGML_CUDA_MMQ_J64_I64=1` |

### Generation, tok/s

Per load: pass 1, pass 2 → the load, with the load's time per token. A load's speed is its two passes' tokens over their time. Spread: the difference between the arm's fastest and slowest load. Class against `unset`: gain when the mean is above the control's by more than the larger of the two spreads, loss when below by more than it, else negligible.

| Depth (tokens) | Arm | Per load | Spread | Against `unset` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | unset | unset1: 47.77, 56.56 → **51.80** (19.31 ms a token); unset2: 47.92, 56.46 → **51.84** (19.29 ms a token) | 0.04 | control | | |
|  | j64 | j641: 47.74, 56.57 → **51.78** (19.31 ms a token); j642: 42.06, 56.64 → **48.27** (20.71 ms a token) | 3.50 | -1.79 tok/s | -3.5 % | negligible |
| 4,096 | unset | unset1: 76.50, 82.02 → **79.16** (12.63 ms a token); unset2: 76.44, 82.06 → **79.15** (12.63 ms a token) | 0.01 | control | | |
|  | j64 | j641: 76.58, 82.13 → **79.26** (12.62 ms a token); j642: 76.49, 82.09 → **79.19** (12.63 ms a token) | 0.06 | +0.07 tok/s | +0.1 % | gain |
| 8,192 | unset | unset1: 47.89, 52.34 → **50.01** (20.00 ms a token); unset2: 48.02, 52.28 → **50.06** (19.98 ms a token) | 0.05 | control | | |
|  | j64 | j641: 47.96, 52.36 → **50.06** (19.98 ms a token); j642: 49.40, 52.32 → **50.82** (19.68 ms a token) | 0.75 | +0.40 tok/s | +0.8 % | negligible |

### Generation over all depths, tok/s

Every depth's tokens over their time, per load.

| Depth (tokens) | Arm | Per load | Spread | Against `unset` | | Class |
|---:|---|---|---:|---:|---:|---|
| all | unset | unset1: **57.77**; unset2: **57.80** | 0.04 | control | | |
|  | j64 | j641: **57.80**; j642: **56.58** | 1.22 | -0.60 tok/s | -1.0 % | negligible |

### Prompt reading, tok/s

Per load: the tokens read at that depth over their time (a prime request reads a chunk on top of the cache; without primes each decode request reads its prompt), with the tokens read and the tokens reused. Requests that read under 32 tokens are left out.

| Depth (tokens) | Arm | Per load | Spread | Against `unset` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | unset | unset1: 90.5 (64 read on 0 cached); unset2: 89.9 (64 read on 0 cached) | 0.6 | control | | |
|  | j64 | j641: 87.7 (64 read on 0 cached); j642: 70.0 (64 read on 0 cached) | 17.7 | -11.3 tok/s | -12.6 % | negligible |
| 4,096 | unset | unset1: 560.6 (4032 read on 0 cached); unset2: 567.3 (4032 read on 0 cached) | 6.7 | control | | |
|  | j64 | j641: 563.2 (4032 read on 0 cached); j642: 433.5 (4032 read on 0 cached) | 129.8 | -65.6 tok/s | -11.6 % | negligible |
| 8,192 | unset | unset1: 545.9 (4100 read on 4028 cached); unset2: 546.4 (4100 read on 4028 cached) | 0.5 | control | | |
|  | j64 | j641: 532.4 (4100 read on 4028 cached); j642: 494.4 (4100 read on 4028 cached) | 38.0 | -32.8 tok/s | -6.0 % | negligible |

### The draft and the step

Per load, pass 1: draft tokens accepted of drafted (the share), tokens a step (generated tokens over steps, a step being one pass of the main model: generated minus accepted), and the time of a step. Pass 2 is listed where its counters differ.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | unset | unset1: no draft, 1.00 a step, 20.9 ms a step; unset2: no draft, 1.00 a step, 20.9 ms a step |
|  | j64 | j641: no draft, 1.00 a step, 20.9 ms a step; j642: no draft, 1.00 a step, 23.8 ms a step |
| 4,096 | unset | unset1: no draft, 1.00 a step, 13.1 ms a step; unset2: no draft, 1.00 a step, 13.1 ms a step |
|  | j64 | j641: no draft, 1.00 a step, 13.1 ms a step; j642: no draft, 1.00 a step, 13.1 ms a step |
| 8,192 | unset | unset1: no draft, 1.00 a step, 20.9 ms a step; unset2: no draft, 1.00 a step, 20.8 ms a step |
|  | j64 | j641: no draft, 1.00 a step, 20.9 ms a step; j642: no draft, 1.00 a step, 20.2 ms a step |

### The sampler pass

One request per depth and load with the server line's own sampler and a fixed seed: tok/s, the draft counters and the step.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | unset | unset1: 62.85 tok/s, no draft, 1.00 a step, 15.9 ms a step; unset2: 62.32 tok/s, no draft, 1.00 a step, 16.0 ms a step |
|  | j64 | j641: 62.03 tok/s, no draft, 1.00 a step, 16.1 ms a step; j642: 60.34 tok/s, no draft, 1.00 a step, 16.6 ms a step |
| 4,096 | unset | unset1: 67.79 tok/s, no draft, 1.00 a step, 14.8 ms a step; unset2: 67.76 tok/s, no draft, 1.00 a step, 14.8 ms a step |
|  | j64 | j641: 67.80 tok/s, no draft, 1.00 a step, 14.7 ms a step; j642: 67.85 tok/s, no draft, 1.00 a step, 14.7 ms a step |
| 8,192 | unset | unset1: 44.21 tok/s, no draft, 1.00 a step, 22.6 ms a step; unset2: 43.84 tok/s, no draft, 1.00 a step, 22.8 ms a step |
|  | j64 | j641: 43.57 tok/s, no draft, 1.00 a step, 23.0 ms a step; j642: 43.46 tok/s, no draft, 1.00 a step, 23.0 ms a step |

### Checks

- Times: every load started on a machine checked quiet (valid=1).
- Prompts: 64 ids at depth 0, 4096 ids at depth 4,096, 8192 ids at depth 8,192; the same ids on every load and pass.
- Tokens out: 256 on every timed request.
- Greedy reply at depth 0: the same on every pass and load of every arm.
- Greedy reply at depth 4,096: the same on every pass and load of every arm.
- Greedy reply at depth 8,192: the same on every pass and load of every arm.
- Primes (read on cached, the cache's source): 4,096: 4032 on 0 (none); 8,192: 4100 on 4028 (checkpoint).
- Graph reallocations of the target, requests with one per load: unset1: 0 of 12 requests; j641: 0 of 12 requests; unset2: 0 of 12 requests; j642: 0 of 12 requests.
