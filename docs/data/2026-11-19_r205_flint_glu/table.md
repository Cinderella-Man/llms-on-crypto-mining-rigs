## The deep lane: off, u1, u2

Model file `flint-near-q8-p065.gguf`, llama-server version: 1.0.0-dev (build 12967, commit 6026911533).

256 tokens per request at temperature 0, each request twice per load (pass 1 is the first request at a new depth, pass 2 the same request again). Timed loads, in their order: f1, a1, b1, f2, a2, b2.

| Arm | Loads | Draft | Context | Its own variables and flags |
|---|---|---|---:|---|
| off (control) | f1, f2 | off | 262144 | — |
| u1 | a1, a2 | off | 262144 | `GGML_CUDA_MMID_GLU_UNFUSE=1` |
| u2 | b1, b2 | off | 262144 | `GGML_CUDA_MMID_GLU_UNFUSE=2` |

### Generation, tok/s

Per load: pass 1, pass 2 → the load, with the load's time per token. A load's speed is its two passes' tokens over their time. Spread: the difference between the arm's fastest and slowest load. Class against `off`: gain when the mean is above the control's by more than the larger of the two spreads, loss when below by more than it, else negligible.

| Depth (tokens) | Arm | Per load | Spread | Against `off` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | off | f1: 46.80, 55.29 → **50.69** (19.73 ms a token); f2: 46.27, 54.96 → **50.24** (19.90 ms a token) | 0.45 | control | | |
|  | u1 | a1: 46.98, 54.93 → **50.64** (19.75 ms a token); a2: 37.47, 54.95 → **44.56** (22.44 ms a token) | 6.09 | -2.87 tok/s | -5.7 % | negligible |
|  | u2 | b1: 47.24, 55.51 → **51.04** (19.59 ms a token); b2: 47.17, 55.23 → **50.88** (19.65 ms a token) | 0.15 | +0.49 tok/s | +1.0 % | gain |
| 8,192 | off | f1: 47.04, 51.41 → **49.13** (20.36 ms a token); f2: 47.60, 51.35 → **49.40** (20.24 ms a token) | 0.28 | control | | |
|  | u1 | a1: 46.73, 51.16 → **48.84** (20.47 ms a token); a2: 45.79, 51.23 → **48.36** (20.68 ms a token) | 0.49 | -0.66 tok/s | -1.3 % | loss |
|  | u2 | b1: 46.39, 51.71 → **48.91** (20.45 ms a token); b2: 47.00, 51.79 → **49.28** (20.29 ms a token) | 0.37 | -0.17 tok/s | -0.4 % | negligible |

### Generation over all depths, tok/s

Every depth's tokens over their time, per load.

| Depth (tokens) | Arm | Per load | Spread | Against `off` | | Class |
|---:|---|---|---:|---:|---:|---|
| all | off | f1: **49.90**; f2: **49.82** | 0.08 | control | | |
|  | u1 | a1: **49.73**; a2: **46.38** | 3.35 | -1.80 tok/s | -3.6 % | negligible |
|  | u2 | b1: **49.95**; b2: **50.07** | 0.12 | +0.15 tok/s | +0.3 % | gain |

### Prompt reading, tok/s

Per load: the tokens read at that depth over their time (a prime request reads a chunk on top of the cache; without primes each decode request reads its prompt), with the tokens read and the tokens reused. Requests that read under 32 tokens are left out.

| Depth (tokens) | Arm | Per load | Spread | Against `off` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | off | f1: 87.2 (64 read on 0 cached); f2: 88.7 (64 read on 0 cached) | 1.5 | control | | |
|  | u1 | a1: 91.5 (64 read on 0 cached); a2: 86.8 (64 read on 0 cached) | 4.7 | +1.2 tok/s | +1.4 % | negligible |
|  | u2 | b1: 89.4 (64 read on 0 cached); b2: 88.8 (64 read on 0 cached) | 0.6 | +1.2 tok/s | +1.3 % | negligible |
| 8,192 | off | f1: 593.3 (8128 read on 0 cached); f2: 592.1 (8128 read on 0 cached) | 1.3 | control | | |
|  | u1 | a1: 592.7 (8128 read on 0 cached); a2: 445.4 (8128 read on 0 cached) | 147.3 | -73.7 tok/s | -12.4 % | negligible |
|  | u2 | b1: 593.4 (8128 read on 0 cached); b2: 594.3 (8128 read on 0 cached) | 0.9 | +1.2 tok/s | +0.2 % | negligible |

### The draft and the step

Per load, pass 1: draft tokens accepted of drafted (the share), tokens a step (generated tokens over steps, a step being one pass of the main model: generated minus accepted), and the time of a step. Pass 2 is listed where its counters differ.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | off | f1: no draft, 1.00 a step, 21.4 ms a step; f2: no draft, 1.00 a step, 21.6 ms a step |
|  | u1 | a1: no draft, 1.00 a step, 21.3 ms a step; a2: no draft, 1.00 a step, 26.7 ms a step |
|  | u2 | b1: no draft, 1.00 a step, 21.2 ms a step; b2: no draft, 1.00 a step, 21.2 ms a step |
| 8,192 | off | f1: no draft, 1.00 a step, 21.3 ms a step; f2: no draft, 1.00 a step, 21.0 ms a step |
|  | u1 | a1: no draft, 1.00 a step, 21.4 ms a step; a2: no draft, 1.00 a step, 21.8 ms a step |
|  | u2 | b1: no draft, 1.00 a step, 21.6 ms a step; b2: no draft, 1.00 a step, 21.3 ms a step |

### The sampler pass

One request per depth and load with the server line's own sampler and a fixed seed: tok/s, the draft counters and the step.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | off | f1: 60.53 tok/s, no draft, 1.00 a step, 16.5 ms a step; f2: 60.24 tok/s, no draft, 1.00 a step, 16.6 ms a step |
|  | u1 | a1: 60.04 tok/s, no draft, 1.00 a step, 16.7 ms a step; a2: 53.16 tok/s, no draft, 1.00 a step, 18.8 ms a step |
|  | u2 | b1: 60.16 tok/s, no draft, 1.00 a step, 16.6 ms a step; b2: 61.69 tok/s, no draft, 1.00 a step, 16.2 ms a step |
| 8,192 | off | f1: 43.29 tok/s, no draft, 1.00 a step, 23.1 ms a step; f2: 43.21 tok/s, no draft, 1.00 a step, 23.1 ms a step |
|  | u1 | a1: 43.16 tok/s, no draft, 1.00 a step, 23.2 ms a step; a2: 43.13 tok/s, no draft, 1.00 a step, 23.2 ms a step |
|  | u2 | b1: 43.59 tok/s, no draft, 1.00 a step, 22.9 ms a step; b2: 43.66 tok/s, no draft, 1.00 a step, 22.9 ms a step |

### Checks

- Times: every load started on a machine checked quiet (valid=1).
- Prompts: 64 ids at depth 0, 8192 ids at depth 8,192; the same ids on every load and pass.
- Tokens out: 256 on every timed request.
- Greedy reply at depth 0: the same on every pass and load of every arm.
- Greedy reply at depth 8,192: the same on every pass and load of every arm.
- Primes (read on cached, the cache's source): 8,192: 8128 on 0 (none).
- Graph reallocations of the target, requests with one per load: f1: 0 of 8 requests; a1: 0 of 8 requests; b1: 0 of 8 requests; f2: 0 of 8 requests; a2: 0 of 8 requests; b2: 0 of 8 requests.
