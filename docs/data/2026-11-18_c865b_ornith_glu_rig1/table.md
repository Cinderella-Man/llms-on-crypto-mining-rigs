## The deep lane: off, u1, u2

Model file `Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf`, llama-server version: 1.0.0-dev (build 12892, commit 1f1110c673).

256 tokens per request at temperature 0, each request twice per load (pass 1 is the first request at a new depth, pass 2 the same request again). Timed loads, in their order: f1, a1, b1, f2, a2, b2.

| Arm | Loads | Draft | Context | Its own variables and flags |
|---|---|---|---:|---|
| off (control) | f1, f2 | off | 49152 | — |
| u1 | a1, a2 | off | 49152 | `GGML_CUDA_MMID_GLU_UNFUSE=1` |
| u2 | b1, b2 | off | 49152 | `GGML_CUDA_MMID_GLU_UNFUSE=2` |

### Generation, tok/s

Per load: pass 1, pass 2 → the load, with the load's time per token. A load's speed is its two passes' tokens over their time. Spread: the difference between the arm's fastest and slowest load. Class against `off`: gain when the mean is above the control's by more than the larger of the two spreads, loss when below by more than it, else negligible.

| Depth (tokens) | Arm | Per load | Spread | Against `off` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | off | f1: 17.79, 18.64 → **18.20** (54.93 ms a token); f2: 16.41, 18.62 → **17.44** (57.33 ms a token) | 0.76 | control | | |
|  | u1 | a1: 18.82, 19.82 → **19.31** (51.79 ms a token); a2: 18.62, 19.82 → **19.20** (52.08 ms a token) | 0.11 | +1.43 tok/s | +8.0 % | gain |
|  | u2 | b1: 18.01, 20.00 → **18.95** (52.77 ms a token); b2: 18.75, 19.98 → **19.35** (51.69 ms a token) | 0.40 | +1.33 tok/s | +7.4 % | gain |
| 8,192 | off | f1: 15.43, 15.62 → **15.52** (64.42 ms a token); f2: 15.42, 15.59 → **15.51** (64.49 ms a token) | 0.02 | control | | |
|  | u1 | a1: 16.23, 16.44 → **16.33** (61.23 ms a token); a2: 16.22, 16.41 → **16.31** (61.30 ms a token) | 0.02 | +0.81 tok/s | +5.2 % | gain |
|  | u2 | b1: 16.50, 16.62 → **16.56** (60.38 ms a token); b2: 16.40, 16.60 → **16.50** (60.60 ms a token) | 0.06 | +1.02 tok/s | +6.5 % | gain |

### Generation over all depths, tok/s

Every depth's tokens over their time, per load.

| Depth (tokens) | Arm | Per load | Spread | Against `off` | | Class |
|---:|---|---|---:|---:|---:|---|
| all | off | f1: **16.76**; f2: **16.42** | 0.34 | control | | |
|  | u1 | a1: **17.70**; a2: **17.64** | 0.06 | +1.08 tok/s | +6.5 % | gain |
|  | u2 | b1: **17.68**; b2: **17.81** | 0.13 | +1.16 tok/s | +7.0 % | gain |

### Prompt reading, tok/s

Per load: the tokens read at that depth over their time (a prime request reads a chunk on top of the cache; without primes each decode request reads its prompt), with the tokens read and the tokens reused. Requests that read under 32 tokens are left out.

| Depth (tokens) | Arm | Per load | Spread | Against `off` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | off | f1: 39.3 (64 read on 0 cached); f2: 39.2 (64 read on 0 cached) | 0.0 | control | | |
|  | u1 | a1: 39.3 (64 read on 0 cached); a2: 39.3 (64 read on 0 cached) | 0.1 | +0.0 tok/s | +0.1 % | negligible |
|  | u2 | b1: 39.2 (64 read on 0 cached); b2: 39.3 (64 read on 0 cached) | 0.1 | +0.0 tok/s | +0.1 % | negligible |
| 8,192 | off | f1: 217.0 (8128 read on 0 cached); f2: 216.6 (8128 read on 0 cached) | 0.4 | control | | |
|  | u1 | a1: 215.3 (8128 read on 0 cached); a2: 213.6 (8128 read on 0 cached) | 1.8 | -2.3 tok/s | -1.1 % | loss |
|  | u2 | b1: 215.4 (8128 read on 0 cached); b2: 216.2 (8128 read on 0 cached) | 0.8 | -1.0 tok/s | -0.5 % | loss |

### The draft and the step

Per load, pass 1: draft tokens accepted of drafted (the share), tokens a step (generated tokens over steps, a step being one pass of the main model: generated minus accepted), and the time of a step. Pass 2 is listed where its counters differ.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | off | f1: no draft, 1.00 a step, 56.2 ms a step; f2: no draft, 1.00 a step, 61.0 ms a step |
|  | u1 | a1: no draft, 1.00 a step, 53.1 ms a step; a2: no draft, 1.00 a step, 53.7 ms a step |
|  | u2 | b1: no draft, 1.00 a step, 55.5 ms a step; b2: no draft, 1.00 a step, 53.3 ms a step |
| 8,192 | off | f1: no draft, 1.00 a step, 64.8 ms a step; f2: no draft, 1.00 a step, 64.8 ms a step |
|  | u1 | a1: no draft, 1.00 a step, 61.6 ms a step; a2: no draft, 1.00 a step, 61.7 ms a step |
|  | u2 | b1: no draft, 1.00 a step, 60.6 ms a step; b2: no draft, 1.00 a step, 61.0 ms a step |

### The sampler pass

One request per depth and load with the server line's own sampler and a fixed seed: tok/s, the draft counters and the step.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | off | f1: 17.75 tok/s, no draft, 1.00 a step, 56.3 ms a step; f2: 16.52 tok/s, no draft, 1.00 a step, 60.5 ms a step |
|  | u1 | a1: 19.19 tok/s, no draft, 1.00 a step, 52.1 ms a step; a2: 17.82 tok/s, no draft, 1.00 a step, 56.1 ms a step |
|  | u2 | b1: 17.50 tok/s, no draft, 1.00 a step, 57.1 ms a step; b2: 18.98 tok/s, no draft, 1.00 a step, 52.7 ms a step |
| 8,192 | off | f1: 15.56 tok/s, no draft, 1.00 a step, 64.3 ms a step; f2: 15.53 tok/s, no draft, 1.00 a step, 64.4 ms a step |
|  | u1 | a1: 16.38 tok/s, no draft, 1.00 a step, 61.1 ms a step; a2: 16.37 tok/s, no draft, 1.00 a step, 61.1 ms a step |
|  | u2 | b1: 16.52 tok/s, no draft, 1.00 a step, 60.5 ms a step; b2: 16.48 tok/s, no draft, 1.00 a step, 60.7 ms a step |

### Checks

- Times: every load started on a machine checked quiet (valid=1).
- Prompts: 64 ids at depth 0, 8192 ids at depth 8,192; the same ids on every load and pass.
- Tokens out: 256 on every timed request.
- Greedy reply at depth 0: the same on every pass and load of every arm.
- Greedy reply at depth 8,192: the same on every pass and load of every arm.
- Primes (read on cached, the cache's source): 8,192: 8128 on 0 (none).
- Graph reallocations of the target, requests with one per load: f1: 0 of 8 requests; a1: 0 of 8 requests; b1: 0 of 8 requests; f2: 0 of 8 requests; a2: 0 of 8 requests; b2: 0 of 8 requests.
