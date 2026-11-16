## The deep lane: o4, o4nk, o4ne

Model file `Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf`, llama-server version: 1.0.0-dev (build 12832, commit eb312b7bce).

256 tokens per request at temperature 0, each request twice per load (pass 1 is the first request at a new depth, pass 2 the same request again). Timed loads, in their order: o1, n1, e1, o2, n2, e2.

| Arm | Loads | Draft | Context | Its own variables and flags |
|---|---|---|---:|---|
| o4 (control) | o1, o2 | off | 49152 | `GGML_CUDA_KEEPALIVE=decode GGML_CUDA_KEEPALIVE_EAGER=1` |
| o4nk | n1, n2 | off | 49152 | — |
| o4ne | e1, e2 | off | 49152 | `GGML_CUDA_KEEPALIVE=decode` |

### Generation, tok/s

Per load: pass 1, pass 2 → the load, with the load's time per token. A load's speed is its pass 2. Spread: the difference between the arm's fastest and slowest load. Class against `o4`: gain when the mean is above the control's by more than the larger of the two spreads, loss when below by more than it, else negligible.

| Depth (tokens) | Arm | Per load | Spread | Against `o4` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | o4 | o1: 18.34, 18.69 → **18.69** (53.50 ms a token); o2: 18.28, 18.71 → **18.71** (53.45 ms a token) | 0.01 | control | | |
|  | o4nk | n1: 17.11, 17.24 → **17.24** (57.99 ms a token); n2: 17.18, 17.37 → **17.37** (57.57 ms a token) | 0.13 | -1.39 tok/s | -7.4 % | loss |
|  | o4ne | e1: 18.36, 18.70 → **18.70** (53.48 ms a token); e2: 18.35, 18.71 → **18.71** (53.46 ms a token) | 0.01 | +0.00 tok/s | +0.0 % | negligible |
| 4,096 | o4 | o1: 16.93, 17.03 → **17.03** (58.71 ms a token); o2: 16.92, 17.04 → **17.04** (58.68 ms a token) | 0.01 | control | | |
|  | o4nk | n1: 16.88, 14.14 → **14.14** (70.70 ms a token); n2: 17.39, 14.59 → **14.59** (68.53 ms a token) | 0.45 | -2.67 tok/s | -15.7 % | loss |
|  | o4ne | e1: 16.97, 17.05 → **17.05** (58.66 ms a token); e2: 16.99, 17.04 → **17.04** (58.68 ms a token) | 0.01 | +0.01 tok/s | +0.0 % | negligible |
| 8,192 | o4 | o1: 15.56, 15.63 → **15.63** (63.98 ms a token); o2: 15.58, 15.65 → **15.65** (63.89 ms a token) | 0.02 | control | | |
|  | o4nk | n1: 16.32, 15.37 → **15.37** (65.07 ms a token); n2: 16.17, 15.31 → **15.31** (65.32 ms a token) | 0.06 | -0.30 tok/s | -1.9 % | loss |
|  | o4ne | e1: 15.63, 15.64 → **15.64** (63.95 ms a token); e2: 15.63, 15.64 → **15.64** (63.92 ms a token) | 0.01 | +0.00 tok/s | +0.0 % | negligible |
| 16,384 | o4 | o1: 13.41, 13.45 → **13.45** (74.34 ms a token); o2: 13.40, 13.44 → **13.44** (74.39 ms a token) | 0.01 | control | | |
|  | o4nk | n1: 13.23, 12.14 → **12.14** (82.38 ms a token); n2: 13.53, 12.31 → **12.31** (81.25 ms a token) | 0.17 | -1.22 tok/s | -9.1 % | loss |
|  | o4ne | e1: 13.45, 13.45 → **13.45** (74.36 ms a token); e2: 13.44, 13.44 → **13.44** (74.38 ms a token) | 0.00 | -0.00 tok/s | -0.0 % | negligible |
| 32,768 | o4 | o1: 10.45, 10.48 → **10.48** (95.41 ms a token); o2: 10.46, 10.49 → **10.49** (95.31 ms a token) | 0.01 | control | | |
|  | o4nk | n1: 10.99, 9.32 → **9.32** (107.28 ms a token); n2: 11.20, 9.23 → **9.23** (108.36 ms a token) | 0.09 | -1.21 tok/s | -11.6 % | loss |
|  | o4ne | e1: 10.49, 10.50 → **10.50** (95.22 ms a token); e2: 10.49, 10.50 → **10.50** (95.25 ms a token) | 0.00 | +0.01 tok/s | +0.1 % | gain |
| 48,640 | o4 | o1: 8.65, 8.67 → **8.67** (115.36 ms a token); o2: 8.67, 8.68 → **8.68** (115.22 ms a token) | 0.01 | control | | |
|  | o4nk | n1: 9.44, 7.76 → **7.76** (128.91 ms a token); n2: 9.80, 7.91 → **7.91** (126.45 ms a token) | 0.15 | -0.84 tok/s | -9.7 % | loss |
|  | o4ne | e1: 8.69, 8.69 → **8.69** (115.09 ms a token); e2: 8.69, 8.69 → **8.69** (115.10 ms a token) | 0.00 | +0.01 tok/s | +0.2 % | gain |

### Generation over all depths, tok/s

Every depth's tokens over their time, per load.

| Depth (tokens) | Arm | Per load | Spread | Against `o4` | | Class |
|---:|---|---|---:|---:|---:|---|
| all | o4 | o1: **13.01**; o2: **13.02** | 0.01 | control | | |
|  | o4nk | n1: **11.71**; n2: **11.82** | 0.11 | -1.24 tok/s | -9.6 % | loss |
|  | o4ne | e1: **13.02**; e2: **13.02** | 0.00 | +0.01 tok/s | +0.1 % | negligible |

### Prompt reading, tok/s

Per load: the tokens read at that depth over their time (a prime request reads a chunk on top of the cache; without primes each decode request reads its prompt), with the tokens read and the tokens reused. Requests that read under 32 tokens are left out.

| Depth (tokens) | Arm | Per load | Spread | Against `o4` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | o4 | o1: 40.9 (64 read on 0 cached); o2: 40.8 (64 read on 0 cached) | 0.1 | control | | |
|  | o4nk | n1: 33.9 (64 read on 0 cached); n2: 33.2 (64 read on 0 cached) | 0.8 | -7.3 tok/s | -17.8 % | loss |
|  | o4ne | e1: 40.8 (64 read on 0 cached); e2: 40.7 (64 read on 0 cached) | 0.0 | -0.1 tok/s | -0.3 % | loss |
| 4,096 | o4 | o1: 170.7 (4032 read on 0 cached); o2: 168.6 (4032 read on 0 cached) | 2.1 | control | | |
|  | o4nk | n1: 173.8 (4032 read on 0 cached); n2: 172.5 (4032 read on 0 cached) | 1.3 | +3.5 tok/s | +2.1 % | gain |
|  | o4ne | e1: 172.5 (4032 read on 0 cached); e2: 172.6 (4032 read on 0 cached) | 0.1 | +2.9 tok/s | +1.7 % | gain |
| 8,192 | o4 | o1: 158.2 (4612 read on 3516 cached); o2: 158.8 (4612 read on 3516 cached) | 0.6 | control | | |
|  | o4nk | n1: 160.3 (4612 read on 3516 cached); n2: 161.3 (4612 read on 3516 cached) | 0.9 | +2.3 tok/s | +1.5 % | gain |
|  | o4ne | e1: 162.4 (4612 read on 3516 cached); e2: 162.4 (4612 read on 3516 cached) | 0.0 | +3.9 tok/s | +2.5 % | gain |
| 16,384 | o4 | o1: 203.0 (12804 read on 3516 cached); o2: 203.1 (12804 read on 3516 cached) | 0.1 | control | | |
|  | o4nk | n1: 206.5 (12804 read on 3516 cached); n2: 206.1 (12804 read on 3516 cached) | 0.5 | +3.3 tok/s | +1.6 % | gain |
|  | o4ne | e1: 206.3 (12804 read on 3516 cached); e2: 206.1 (12804 read on 3516 cached) | 0.1 | +3.2 tok/s | +1.6 % | gain |
| 32,768 | o4 | o1: 157.8 (16900 read on 15804 cached); o2: 157.8 (16900 read on 15804 cached) | 0.0 | control | | |
|  | o4nk | n1: 162.5 (16900 read on 15804 cached); n2: 162.8 (16900 read on 15804 cached) | 0.3 | +4.9 tok/s | +3.1 % | gain |
|  | o4ne | e1: 162.3 (16900 read on 15804 cached); e2: 162.1 (16900 read on 15804 cached) | 0.1 | +4.4 tok/s | +2.8 % | gain |
| 48,640 | o4 | o1: 137.2 (16388 read on 32188 cached); o2: 137.2 (16388 read on 32188 cached) | 0.0 | control | | |
|  | o4nk | n1: 141.5 (16388 read on 32188 cached); n2: 141.7 (16388 read on 32188 cached) | 0.3 | +4.4 tok/s | +3.2 % | gain |
|  | o4ne | e1: 140.6 (16388 read on 32188 cached); e2: 140.6 (16388 read on 32188 cached) | 0.0 | +3.4 tok/s | +2.5 % | gain |

### The draft and the step

Per load, pass 1: draft tokens accepted of drafted (the share), tokens a step (generated tokens over steps, a step being one pass of the main model: generated minus accepted), and the time of a step. Pass 2 is listed where its counters differ.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | o4 | o1: no draft, 1.00 a step, 54.5 ms a step; o2: no draft, 1.00 a step, 54.7 ms a step |
|  | o4nk | n1: no draft, 1.00 a step, 58.4 ms a step; n2: no draft, 1.00 a step, 58.2 ms a step |
|  | o4ne | e1: no draft, 1.00 a step, 54.5 ms a step; e2: no draft, 1.00 a step, 54.5 ms a step |
| 4,096 | o4 | o1: no draft, 1.00 a step, 59.1 ms a step; o2: no draft, 1.00 a step, 59.1 ms a step |
|  | o4nk | n1: no draft, 1.00 a step, 59.2 ms a step; n2: no draft, 1.00 a step, 57.5 ms a step |
|  | o4ne | e1: no draft, 1.00 a step, 58.9 ms a step; e2: no draft, 1.00 a step, 58.9 ms a step |
| 8,192 | o4 | o1: no draft, 1.00 a step, 64.3 ms a step; o2: no draft, 1.00 a step, 64.2 ms a step |
|  | o4nk | n1: no draft, 1.00 a step, 61.3 ms a step; n2: no draft, 1.00 a step, 61.8 ms a step |
|  | o4ne | e1: no draft, 1.00 a step, 64.0 ms a step; e2: no draft, 1.00 a step, 64.0 ms a step |
| 16,384 | o4 | o1: no draft, 1.00 a step, 74.6 ms a step; o2: no draft, 1.00 a step, 74.6 ms a step |
|  | o4nk | n1: no draft, 1.00 a step, 75.6 ms a step; n2: no draft, 1.00 a step, 73.9 ms a step |
|  | o4ne | e1: no draft, 1.00 a step, 74.4 ms a step; e2: no draft, 1.00 a step, 74.4 ms a step |
| 32,768 | o4 | o1: no draft, 1.00 a step, 95.7 ms a step; o2: no draft, 1.00 a step, 95.6 ms a step |
|  | o4nk | n1: no draft, 1.00 a step, 91.0 ms a step; n2: no draft, 1.00 a step, 89.3 ms a step |
|  | o4ne | e1: no draft, 1.00 a step, 95.3 ms a step; e2: no draft, 1.00 a step, 95.3 ms a step |
| 48,640 | o4 | o1: no draft, 1.00 a step, 115.6 ms a step; o2: no draft, 1.00 a step, 115.4 ms a step |
|  | o4nk | n1: no draft, 1.00 a step, 105.9 ms a step; n2: no draft, 1.00 a step, 102.0 ms a step |
|  | o4ne | e1: no draft, 1.00 a step, 115.1 ms a step; e2: no draft, 1.00 a step, 115.1 ms a step |

### The sampler pass

One request per depth and load with the server line's own sampler and a fixed seed: tok/s, the draft counters and the step.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | o4 | o1: 18.30 tok/s, no draft, 1.00 a step, 54.7 ms a step; o2: 18.26 tok/s, no draft, 1.00 a step, 54.8 ms a step |
|  | o4nk | n1: 16.15 tok/s, no draft, 1.00 a step, 61.9 ms a step; n2: 16.33 tok/s, no draft, 1.00 a step, 61.2 ms a step |
|  | o4ne | e1: 18.25 tok/s, no draft, 1.00 a step, 54.8 ms a step; e2: 18.29 tok/s, no draft, 1.00 a step, 54.7 ms a step |
| 4,096 | o4 | o1: 16.97 tok/s, no draft, 1.00 a step, 58.9 ms a step; o2: 16.97 tok/s, no draft, 1.00 a step, 58.9 ms a step |
|  | o4nk | n1: 14.06 tok/s, no draft, 1.00 a step, 71.1 ms a step; n2: 14.21 tok/s, no draft, 1.00 a step, 70.4 ms a step |
|  | o4ne | e1: 17.00 tok/s, no draft, 1.00 a step, 58.8 ms a step; e2: 16.98 tok/s, no draft, 1.00 a step, 58.9 ms a step |
| 8,192 | o4 | o1: 15.62 tok/s, no draft, 1.00 a step, 64.0 ms a step; o2: 15.64 tok/s, no draft, 1.00 a step, 64.0 ms a step |
|  | o4nk | n1: 13.04 tok/s, no draft, 1.00 a step, 76.7 ms a step; n2: 13.04 tok/s, no draft, 1.00 a step, 76.7 ms a step |
|  | o4ne | e1: 15.64 tok/s, no draft, 1.00 a step, 64.0 ms a step; e2: 15.64 tok/s, no draft, 1.00 a step, 63.9 ms a step |
| 16,384 | o4 | o1: 13.43 tok/s, no draft, 1.00 a step, 74.5 ms a step; o2: 13.42 tok/s, no draft, 1.00 a step, 74.5 ms a step |
|  | o4nk | n1: 11.37 tok/s, no draft, 1.00 a step, 88.0 ms a step; n2: 11.40 tok/s, no draft, 1.00 a step, 87.7 ms a step |
|  | o4ne | e1: 13.43 tok/s, no draft, 1.00 a step, 74.5 ms a step; e2: 13.43 tok/s, no draft, 1.00 a step, 74.5 ms a step |
| 32,768 | o4 | o1: 10.48 tok/s, no draft, 1.00 a step, 95.4 ms a step; o2: 10.49 tok/s, no draft, 1.00 a step, 95.4 ms a step |
|  | o4nk | n1: 9.03 tok/s, no draft, 1.00 a step, 110.8 ms a step; n2: 9.03 tok/s, no draft, 1.00 a step, 110.8 ms a step |
|  | o4ne | e1: 10.49 tok/s, no draft, 1.00 a step, 95.3 ms a step; e2: 10.50 tok/s, no draft, 1.00 a step, 95.2 ms a step |
| 48,640 | o4 | o1: 8.66 tok/s, no draft, 1.00 a step, 115.4 ms a step; o2: 8.68 tok/s, no draft, 1.00 a step, 115.3 ms a step |
|  | o4nk | n1: 7.54 tok/s, no draft, 1.00 a step, 132.6 ms a step; n2: 7.54 tok/s, no draft, 1.00 a step, 132.6 ms a step |
|  | o4ne | e1: 8.69 tok/s, no draft, 1.00 a step, 115.1 ms a step; e2: 8.68 tok/s, no draft, 1.00 a step, 115.2 ms a step |

### Checks

- Times: every load started on a machine checked quiet (valid=1).
- Prompts: 64 ids at depth 0, 4096 ids at depth 4,096, 8192 ids at depth 8,192, 16384 ids at depth 16,384, 32768 ids at depth 32,768, 48640 ids at depth 48,640; the same ids on every load and pass.
- Tokens out: 256 on every timed request.
- Greedy reply at depth 0: the same on every pass and load of every arm.
- Greedy reply at depth 4,096: the same on every pass and load of every arm.
- Greedy reply at depth 8,192: the same on every pass and load of every arm.
- Greedy reply at depth 16,384: the same on every pass and load of every arm.
- Greedy reply at depth 32,768: the same on every pass and load of every arm.
- Greedy reply at depth 48,640: the same on every pass and load of every arm.
- Primes (read on cached, the cache's source): 4,096: 4032 on 0 (none); 8,192: 4612 on 3516 (checkpoint); 16,384: 12804 on 3516 (checkpoint); 32,768: 16900 on 15804 (checkpoint); 48,640: 16388 on 32188 (checkpoint).
- Graph reallocations of the target, requests with one per load: o1: 0 of 24 requests; n1: 0 of 24 requests; e1: 0 of 24 requests; o2: 0 of 24 requests; n2: 0 of 24 requests; e2: 0 of 24 requests.
