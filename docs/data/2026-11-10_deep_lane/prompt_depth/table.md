## The deep lane: ub512, ub128, sp128

Model file `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf`, llama-server version: 0.8.0-dev (build 12358, commit 1551f5fe1).

16 tokens per request at temperature 0, each request twice per load (pass 1 is the first request at a new depth, pass 2 the same request again). Timed loads, in their order: ub512_1, ub128_1, sp128_1, ub512_2, ub128_2, sp128_2.

| Arm | Loads | Draft | Context | Its own variables and flags |
|---|---|---|---:|---|
| ub512 (control) | ub512_1, ub512_2 | on | 65536 | `GGML_CUDA_FATTN_SPARSE_PASCAL=1` |
| ub128 | ub128_1, ub128_2 | on | 65536 | `GGML_CUDA_FATTN_SPARSE_PASCAL=1` |
| sp128 | sp128_1, sp128_2 | on | 65536 | `GGML_CUDA_FATTN_SPARSE_PASCAL=2 GGML_CUDA_FATTN_SPARSE_PASCAL_NQ=128` |

### Generation, tok/s

Per load: pass 1, pass 2 → the load, with the load's time per token. A load's speed is its two passes' tokens over their time. Spread: the difference between the arm's fastest and slowest load. Class against `ub512`: gain when the mean is above the control's by more than the larger of the two spreads, loss when below by more than it, else negligible.

| Depth (tokens) | Arm | Per load | Spread | Against `ub512` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | ub512 | ub512_1: 7.24, 15.72 → **9.92** (100.84 ms a token); ub512_2: 5.49, 15.71 → **8.13** (122.93 ms a token) | 1.78 | control | | |
|  | ub128 | ub128_1: 7.08, 15.72 → **9.77** (102.39 ms a token); ub128_2: 7.11, 15.71 → **9.78** (102.21 ms a token) | 0.02 | +0.75 tok/s | +8.3 % | negligible |
|  | sp128 | sp128_1: 7.72, 8.10 → **7.90** (126.55 ms a token); sp128_2: 7.72, 8.09 → **7.90** (126.60 ms a token) | 0.00 | -1.13 tok/s | -12.5 % | negligible |
| 16,384 | ub512 | ub512_1: 5.84, 15.60 → **8.50** (117.64 ms a token); ub512_2: 5.58, 15.05 → **8.14** (122.79 ms a token) | 0.36 | control | | |
|  | ub128 | ub128_1: 18.89, 20.49 → **19.66** (50.87 ms a token); ub128_2: 11.84, 12.73 → **12.27** (81.53 ms a token) | 7.39 | +7.64 tok/s | +91.8 % | gain |
|  | sp128 | sp128_1: 18.80, 20.45 → **19.59** (51.05 ms a token); sp128_2: 13.57, 20.38 → **16.29** (61.38 ms a token) | 3.30 | +9.62 tok/s | +115.6 % | gain |
| 32,768 | ub512 | ub512_1: 11.70, 13.42 → **12.50** (79.98 ms a token); ub512_2: 15.07, 15.67 → **15.36** (65.09 ms a token) | 2.86 | control | | |
|  | ub128 | ub128_1: 5.41, 13.96 → **7.80** (128.19 ms a token); ub128_2: 4.62, 11.89 → **6.66** (150.26 ms a token) | 1.15 | -6.70 tok/s | -48.1 % | loss |
|  | sp128 | sp128_1: 7.83, 14.14 → **10.07** (99.26 ms a token); sp128_2: 6.64, 7.82 → **7.18** (139.21 ms a token) | 2.89 | -5.30 tok/s | -38.1 % | loss |
| 49,152 | ub512 | ub512_1: 16.59, 17.19 → **16.88** (59.23 ms a token); ub512_2: 16.68, 17.20 → **16.93** (59.05 ms a token) | 0.05 | control | | |
|  | ub128 | ub128_1: 16.45, 17.09 → **16.77** (59.65 ms a token); ub128_2: 14.79, 15.74 → **15.25** (65.58 ms a token) | 1.52 | -0.90 tok/s | -5.3 % | negligible |
|  | sp128 | sp128_1: 11.17, 17.60 → **13.67** (73.16 ms a token); sp128_2: 16.55, 17.17 → **16.85** (59.34 ms a token) | 3.18 | -1.65 tok/s | -9.8 % | negligible |

### Generation over all depths, tok/s

Every depth's tokens over their time, per load.

| Depth (tokens) | Arm | Per load | Spread | Against `ub512` | | Class |
|---:|---|---|---:|---:|---:|---|
| all | ub512 | ub512_1: **11.18**; ub512_2: **10.81** | 0.37 | control | | |
|  | ub128 | ub128_1: **11.73**; ub128_2: **10.01** | 1.72 | -0.13 tok/s | -1.2 % | negligible |
|  | sp128 | sp128_1: **11.43**; sp128_2: **10.35** | 1.08 | -0.11 tok/s | -1.0 % | negligible |

### The step on pass 1 (the first request at a new depth), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `ub512`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `ub512` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | ub512 | ub512_1: 245.43 (9 steps); ub512_2: 323.93 (9 steps) | 78.51 | control | | |
|  | ub128 | ub128_1: 250.95 (9 steps); ub128_2: 250.21 (9 steps) | 0.74 | -34.10 ms | -12.0 % | negligible |
|  | sp128 | sp128_1: 259.16 (8 steps); sp128_2: 259.19 (8 steps) | 0.02 | -25.50 ms | -9.0 % | negligible |
| 16,384 | ub512 | ub512_1: 391.22 (7 steps); ub512_2: 409.42 (7 steps) | 18.19 | control | | |
|  | ub128 | ub128_1: 141.15 (6 steps); ub128_2: 193.11 (7 steps) | 51.96 | -233.19 ms | -58.3 % | gain |
|  | sp128 | sp128_1: 141.86 (6 steps); sp128_2: 196.49 (6 steps) | 54.62 | -231.15 ms | -57.7 % | gain |
| 32,768 | ub512 | ub512_1: 195.29 (7 steps); ub512_2: 151.72 (7 steps) | 43.56 | control | | |
|  | ub128 | ub128_1: 369.47 (8 steps); ub128_2: 494.69 (7 steps) | 125.22 | +258.58 ms | +149.0 % | loss |
|  | sp128 | sp128_1: 255.57 (8 steps); sp128_2: 267.63 (9 steps) | 12.06 | +88.10 ms | +50.8 % | loss |
| 49,152 | ub512 | ub512_1: 160.77 (6 steps); ub512_2: 159.92 (6 steps) | 0.86 | control | | |
|  | ub128 | ub128_1: 162.09 (6 steps); ub128_2: 180.35 (6 steps) | 18.25 | +10.88 ms | +6.8 % | negligible |
|  | sp128 | sp128_1: 238.65 (6 steps); sp128_2: 161.15 (6 steps) | 77.51 | +39.56 ms | +24.7 % | negligible |

### The step on pass 2 (the same request again), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `ub512`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `ub512` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | ub512 | ub512_1: 113.12 (9 steps); ub512_2: 113.14 (9 steps) | 0.03 | control | | |
|  | ub128 | ub128_1: 113.11 (9 steps); ub128_2: 113.19 (9 steps) | 0.09 | +0.02 ms | +0.0 % | negligible |
|  | sp128 | sp128_1: 282.35 (7 steps); sp128_2: 282.53 (7 steps) | 0.18 | +169.31 ms | +149.7 % | loss |
| 16,384 | ub512 | ub512_1: 146.56 (7 steps); ub512_2: 151.89 (7 steps) | 5.34 | control | | |
|  | ub128 | ub128_1: 130.15 (6 steps); ub128_2: 179.60 (7 steps) | 49.45 | +5.65 ms | +3.8 % | negligible |
|  | sp128 | sp128_1: 130.39 (6 steps); sp128_2: 130.88 (6 steps) | 0.48 | -18.59 ms | -12.5 % | gain |
| 32,768 | ub512 | ub512_1: 170.32 (7 steps); ub512_2: 145.84 (7 steps) | 24.48 | control | | |
|  | ub128 | ub128_1: 143.27 (8 steps); ub128_2: 192.20 (7 steps) | 48.93 | +9.66 ms | +6.1 % | negligible |
|  | sp128 | sp128_1: 141.46 (8 steps); sp128_2: 255.75 (8 steps) | 114.28 | +40.52 ms | +25.6 % | negligible |
| 49,152 | ub512 | ub512_1: 155.14 (6 steps); ub512_2: 155.02 (6 steps) | 0.12 | control | | |
|  | ub128 | ub128_1: 156.02 (6 steps); ub128_2: 169.39 (6 steps) | 13.37 | +7.63 ms | +4.9 % | negligible |
|  | sp128 | sp128_1: 151.54 (6 steps); sp128_2: 155.34 (6 steps) | 3.80 | -1.63 ms | -1.1 % | negligible |

### Prompt reading, tok/s

Per load: the tokens read at that depth over their time (a prime request reads a chunk on top of the cache; without primes each decode request reads its prompt), with the tokens read and the tokens reused. Requests that read under 32 tokens are left out.

| Depth (tokens) | Arm | Per load | Spread | Against `ub512` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | ub512 | ub512_1: 32.7 (64 read on 0 cached); ub512_2: 28.0 (64 read on 0 cached) | 4.7 | control | | |
|  | ub128 | ub128_1: 32.4 (64 read on 0 cached); ub128_2: 32.7 (64 read on 0 cached) | 0.2 | +2.2 tok/s | +7.2 % | negligible |
|  | sp128 | sp128_1: 32.6 (64 read on 0 cached); sp128_2: 32.1 (64 read on 0 cached) | 0.5 | +2.0 tok/s | +6.6 % | negligible |
| 16,384 | ub512 | ub512_1: 123.8 (16320 read on 0 cached); ub512_2: 123.7 (16320 read on 0 cached) | 0.1 | control | | |
|  | ub128 | ub128_1: 118.2 (16320 read on 0 cached); ub128_2: 118.6 (16320 read on 0 cached) | 0.3 | -5.4 tok/s | -4.3 % | loss |
|  | sp128 | sp128_1: 128.2 (16320 read on 0 cached); sp128_2: 128.3 (16320 read on 0 cached) | 0.1 | +4.5 tok/s | +3.6 % | gain |
| 32,768 | ub512 | ub512_1: 90.2 (16900 read on 15804 cached); ub512_2: 90.3 (16900 read on 15804 cached) | 0.1 | control | | |
|  | ub128 | ub128_1: 98.5 (16516 read on 16188 cached); ub128_2: 98.5 (16516 read on 16188 cached) | 0.0 | +8.2 tok/s | +9.1 % | gain |
|  | sp128 | sp128_1: 133.8 (16516 read on 16188 cached); sp128_2: 133.9 (16516 read on 16188 cached) | 0.0 | +43.6 tok/s | +48.3 % | gain |
| 49,152 | ub512 | ub512_1: 70.2 (16900 read on 32188 cached); ub512_2: 70.0 (16900 read on 32188 cached) | 0.2 | control | | |
|  | ub128 | ub128_1: 80.5 (16516 read on 32572 cached); ub128_2: 80.7 (16516 read on 32572 cached) | 0.2 | +10.6 tok/s | +15.1 % | gain |
|  | sp128 | sp128_1: 131.3 (16516 read on 32572 cached); sp128_2: 131.9 (16516 read on 32572 cached) | 0.6 | +61.5 tok/s | +87.8 % | gain |

### The draft and the step

Per load, pass 1: draft tokens accepted of drafted (the share), tokens a step (generated tokens over steps, a step being one pass of the main model: generated minus accepted), and the time of a step. Pass 2 is listed where its counters differ.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | ub512 | ub512_1: 7 of 23 (0.304), 1.78 a step, 245.4 ms a step; ub512_2: 7 of 23 (0.304), 1.78 a step, 323.9 ms a step |
|  | ub128 | ub128_1: 7 of 23 (0.304), 1.78 a step, 250.9 ms a step; ub128_2: 7 of 23 (0.304), 1.78 a step, 250.2 ms a step |
|  | sp128 | sp128_1: 8 of 18 (0.444), 2.00 a step, 259.2 ms a step (pass 2: 9 of 17 (0.529), 2.29 a step, 282.4 ms a step); sp128_2: 8 of 18 (0.444), 2.00 a step, 259.2 ms a step (pass 2: 9 of 17 (0.529), 2.29 a step, 282.5 ms a step) |
| 16,384 | ub512 | ub512_1: 9 of 15 (0.600), 2.29 a step, 391.2 ms a step; ub512_2: 9 of 15 (0.600), 2.29 a step, 409.4 ms a step |
|  | ub128 | ub128_1: 10 of 15 (0.667), 2.67 a step, 141.2 ms a step; ub128_2: 9 of 16 (0.562), 2.29 a step, 193.1 ms a step |
|  | sp128 | sp128_1: 10 of 15 (0.667), 2.67 a step, 141.9 ms a step; sp128_2: 10 of 14 (0.714), 2.67 a step, 196.5 ms a step |
| 32,768 | ub512 | ub512_1: 9 of 15 (0.600), 2.29 a step, 195.3 ms a step; ub512_2: 9 of 17 (0.529), 2.29 a step, 151.7 ms a step |
|  | ub128 | ub128_1: 8 of 16 (0.500), 2.00 a step, 369.5 ms a step (pass 2: 8 of 15 (0.533), 2.00 a step, 143.3 ms a step); ub128_2: 9 of 15 (0.600), 2.29 a step, 494.7 ms a step (pass 2: 9 of 14 (0.643), 2.29 a step, 192.2 ms a step) |
|  | sp128 | sp128_1: 8 of 18 (0.444), 2.00 a step, 255.6 ms a step (pass 2: 8 of 16 (0.500), 2.00 a step, 141.5 ms a step); sp128_2: 7 of 20 (0.350), 1.78 a step, 267.6 ms a step (pass 2: 8 of 20 (0.400), 2.00 a step, 255.7 ms a step) |
| 49,152 | ub512 | ub512_1: 10 of 14 (0.714), 2.67 a step, 160.8 ms a step; ub512_2: 10 of 14 (0.714), 2.67 a step, 159.9 ms a step |
|  | ub128 | ub128_1: 10 of 14 (0.714), 2.67 a step, 162.1 ms a step; ub128_2: 10 of 14 (0.714), 2.67 a step, 180.3 ms a step |
|  | sp128 | sp128_1: 10 of 13 (0.769), 2.67 a step, 238.7 ms a step; sp128_2: 10 of 14 (0.714), 2.67 a step, 161.1 ms a step |

### Checks

- Times: every load started on a machine checked quiet (valid=1).
- Prompts: 64 ids at depth 0, 16384 ids at depth 16,384, 32768 ids at depth 32,768, 49152 ids at depth 49,152; the same ids on every load and pass.
- Tokens out: 16 on every timed request.
- Greedy reply at depth 0: **differs from `ub512` in sp128.**
- Greedy reply at depth 16,384: **differs from `ub512` in ub128, sp128.**
- Greedy reply at depth 32,768: 6 different replies over the 6 loads (`ub512`: 3 over its 2; `ub128`: 3 over its 2; `sp128`: 2 over its 2); **ub512_1, ub128_2 wrote another reply on pass 2 than on pass 1.**
- Greedy reply at depth 49,152: **differs from `ub512` in sp128.**
- Primes (read on cached, the cache's source): 16,384: 16320 on 0 (none); 32,768: 16516 on 16188 (checkpoint); 32,768: 16900 on 15804 (checkpoint); 49,152: 16516 on 32572 (checkpoint); 49,152: 16900 on 32188 (checkpoint).
