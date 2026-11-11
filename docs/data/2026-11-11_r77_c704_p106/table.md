## The deep lane: v4c, rows, prof

Model file `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf`, llama-server version: 0.9.0-dev (build 12581, commit c408dabcb3).

256 tokens per request at temperature 0, each request twice per load (pass 1 is the first request at a new depth, pass 2 the same request again). Timed loads, in their order: v4c1, rows1, v4c2, rows2, prof.

| Arm | Loads | Draft | Context | Its own variables and flags |
|---|---|---|---:|---|
| v4c (control) | v4c1, v4c2 | on | 65536 | `GGML_CUDA_FATTN_SPARSE_PASCAL=1 GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=1.0 GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=5`, `--spec-draft-sampling probabilistic --spec-draft-p-min 0 --ctx-checkpoints-vram 4 --ubatch-taper on --spec-draft-n-ubatch 512` |
| rows | rows1, rows2 | on | 65536 | `GGML_CUDA_FATTN_SPARSE_PASCAL=1 GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=1.0 GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=5 LLAMA_QSA_INDEXER_ROWS=1`, `--spec-draft-sampling probabilistic --spec-draft-p-min 0 --ctx-checkpoints-vram 4 --ubatch-taper on --spec-draft-n-ubatch 512` |
| prof | prof | on | 65536 | `GGML_CUDA_FATTN_SPARSE_PASCAL=1 GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=1.0 GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=5 LLAMA_QSA_INDEXER_ROWS=1 GGML_CUDA_PROFILE=$OUT/prof-rows.csv GGML_CUDA_PROFILE_MODE=group`, `--spec-draft-sampling probabilistic --spec-draft-p-min 0 --ctx-checkpoints-vram 4 --ubatch-taper on --spec-draft-n-ubatch 512` |

### Generation, tok/s

Per load: pass 1, pass 2 → the load, with the load's time per token. A load's speed is its two passes' tokens over their time. Spread: the difference between the arm's fastest and slowest load. Class against `v4c`: gain when the mean is above the control's by more than the larger of the two spreads, loss when below by more than it, else negligible.

| Depth (tokens) | Arm | Per load | Spread | Against `v4c` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | v4c | v4c1: 26.09, 28.07 → **27.04** (36.98 ms a token); v4c2: 26.12, 28.06 → **27.06** (36.96 ms a token) | 0.01 | control | | |
|  | rows | rows1: 26.14, 28.06 → **27.07** (36.94 ms a token); rows2: 26.18, 28.08 → **27.09** (36.91 ms a token) | 0.02 | +0.03 tok/s | +0.1 % | gain |
|  | prof | prof: 20.78, 22.09 → **21.42** (46.69 ms a token) | — | -5.63 tok/s | -20.8 % | no class (one load) |
| 16,384 | v4c | v4c1: 23.84, 24.15 → **24.00** (41.67 ms a token); v4c2: 23.87, 24.12 → **24.00** (41.67 ms a token) | 0.00 | control | | |
|  | rows | rows1: 24.51, 24.81 → **24.66** (40.55 ms a token); rows2: 24.58, 24.88 → **24.73** (40.44 ms a token) | 0.07 | +0.70 tok/s | +2.9 % | gain |
|  | prof | prof: 19.44, 19.62 → **19.53** (51.21 ms a token) | — | -4.47 tok/s | -18.6 % | no class (one load) |
| 32,768 | v4c | v4c1: 14.53, 16.08 → **15.26** (65.52 ms a token); v4c2: 14.55, 16.07 → **15.27** (65.49 ms a token) | 0.01 | control | | |
|  | rows | rows1: 15.14, 16.83 → **15.94** (62.72 ms a token); rows2: 15.16, 16.76 → **15.92** (62.80 ms a token) | 0.02 | +0.67 tok/s | +4.4 % | gain |
|  | prof | prof: 11.54, 12.94 → **12.20** (81.97 ms a token) | — | -3.07 tok/s | -20.1 % | no class (one load) |
| 49,152 | v4c | v4c1: 18.11, 18.51 → **18.31** (54.62 ms a token); v4c2: 18.10, 18.46 → **18.27** (54.72 ms a token) | 0.03 | control | | |
|  | rows | rows1: 19.19, 19.68 → **19.43** (51.46 ms a token); rows2: 19.21, 19.65 → **19.43** (51.47 ms a token) | 0.01 | +1.14 tok/s | +6.2 % | gain |
|  | prof | prof: 14.97, 15.24 → **15.10** (66.21 ms a token) | — | -3.19 tok/s | -17.4 % | no class (one load) |

### Generation over all depths, tok/s

Every depth's tokens over their time, per load.

| Depth (tokens) | Arm | Per load | Spread | Against `v4c` | | Class |
|---:|---|---|---:|---:|---:|---|
| all | v4c | v4c1: **20.12**; v4c2: **20.12** | 0.01 | control | | |
|  | rows | rows1: **20.87**; rows2: **20.87** | 0.01 | +0.75 tok/s | +3.7 % | gain |
|  | prof | prof: **16.25** | — | -3.86 tok/s | -19.2 % | no class (one load) |

### The step on pass 1 (the first request at a new depth), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `v4c`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `v4c` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | v4c | v4c1: 125.81 (78 steps); v4c2: 125.66 (78 steps) | 0.15 | control | | |
|  | rows | rows1: 125.54 (78 steps); rows2: 125.39 (78 steps) | 0.15 | -0.27 ms | -0.2 % | gain |
|  | prof | prof: 157.95 (78 steps) | — | +32.22 ms | +25.6 % | no class (one load) |
| 16,384 | v4c | v4c1: 151.23 (71 steps); v4c2: 151.07 (71 steps) | 0.16 | control | | |
|  | rows | rows1: 147.11 (71 steps); rows2: 146.70 (71 steps) | 0.41 | -4.25 ms | -2.8 % | gain |
|  | prof | prof: 185.49 (71 steps) | — | +34.34 ms | +22.7 % | no class (one load) |
| 32,768 | v4c | v4c1: 181.69 (97 steps); v4c2: 181.41 (97 steps) | 0.27 | control | | |
|  | rows | rows1: 174.26 (97 steps); rows2: 174.06 (97 steps) | 0.21 | -7.39 ms | -4.1 % | gain |
|  | prof | prof: 228.72 (97 steps) | — | +47.17 ms | +26.0 % | no class (one load) |
| 49,152 | v4c | v4c1: 183.62 (77 steps); v4c2: 183.73 (77 steps) | 0.11 | control | | |
|  | rows | rows1: 173.22 (77 steps); rows2: 173.10 (77 steps) | 0.12 | -10.51 ms | -5.7 % | gain |
|  | prof | prof: 222.09 (77 steps) | — | +38.41 ms | +20.9 % | no class (one load) |

### The step on pass 2 (the same request again), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `v4c`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `v4c` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | v4c | v4c1: 121.58 (75 steps); v4c2: 121.62 (75 steps) | 0.04 | control | | |
|  | rows | rows1: 121.62 (75 steps); rows2: 121.56 (75 steps) | 0.06 | -0.01 ms | -0.0 % | negligible |
|  | prof | prof: 154.49 (75 steps) | — | +32.89 ms | +27.0 % | no class (one load) |
| 16,384 | v4c | v4c1: 149.29 (71 steps); v4c2: 149.46 (71 steps) | 0.17 | control | | |
|  | rows | rows1: 145.34 (71 steps); rows2: 144.94 (71 steps) | 0.41 | -4.24 ms | -2.8 % | gain |
|  | prof | prof: 183.80 (71 steps) | — | +34.42 ms | +23.0 % | no class (one load) |
| 32,768 | v4c | v4c1: 164.16 (97 steps); v4c2: 164.28 (97 steps) | 0.12 | control | | |
|  | rows | rows1: 156.79 (97 steps); rows2: 157.42 (97 steps) | 0.63 | -7.11 ms | -4.3 % | gain |
|  | prof | prof: 203.96 (97 steps) | — | +39.74 ms | +24.2 % | no class (one load) |
| 49,152 | v4c | v4c1: 179.58 (77 steps); v4c2: 180.15 (77 steps) | 0.57 | control | | |
|  | rows | rows1: 168.94 (77 steps); rows2: 169.16 (77 steps) | 0.22 | -10.81 ms | -6.0 % | gain |
|  | prof | prof: 218.14 (77 steps) | — | +38.28 ms | +21.3 % | no class (one load) |

### Prompt reading, tok/s

Per load: the tokens read at that depth over their time (a prime request reads a chunk on top of the cache; without primes each decode request reads its prompt), with the tokens read and the tokens reused. Requests that read under 32 tokens are left out.

| Depth (tokens) | Arm | Per load | Spread | Against `v4c` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | v4c | v4c1: 49.0 (64 read on 0 cached); v4c2: 49.5 (64 read on 0 cached) | 0.5 | control | | |
|  | rows | rows1: 49.6 (64 read on 0 cached); rows2: 49.6 (64 read on 0 cached) | 0.1 | +0.3 tok/s | +0.7 % | negligible |
|  | prof | prof: 48.2 (64 read on 0 cached) | — | -1.1 tok/s | -2.2 % | no class (one load) |
| 16,384 | v4c | v4c1: 135.4 (16320 read on 0 cached); v4c2: 134.6 (16320 read on 0 cached) | 0.8 | control | | |
|  | rows | rows1: 134.2 (16320 read on 0 cached); rows2: 134.5 (16320 read on 0 cached) | 0.3 | -0.7 tok/s | -0.5 % | negligible |
|  | prof | prof: 64.8 (16320 read on 0 cached) | — | -70.2 tok/s | -52.0 % | no class (one load) |
| 32,768 | v4c | v4c1: 100.2 (16900 read on 15804 cached); v4c2: 100.3 (16900 read on 15804 cached) | 0.2 | control | | |
|  | rows | rows1: 100.3 (16900 read on 15804 cached); rows2: 100.2 (16900 read on 15804 cached) | 0.1 | +0.0 tok/s | +0.0 % | negligible |
|  | prof | prof: 46.7 (16900 read on 15804 cached) | — | -53.5 tok/s | -53.4 % | no class (one load) |
| 49,152 | v4c | v4c1: 77.4 (16900 read on 32188 cached); v4c2: 77.3 (16900 read on 32188 cached) | 0.1 | control | | |
|  | rows | rows1: 77.5 (16900 read on 32188 cached); rows2: 77.4 (16900 read on 32188 cached) | 0.1 | +0.1 tok/s | +0.1 % | negligible |
|  | prof | prof: 35.9 (16900 read on 32188 cached) | — | -41.4 tok/s | -53.6 % | no class (one load) |

### The draft and the step

Per load, pass 1: draft tokens accepted of drafted (the share), tokens a step (generated tokens over steps, a step being one pass of the main model: generated minus accepted), and the time of a step. Pass 2 is listed where its counters differ.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | v4c | v4c1: 178 of 229 (0.777), 3.28 a step, 125.8 ms a step (pass 2: 181 of 220 (0.823), 3.41 a step, 121.6 ms a step); v4c2: 178 of 229 (0.777), 3.28 a step, 125.7 ms a step (pass 2: 181 of 220 (0.823), 3.41 a step, 121.6 ms a step) |
|  | rows | rows1: 178 of 229 (0.777), 3.28 a step, 125.5 ms a step (pass 2: 181 of 220 (0.823), 3.41 a step, 121.6 ms a step); rows2: 178 of 229 (0.777), 3.28 a step, 125.4 ms a step (pass 2: 181 of 220 (0.823), 3.41 a step, 121.6 ms a step) |
|  | prof | prof: 178 of 229 (0.777), 3.28 a step, 158.0 ms a step (pass 2: 181 of 220 (0.823), 3.41 a step, 154.5 ms a step) |
| 16,384 | v4c | v4c1: 185 of 210 (0.881), 3.61 a step, 151.2 ms a step; v4c2: 185 of 210 (0.881), 3.61 a step, 151.1 ms a step |
|  | rows | rows1: 185 of 210 (0.881), 3.61 a step, 147.1 ms a step; rows2: 185 of 210 (0.881), 3.61 a step, 146.7 ms a step |
|  | prof | prof: 185 of 210 (0.881), 3.61 a step, 185.5 ms a step |
| 32,768 | v4c | v4c1: 159 of 287 (0.554), 2.64 a step, 181.7 ms a step; v4c2: 159 of 287 (0.554), 2.64 a step, 181.4 ms a step |
|  | rows | rows1: 159 of 287 (0.554), 2.64 a step, 174.3 ms a step; rows2: 159 of 287 (0.554), 2.64 a step, 174.1 ms a step |
|  | prof | prof: 159 of 287 (0.554), 2.64 a step, 228.7 ms a step |
| 49,152 | v4c | v4c1: 179 of 227 (0.789), 3.32 a step, 183.6 ms a step; v4c2: 179 of 227 (0.789), 3.32 a step, 183.7 ms a step |
|  | rows | rows1: 179 of 227 (0.789), 3.32 a step, 173.2 ms a step; rows2: 179 of 227 (0.789), 3.32 a step, 173.1 ms a step |
|  | prof | prof: 179 of 227 (0.789), 3.32 a step, 222.1 ms a step |

### Checks

- Times: every load started on a machine checked quiet (valid=1).
- Prompts: 64 ids at depth 0, 16384 ids at depth 16,384, 32768 ids at depth 32,768, 49152 ids at depth 49,152; the same ids on every load and pass.
- Tokens out: 256 on every timed request.
- Greedy reply at depth 0: the same on every pass and load of every arm.
- Greedy reply at depth 16,384: the same on every pass and load of every arm.
- Greedy reply at depth 32,768: the same on every pass and load of every arm.
- Greedy reply at depth 49,152: the same on every pass and load of every arm.
- Primes (read on cached, the cache's source): 16,384: 16320 on 0 (none); 32,768: 16900 on 15804 (checkpoint); 49,152: 16900 on 32188 (checkpoint).
