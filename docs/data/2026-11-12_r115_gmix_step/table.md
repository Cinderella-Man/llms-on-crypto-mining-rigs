## The deep lane: pmix, gmix

Model file `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-pmix-fast-00001-of-00003.gguf`, llama-server version: 0.9.0-dev (build 12661, commit d097d6d88a).

256 tokens per request at temperature 0, each request twice per load (pass 1 is the first request at a new depth, pass 2 the same request again). Timed loads, in their order: pmix1, gmix1, pmix2, gmix2.

| Arm | Loads | Draft | Context | Its own variables and flags |
|---|---|---|---:|---|
| pmix (control) | pmix1, pmix2 | on | 65536 | `GGML_CUDA_FATTN_SPARSE_PASCAL=1 GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=1.0 GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=5 GGML_CUDA_MMVQ_PASCAL_DMT=1 GGML_CUDA_MMVQ_PASCAL_Q5K_MT=1 LLAMA_QSA_INDEXER_ROWS=1 LLAMA_QSA_KPOOL_DEV=1`, `--spec-draft-sampling probabilistic --spec-draft-p-min 0 --ctx-checkpoints-vram 4 --ctx-checkpoints-incall --ubatch-taper short=600:64 --spec-draft-n-ubatch 512` |
| gmix | gmix1, gmix2 | on | 65536 | `GGML_CUDA_FATTN_SPARSE_PASCAL=1 GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=1.0 GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=5 GGML_CUDA_MMVQ_PASCAL_DMT=1 GGML_CUDA_MMVQ_PASCAL_Q5K_MT=1 LLAMA_QSA_INDEXER_ROWS=1 LLAMA_QSA_KPOOL_DEV=1`, `--spec-draft-sampling probabilistic --spec-draft-p-min 0 --ctx-checkpoints-vram 4 --ctx-checkpoints-incall --ubatch-taper short=600:64 --spec-draft-n-ubatch 512 -m Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-pmix-fast-gmix-q2-00001-of-00003.gguf` |

### Generation, tok/s

Per load: pass 1, pass 2 → the load, with the load's time per token. A load's speed is its two passes' tokens over their time. Spread: the difference between the arm's fastest and slowest load. Class against `pmix`: gain when the mean is above the control's by more than the larger of the two spreads, loss when below by more than it, else negligible.

| Depth (tokens) | Arm | Per load | Spread | Against `pmix` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | pmix | pmix1: 26.90, 28.00 → **27.44** (36.45 ms a token); pmix2: 25.96, 27.94 → **26.91** (37.15 ms a token) | 0.52 | control | | |
|  | gmix | gmix1: 17.22, 19.25 → **18.18** (55.00 ms a token); gmix2: 17.24, 19.27 → **18.19** (54.96 ms a token) | 0.01 | -8.99 tok/s | -33.1 % | loss |
| 8,192 | pmix | pmix1: 20.24, 26.62 → **23.00** (43.48 ms a token); pmix2: 20.73, 26.70 → **23.34** (42.84 ms a token) | 0.34 | control | | |
|  | gmix | gmix1: 16.29, 20.90 → **18.31** (54.62 ms a token); gmix2: 16.29, 20.88 → **18.30** (54.64 ms a token) | 0.01 | -4.87 tok/s | -21.0 % | loss |

### Generation over all depths, tok/s

Every depth's tokens over their time, per load.

| Depth (tokens) | Arm | Per load | Spread | Against `pmix` | | Class |
|---:|---|---|---:|---:|---:|---|
| all | pmix | pmix1: **25.02**; pmix2: **25.00** | 0.02 | control | | |
|  | gmix | gmix1: **18.24**; gmix2: **18.25** | 0.00 | -6.77 tok/s | -27.0 % | loss |

### The step on pass 1 (the first request at a new depth), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `pmix`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `pmix` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | pmix | pmix1: 102.34 (93 steps); pmix2: 106.02 (93 steps) | 3.67 | control | | |
|  | gmix | gmix1: 125.98 (118 steps); gmix2: 125.87 (118 steps) | 0.11 | +21.74 ms | +20.9 % | loss |
| 8,192 | pmix | pmix1: 150.56 (84 steps); pmix2: 146.98 (84 steps) | 3.58 | control | | |
|  | gmix | gmix1: 170.84 (92 steps); gmix2: 170.81 (92 steps) | 0.02 | +22.05 ms | +14.8 % | loss |

### The step on pass 2 (the same request again), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `pmix`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `pmix` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | pmix | pmix1: 99.37 (92 steps); pmix2: 99.60 (92 steps) | 0.23 | control | | |
|  | gmix | gmix1: 118.71 (112 steps); gmix2: 118.63 (112 steps) | 0.08 | +19.19 ms | +19.3 % | loss |
| 8,192 | pmix | pmix1: 117.26 (82 steps); pmix2: 116.94 (82 steps) | 0.31 | control | | |
|  | gmix | gmix1: 134.59 (91 steps); gmix2: 134.75 (91 steps) | 0.16 | +17.57 ms | +15.0 % | loss |

### Prompt reading, tok/s

Per load: the tokens read at that depth over their time (a prime request reads a chunk on top of the cache; without primes each decode request reads its prompt), with the tokens read and the tokens reused. Requests that read under 32 tokens are left out.

| Depth (tokens) | Arm | Per load | Spread | Against `pmix` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | pmix | pmix1: 54.4 (64 read on 0 cached); pmix2: 53.0 (64 read on 0 cached) | 1.4 | control | | |
|  | gmix | gmix1: 51.3 (64 read on 0 cached); gmix2: 51.2 (64 read on 0 cached) | 0.1 | -2.5 tok/s | -4.6 % | loss |
| 8,192 | pmix | pmix1: 152.2 (8128 read on 0 cached); pmix2: 150.6 (8128 read on 0 cached) | 1.5 | control | | |
|  | gmix | gmix1: 122.5 (8128 read on 0 cached); gmix2: 127.5 (8128 read on 0 cached) | 4.9 | -26.4 tok/s | -17.4 % | loss |

### The draft and the step

Per load, pass 1: draft tokens accepted of drafted (the share), tokens a step (generated tokens over steps, a step being one pass of the main model: generated minus accepted), and the time of a step. Pass 2 is listed where its counters differ.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | pmix | pmix1: 163 of 274 (0.595), 2.75 a step, 102.3 ms a step (pass 2: 164 of 271 (0.605), 2.78 a step, 99.4 ms a step); pmix2: 163 of 274 (0.595), 2.75 a step, 106.0 ms a step (pass 2: 164 of 271 (0.605), 2.78 a step, 99.6 ms a step) |
|  | gmix | gmix1: 138 of 349 (0.395), 2.17 a step, 126.0 ms a step (pass 2: 144 of 331 (0.435), 2.29 a step, 118.7 ms a step); gmix2: 138 of 349 (0.395), 2.17 a step, 125.9 ms a step (pass 2: 144 of 331 (0.435), 2.29 a step, 118.6 ms a step) |
| 8,192 | pmix | pmix1: 172 of 249 (0.691), 3.05 a step, 150.6 ms a step (pass 2: 174 of 243 (0.716), 3.12 a step, 117.3 ms a step); pmix2: 172 of 249 (0.691), 3.05 a step, 147.0 ms a step (pass 2: 174 of 243 (0.716), 3.12 a step, 116.9 ms a step) |
|  | gmix | gmix1: 164 of 270 (0.607), 2.78 a step, 170.8 ms a step (pass 2: 165 of 267 (0.618), 2.81 a step, 134.6 ms a step); gmix2: 164 of 270 (0.607), 2.78 a step, 170.8 ms a step (pass 2: 165 of 267 (0.618), 2.81 a step, 134.8 ms a step) |

### The sampler pass

One request per depth and load with the server line's own sampler and a fixed seed: tok/s, the draft counters and the step.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | pmix | pmix1: 20.69 tok/s, 156 of 296 (0.527), 2.56 a step, 123.7 ms a step; pmix2: 20.04 tok/s, 156 of 296 (0.527), 2.56 a step, 127.7 ms a step |
|  | gmix | gmix1: 20.22 tok/s, 167 of 261 (0.640), 2.88 a step, 142.3 ms a step; gmix2: 20.61 tok/s, 167 of 261 (0.640), 2.88 a step, 139.6 ms a step |
| 8,192 | pmix | pmix1: 23.29 tok/s, 167 of 263 (0.635), 2.88 a step, 123.5 ms a step; pmix2: 23.18 tok/s, 167 of 263 (0.635), 2.88 a step, 124.1 ms a step |
|  | gmix | gmix1: 19.68 tok/s, 165 of 268 (0.616), 2.81 a step, 142.9 ms a step; gmix2: 19.68 tok/s, 165 of 268 (0.616), 2.81 a step, 142.9 ms a step |

### Checks

- Times: every load started on a machine checked quiet (valid=1).
- Prompts: 64 ids at depth 0, 8192 ids at depth 8,192; the same ids on every load and pass.
- Tokens out: 256 on every timed request.
- Greedy reply at depth 0: **differs from `pmix` in gmix.**
- Greedy reply at depth 8,192: **differs from `pmix` in gmix.**
- Primes (read on cached, the cache's source): 8,192: 8128 on 0 (none).
