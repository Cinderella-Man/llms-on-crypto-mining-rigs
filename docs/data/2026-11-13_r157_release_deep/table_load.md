## The deep lane: v4r, v3

Model file `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf`, llama-server version: 1.0.0-dev (build 12832, commit eb312b7bce).

256 tokens per request at temperature 0, each request twice per load (pass 1 is the first request at a new depth, pass 2 the same request again). Timed loads, in their order: v4r1, v32, v4r2, v31.

| Arm | Loads | Draft | Context | Its own variables and flags |
|---|---|---|---:|---|
| v4r | v4r1, v4r2 | on | 65536 | `GGML_CUDA_FATTN_SPARSE_PASCAL=1 GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=1.0 GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=5 GGML_CUDA_MMVQ_PASCAL_DMT=1 GGML_CUDA_MMVQ_PASCAL_Q5K_MT=1 LLAMA_SAMPLING_TOP_K_ROWS=1 LLAMA_QSA_INDEXER_ROWS=1 LLAMA_QSA_KPOOL_DEV=1 LLAMA_QSA_KPOOL_STAGE=1 LLAMA_QSA_POOL_MASK_DEV=1 LLAMA_DRAFT_PREFETCH=1 LLAMA_VERIFY_PREPARE=2 LLAMA_MTP_GRAPH_KEEP=1 LLAMA_KV_SEQ_RM_BOUND=1 LLAMA_SERVER_YIELD=2 LLAMA_MTP_EMBD_FIRST=1 LLAMA_MTP_DRAFT_CAP=1 LLAMA_MTP_SKIP_LAST_ROW=1 GGML_CUDA_GRAPHS_FORCE=1 LLAMA_MTP_KV_ONLY=1 LLAMA_MTP_CATCHUP_OVERLAP=1 GGML_CUDA_MMVQ_PASCAL_Q3K_MT=1 GGML_CUDA_MMVQ_PASCAL_GLU_MT=1 GGML_CUDA_MMVQ_PASCAL_Q6K_MT=1 LLAMA_SPEC_REJECT_STOP=1 LLAMA_DRAFT_EMBD_DEV=layer GGML_CUDA_FATTN_SPARSE_PASCAL_PROMPT=1 LLAMA_SCHED_PER_SIZE_RESERVE=2 GGML_CUDA_MMVF_F32_MT=1 GGML_CUDA_KEEPALIVE_EAGER=1 LLAMA_MTP_HROW_LAZY=1 GGML_CUDA_MMVQ_PASCAL_KCOL_EXACT=1 GGML_CUDA_MMVQ_PASCAL_KCOL_SKIP=q4_0,q5_0`, `-b 4096 --spec-draft-n-ubatch 512 --spec-draft-sampling probabilistic --spec-draft-p-min 0 --ctx-checkpoints-vram 4 --ctx-checkpoints-incall --ubatch-taper short=600:64` |
| v3 (control) | v32, v31 | on | 65536 | `-otd token_embd=CUDA4` |

### Generation, tok/s

Per load: pass 1, pass 2 → the load, with the load's time per token. A load's speed is its two passes' tokens over their time. Spread: the difference between the arm's fastest and slowest load. Class against `v3`: gain when the mean is above the control's by more than the larger of the two spreads, loss when below by more than it, else negligible.

| Depth (tokens) | Arm | Per load | Spread | Against `v3` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | v4r | v4r1: 22.90, 24.13 → **23.50** (42.55 ms a token); v4r2: 22.89, 24.12 → **23.49** (42.57 ms a token) | 0.01 | -3.24 tok/s | -12.1 % | loss |
|  | v3 | v32: 25.84, 27.75 → **26.77** (37.36 ms a token); v31: 25.74, 27.75 → **26.71** (37.44 ms a token) | 0.06 | control | | |
| 4,096 | v4r | v4r1: 28.13, 29.58 → **28.84** (34.68 ms a token); v4r2: 28.44, 29.60 → **29.01** (34.47 ms a token) | 0.17 | +10.15 tok/s | +54.1 % | gain |
|  | v3 | v32: 15.16, 24.51 → **18.73** (53.39 ms a token); v31: 14.85, 25.63 → **18.81** (53.17 ms a token) | 0.08 | control | | |
| 8,192 | v4r | v4r1: 30.06, 31.59 → **30.81** (32.46 ms a token); v4r2: 30.02, 31.63 → **30.80** (32.47 ms a token) | 0.00 | +7.26 tok/s | +30.8 % | gain |
|  | v3 | v32: 20.80, 23.80 → **22.20** (45.05 ms a token); v31: 24.72, 25.07 → **24.89** (40.17 ms a token) | 2.70 | control | | |
| 16,384 | v4r | v4r1: 30.39, 31.41 → **30.89** (32.37 ms a token); v4r2: 30.39, 31.39 → **30.88** (32.39 ms a token) | 0.01 | +12.16 tok/s | +64.9 % | gain |
|  | v3 | v32: 17.57, 19.76 → **18.60** (53.77 ms a token); v31: 17.71, 20.16 → **18.85** (53.04 ms a token) | 0.26 | control | | |
| 32,768 | v4r | v4r1: 21.27, 22.07 → **21.66** (46.16 ms a token); v4r2: 21.21, 22.07 → **21.63** (46.23 ms a token) | 0.03 | +9.78 tok/s | +82.4 % | gain |
|  | v3 | v32: 12.19, 12.60 → **12.39** (80.69 ms a token); v31: 11.21, 11.49 → **11.35** (88.13 ms a token) | 1.05 | control | | |
| 49,152 | v4r | v4r1: 24.01, 24.51 → **24.26** (41.22 ms a token); v4r2: 23.80, 24.49 → **24.14** (41.42 ms a token) | 0.11 | +13.38 tok/s | +123.6 % | gain |
|  | v3 | v32: 10.47, 11.55 → **10.98** (91.04 ms a token); v31: 10.58, 10.75 → **10.66** (93.77 ms a token) | 0.32 | control | | |

### Generation over all depths, tok/s

Every depth's tokens over their time, per load.

| Depth (tokens) | Arm | Per load | Spread | Against `v3` | | Class |
|---:|---|---|---:|---:|---:|---|
| all | v4r | v4r1: **26.15**; v4r2: **26.14** | 0.01 | +9.64 tok/s | +58.4 % | gain |
|  | v3 | v32: **16.61**; v31: **16.41** | 0.20 | control | | |

### The step on pass 1 (the first request at a new depth), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `v3`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `v3` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | v4r | v4r1: 102.55 (109 steps); v4r2: 102.59 (109 steps) | 0.04 | -24.68 ms | -19.4 % | gain |
|  | v3 | v32: 127.00 (78 steps); v31: 127.50 (78 steps) | 0.51 | control | | |
| 4,096 | v4r | v4r1: 115.18 (79 steps); v4r2: 113.94 (79 steps) | 1.24 | -114.58 ms | -50.0 % | gain |
|  | v3 | v32: 222.20 (76 steps); v31: 236.09 (73 steps) | 13.88 | control | | |
| 8,192 | v4r | v4r1: 118.28 (72 steps); v4r2: 118.44 (72 steps) | 0.16 | -44.41 ms | -27.3 % | gain |
|  | v3 | v32: 170.96 (72 steps); v31: 154.57 (67 steps) | 16.39 | control | | |
| 16,384 | v4r | v4r1: 122.09 (69 steps); v4r2: 122.10 (69 steps) | 0.01 | -83.81 ms | -40.7 % | gain |
|  | v3 | v32: 205.27 (71 steps); v31: 206.56 (70 steps) | 1.29 | control | | |
| 32,768 | v4r | v4r1: 130.81 (92 steps); v4r2: 131.18 (92 steps) | 0.37 | -116.66 ms | -47.1 % | gain |
|  | v3 | v32: 246.98 (85 steps); v31: 248.33 (92 steps) | 1.35 | control | | |
| 49,152 | v4r | v4r1: 140.30 (76 steps); v4r2: 141.52 (76 steps) | 1.22 | -177.48 ms | -55.7 % | gain |
|  | v3 | v32: 330.46 (74 steps); v31: 306.32 (79 steps) | 24.15 | control | | |

### The step on pass 2 (the same request again), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `v3`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `v3` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | v4r | v4r1: 98.24 (108 steps); v4r2: 98.27 (108 steps) | 0.03 | -24.74 ms | -20.1 % | gain |
|  | v3 | v32: 122.98 (75 steps); v31: 123.00 (75 steps) | 0.02 | control | | |
| 4,096 | v4r | v4r1: 109.56 (79 steps); v4r2: 109.49 (79 steps) | 0.07 | -27.61 ms | -20.1 % | gain |
|  | v3 | v32: 137.45 (76 steps); v31: 136.81 (73 steps) | 0.63 | control | | |
| 8,192 | v4r | v4r1: 114.14 (71 steps); v4r2: 114.01 (71 steps) | 0.13 | -37.88 ms | -24.9 % | gain |
|  | v3 | v32: 151.51 (71 steps); v31: 152.40 (67 steps) | 0.89 | control | | |
| 16,384 | v4r | v4r1: 118.11 (69 steps); v4r2: 118.20 (69 steps) | 0.09 | -63.81 ms | -35.1 % | gain |
|  | v3 | v32: 182.51 (71 steps); v31: 181.43 (70 steps) | 1.08 | control | | |
| 32,768 | v4r | v4r1: 127.49 (91 steps); v4r2: 127.48 (91 steps) | 0.01 | -114.54 ms | -47.3 % | gain |
|  | v3 | v32: 241.91 (84 steps); v31: 242.15 (92 steps) | 0.24 | control | | |
| 49,152 | v4r | v4r1: 137.42 (76 steps); v4r2: 137.52 (76 steps) | 0.09 | -162.96 ms | -54.2 % | gain |
|  | v3 | v32: 299.47 (74 steps); v31: 301.39 (79 steps) | 1.91 | control | | |

### Prompt reading, tok/s

Per load: the tokens read at that depth over their time (a prime request reads a chunk on top of the cache; without primes each decode request reads its prompt), with the tokens read and the tokens reused. Requests that read under 32 tokens are left out.

| Depth (tokens) | Arm | Per load | Spread | Against `v3` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | v4r | v4r1: 51.8 (64 read on 0 cached); v4r2: 52.4 (64 read on 0 cached) | 0.5 | +20.1 tok/s | +62.7 % | gain |
|  | v3 | v32: 32.2 (64 read on 0 cached); v31: 31.9 (64 read on 0 cached) | 0.4 | control | | |
| 4,096 | v4r | v4r1: 198.6 (4032 read on 0 cached); v4r2: 195.9 (4032 read on 0 cached) | 2.7 | +73.6 tok/s | +59.6 % | gain |
|  | v3 | v32: 123.6 (4032 read on 0 cached); v31: 123.7 (4032 read on 0 cached) | 0.1 | control | | |
| 8,192 | v4r | v4r1: 219.9 (4612 read on 3516 cached); v4r2: 223.0 (4612 read on 3516 cached) | 3.1 | +102.4 tok/s | +86.0 % | gain |
|  | v3 | v32: 119.0 (4612 read on 3516 cached); v31: 119.1 (4612 read on 3516 cached) | 0.1 | control | | |
| 16,384 | v4r | v4r1: 289.3 (12804 read on 3516 cached); v4r2: 290.7 (12804 read on 3516 cached) | 1.3 | +171.0 tok/s | +143.7 % | gain |
|  | v3 | v32: 118.5 (12804 read on 3516 cached); v31: 119.5 (12804 read on 3516 cached) | 1.0 | control | | |
| 32,768 | v4r | v4r1: 229.1 (16900 read on 15804 cached); v4r2: 229.1 (16900 read on 15804 cached) | 0.0 | +138.3 tok/s | +152.4 % | gain |
|  | v3 | v32: 91.1 (16900 read on 15804 cached); v31: 90.4 (16900 read on 15804 cached) | 0.7 | control | | |
| 49,152 | v4r | v4r1: 180.7 (16900 read on 32188 cached); v4r2: 180.8 (16900 read on 32188 cached) | 0.1 | +110.3 tok/s | +156.5 % | gain |
|  | v3 | v32: 70.6 (16900 read on 32188 cached); v31: 70.3 (16900 read on 32188 cached) | 0.2 | control | | |

### The draft and the step

Per load, pass 1: draft tokens accepted of drafted (the share), tokens a step (generated tokens over steps, a step being one pass of the main model: generated minus accepted), and the time of a step. Pass 2 is listed where its counters differ.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | v4r | v4r1: 147 of 321 (0.458), 2.35 a step, 102.5 ms a step (pass 2: 148 of 318 (0.465), 2.37 a step, 98.2 ms a step); v4r2: 147 of 321 (0.458), 2.35 a step, 102.6 ms a step (pass 2: 148 of 318 (0.465), 2.37 a step, 98.3 ms a step) |
|  | v3 | v32: 178 of 229 (0.777), 3.28 a step, 127.0 ms a step (pass 2: 181 of 220 (0.823), 3.41 a step, 123.0 ms a step); v31: 178 of 229 (0.777), 3.28 a step, 127.5 ms a step (pass 2: 181 of 220 (0.823), 3.41 a step, 123.0 ms a step) |
| 4,096 | v4r | v4r1: 177 of 231 (0.766), 3.24 a step, 115.2 ms a step; v4r2: 177 of 231 (0.766), 3.24 a step, 113.9 ms a step |
|  | v3 | v32: 180 of 223 (0.807), 3.37 a step, 222.2 ms a step; v31: 183 of 213 (0.859), 3.51 a step, 236.1 ms a step |
| 8,192 | v4r | v4r1: 184 of 212 (0.868), 3.56 a step, 118.3 ms a step (pass 2: 185 of 209 (0.885), 3.61 a step, 114.1 ms a step); v4r2: 184 of 212 (0.868), 3.56 a step, 118.4 ms a step (pass 2: 185 of 209 (0.885), 3.61 a step, 114.0 ms a step) |
|  | v3 | v32: 184 of 210 (0.876), 3.56 a step, 171.0 ms a step (pass 2: 185 of 207 (0.894), 3.61 a step, 151.5 ms a step); v31: 189 of 196 (0.964), 3.82 a step, 154.6 ms a step |
| 16,384 | v4r | v4r1: 187 of 203 (0.921), 3.71 a step, 122.1 ms a step; v4r2: 187 of 203 (0.921), 3.71 a step, 122.1 ms a step |
|  | v3 | v32: 185 of 210 (0.881), 3.61 a step, 205.3 ms a step; v31: 186 of 204 (0.912), 3.66 a step, 206.6 ms a step |
| 32,768 | v4r | v4r1: 164 of 273 (0.601), 2.78 a step, 130.8 ms a step (pass 2: 165 of 270 (0.611), 2.81 a step, 127.5 ms a step); v4r2: 164 of 273 (0.601), 2.78 a step, 131.2 ms a step (pass 2: 165 of 270 (0.611), 2.81 a step, 127.5 ms a step) |
|  | v3 | v32: 171 of 252 (0.679), 3.01 a step, 247.0 ms a step (pass 2: 172 of 249 (0.691), 3.05 a step, 241.9 ms a step); v31: 164 of 271 (0.605), 2.78 a step, 248.3 ms a step |
| 49,152 | v4r | v4r1: 180 of 225 (0.800), 3.37 a step, 140.3 ms a step; v4r2: 180 of 225 (0.800), 3.37 a step, 141.5 ms a step |
|  | v3 | v32: 182 of 216 (0.843), 3.46 a step, 330.5 ms a step; v31: 177 of 233 (0.760), 3.24 a step, 306.3 ms a step |

### The sampler pass

One request per depth and load with the server line's own sampler and a fixed seed: tok/s, the draft counters and the step.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | v4r | v4r1: 24.05 tok/s, 154 of 302 (0.510), 2.51 a step, 104.4 ms a step; v4r2: 24.07 tok/s, 154 of 302 (0.510), 2.51 a step, 104.3 ms a step |
|  | v3 | v32: 16.56 tok/s, 146 of 324 (0.451), 2.33 a step, 140.6 ms a step; v31: 16.50 tok/s, 146 of 324 (0.451), 2.33 a step, 141.0 ms a step |
| 4,096 | v4r | v4r1: 29.91 tok/s, 179 of 228 (0.785), 3.32 a step, 111.2 ms a step; v4r2: 29.77 tok/s, 179 of 228 (0.785), 3.32 a step, 111.7 ms a step |
|  | v3 | v32: 19.59 tok/s, 166 of 265 (0.626), 2.84 a step, 145.2 ms a step; v31: 17.86 tok/s, 157 of 291 (0.540), 2.59 a step, 144.8 ms a step |
| 8,192 | v4r | v4r1: 29.44 tok/s, 182 of 216 (0.843), 3.46 a step, 117.5 ms a step; v4r2: 29.73 tok/s, 182 of 216 (0.843), 3.46 a step, 116.4 ms a step |
|  | v3 | v32: 16.83 tok/s, 161 of 282 (0.571), 2.69 a step, 160.1 ms a step; v31: 18.93 tok/s, 171 of 252 (0.679), 3.01 a step, 159.1 ms a step |
| 16,384 | v4r | v4r1: 27.25 tok/s, 177 of 233 (0.760), 3.24 a step, 118.9 ms a step; v4r2: 27.22 tok/s, 177 of 233 (0.760), 3.24 a step, 119.1 ms a step |
|  | v3 | v32: 17.98 tok/s, 179 of 226 (0.792), 3.32 a step, 184.9 ms a step; v31: 18.90 tok/s, 182 of 216 (0.843), 3.46 a step, 183.1 ms a step |
| 32,768 | v4r | v4r1: 22.34 tok/s, 168 of 261 (0.644), 2.91 a step, 130.2 ms a step; v4r2: 22.36 tok/s, 168 of 261 (0.644), 2.91 a step, 130.1 ms a step |
|  | v3 | v32: 12.39 tok/s, 171 of 249 (0.687), 3.01 a step, 243.2 ms a step; v31: 11.76 tok/s, 173 of 245 (0.706), 3.08 a step, 262.2 ms a step |
| 49,152 | v4r | v4r1: 23.72 tok/s, 179 of 225 (0.796), 3.32 a step, 140.2 ms a step; v4r2: 23.73 tok/s, 179 of 225 (0.796), 3.32 a step, 140.1 ms a step |
|  | v3 | v32: 10.37 tok/s, 175 of 238 (0.735), 3.16 a step, 304.7 ms a step; v31: 11.01 tok/s, 179 of 225 (0.796), 3.32 a step, 302.1 ms a step |

### Checks

- Times: every load started on a machine checked quiet (valid=1).
- Prompts: 64 ids at depth 0, 4096 ids at depth 4,096, 8192 ids at depth 8,192, 16384 ids at depth 16,384, 32768 ids at depth 32,768, 49152 ids at depth 49,152; the same ids on every load and pass.
- Tokens out: 256 on every timed request.
- Greedy reply at depth 0: **differs from `v3` in v4r.**
- Greedy reply at depth 4,096: 3 different replies over the 4 loads (`v4r`: 1 over its 2; `v3`: 2 over its 2); every load wrote its own reply again on pass 2.
- Greedy reply at depth 8,192: 3 different replies over the 4 loads (`v4r`: 1 over its 2; `v3`: 2 over its 2); every load wrote its own reply again on pass 2.
- Greedy reply at depth 16,384: 3 different replies over the 4 loads (`v4r`: 1 over its 2; `v3`: 2 over its 2); every load wrote its own reply again on pass 2.
- Greedy reply at depth 32,768: 3 different replies over the 4 loads (`v4r`: 1 over its 2; `v3`: 2 over its 2); every load wrote its own reply again on pass 2.
- Greedy reply at depth 49,152: 3 different replies over the 4 loads (`v4r`: 1 over its 2; `v3`: 2 over its 2); every load wrote its own reply again on pass 2.
- Primes (read on cached, the cache's source): 4,096: 4032 on 0 (none); 8,192: 4612 on 3516 (checkpoint); 16,384: 12804 on 3516 (checkpoint); 32,768: 16900 on 15804 (checkpoint); 49,152: 16900 on 32188 (checkpoint).
- Graph reallocations of the target, requests with one per load: v4r1: 0 of 24 requests; v32: 0 of 24 requests; v4r2: 0 of 24 requests; v31: 0 of 24 requests.
- Stopped by the RAM watchdog (1 stop, each load's files set aside): `v31` (v3) after the prime at depth 4,096 (MemAvailable 55 MiB, below 60), `stopped/v31-1`, run again.
