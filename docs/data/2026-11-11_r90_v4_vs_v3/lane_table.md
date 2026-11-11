## The deep lane: v3, gains, off

Model file `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf`, llama-server version: 0.9.0-dev (build 12640, commit 18d0fa3099).

256 tokens per request at temperature 0, each request twice per load (pass 1 is the first request at a new depth, pass 2 the same request again). Timed loads, in their order: v31, gains1, v32, gains2, off1, off2.

| Arm | Loads | Draft | Context | Its own variables and flags |
|---|---|---|---:|---|
| v3 (control) | v31, v32 | on | 65536 | — |
| gains | gains1, gains2 | on | 65536 | `GGML_CUDA_FATTN_SPARSE_PASCAL=1 GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=1.0 GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=5 GGML_CUDA_MMVQ_PASCAL_DMT=1 GGML_CUDA_MMVQ_PASCAL_Q5K_MT=1 LLAMA_SAMPLING_TOP_K_ROWS=1 LLAMA_QSA_INDEXER_ROWS=1 LLAMA_QSA_KPOOL_DEV=1`, `--spec-draft-sampling probabilistic --spec-draft-p-min 0 --ctx-checkpoints-vram 4 --ctx-checkpoints-incall --ubatch-taper on --spec-draft-n-ubatch 512` |
| off | off1, off2 | on | 65536 | — |

### Generation, tok/s

Per load: pass 1, pass 2 → the load, with the load's time per token. A load's speed is its two passes' tokens over their time. Spread: the difference between the arm's fastest and slowest load. Class against `v3`: gain when the mean is above the control's by more than the larger of the two spreads, loss when below by more than it, else negligible.

| Depth (tokens) | Arm | Per load | Spread | Against `v3` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | v3 | v31: 26.27, 27.60 → **26.92** (37.15 ms a token); v32: 25.71, 27.59 → **26.62** (37.57 ms a token) | 0.30 | control | | |
|  | gains | gains1: 26.54, 27.78 → **27.15** (36.84 ms a token); gains2: 26.54, 27.68 → **27.10** (36.90 ms a token) | 0.04 | +0.35 tok/s | +1.3 % | gain |
|  | off | off1: 26.62, 28.06 → **27.32** (36.60 ms a token); off2: 26.07, 27.97 → **26.99** (37.05 ms a token) | 0.34 | +0.39 tok/s | +1.4 % | gain |
| 8,192 | v3 | v31: 16.27, 24.11 → **19.43** (51.47 ms a token); v32: 16.32, 23.62 → **19.30** (51.80 ms a token) | 0.12 | control | | |
|  | gains | gains1: 18.47, 23.96 → **20.86** (47.94 ms a token); gains2: 18.48, 24.03 → **20.90** (47.86 ms a token) | 0.04 | +1.51 tok/s | +7.8 % | gain |
|  | off | off1: 14.28, 24.37 → **18.01** (55.52 ms a token); off2: 20.92, 24.29 → **22.48** (44.49 ms a token) | 4.47 | +0.88 tok/s | +4.5 % | negligible |
| 16,384 | v3 | v31: 17.28, 19.59 → **18.36** (54.46 ms a token); v32: 14.37, 19.68 → **16.61** (60.19 ms a token) | 1.75 | control | | |
|  | gains | gains1: 24.74, 26.24 → **25.47** (39.26 ms a token); gains2: 24.62, 26.19 → **25.38** (39.40 ms a token) | 0.09 | +7.94 tok/s | +45.4 % | gain |
|  | off | off1: 15.29, 18.34 → **16.68** (59.96 ms a token); off2: 14.15, 18.35 → **15.98** (62.58 ms a token) | 0.70 | -1.16 tok/s | -6.6 % | negligible |
| 32,768 | v3 | v31: 11.13, 12.06 → **11.58** (86.38 ms a token); v32: 10.42, 11.28 → **10.83** (92.32 ms a token) | 0.74 | control | | |
|  | gains | gains1: 19.15, 20.31 → **19.71** (50.73 ms a token); gains2: 19.14, 20.23 → **19.67** (50.84 ms a token) | 0.04 | +8.49 tok/s | +75.8 % | gain |
|  | off | off1: 10.79, 12.22 → **11.46** (87.25 ms a token); off2: 10.96, 12.21 → **11.55** (86.59 ms a token) | 0.09 | +0.30 tok/s | +2.7 % | negligible |
| 49,152 | v3 | v31: 11.60, 11.76 → **11.68** (85.60 ms a token); v32: 10.02, 10.72 → **10.36** (96.54 ms a token) | 1.32 | control | | |
|  | gains | gains1: 19.80, 22.48 → **21.06** (47.49 ms a token); gains2: 18.12, 22.44 → **20.05** (49.88 ms a token) | 1.01 | +9.53 tok/s | +86.5 % | gain |
|  | off | off1: 11.24, 11.40 → **11.32** (88.32 ms a token); off2: 11.22, 11.40 → **11.31** (88.45 ms a token) | 0.02 | +0.29 tok/s | +2.7 % | negligible |

### Generation over all depths, tok/s

Every depth's tokens over their time, per load.

| Depth (tokens) | Arm | Per load | Spread | Against `v3` | | Class |
|---:|---|---|---:|---:|---:|---|
| all | v3 | v31: **15.87**; v32: **14.77** | 1.09 | control | | |
|  | gains | gains1: **22.50**; gains2: **22.24** | 0.26 | +7.04 tok/s | +46.0 % | gain |
|  | off | off1: **15.26**; off2: **15.67** | 0.41 | +0.14 tok/s | +0.9 % | negligible |

### The step on pass 1 (the first request at a new depth), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `v3`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `v3` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | v3 | v31: 124.96 (78 steps); v32: 127.65 (78 steps) | 2.69 | control | | |
|  | gains | gains1: 112.16 (86 steps); gains2: 112.15 (86 steps) | 0.02 | -14.15 ms | -11.2 % | gain |
|  | off | off1: 123.27 (78 steps); off2: 125.89 (78 steps) | 2.62 | -1.72 ms | -1.4 % | negligible |
| 8,192 | v3 | v31: 234.86 (67 steps); v32: 217.84 (72 steps) | 17.02 | control | | |
|  | gains | gains1: 155.69 (89 steps); gains2: 155.62 (89 steps) | 0.07 | -70.69 ms | -31.2 % | gain |
|  | off | off1: 256.03 (70 steps); off2: 174.85 (70 steps) | 81.19 | -10.91 ms | -4.8 % | negligible |
| 16,384 | v3 | v31: 208.69 (71 steps); v32: 254.45 (70 steps) | 45.76 | control | | |
|  | gains | gains1: 134.40 (77 steps); gains2: 135.02 (77 steps) | 0.62 | -96.85 ms | -41.8 % | gain |
|  | off | off1: 214.60 (78 steps); off2: 231.94 (78 steps) | 17.34 | -8.30 ms | -3.6 % | negligible |
| 32,768 | v3 | v31: 264.26 (87 steps); v32: 261.33 (94 steps) | 2.94 | control | | |
|  | gains | gains1: 145.29 (92 steps); gains2: 145.39 (92 steps) | 0.11 | -117.45 ms | -44.7 % | gain |
|  | off | off1: 269.51 (88 steps); off2: 265.46 (88 steps) | 4.05 | +4.69 ms | +1.8 % | loss |
| 49,152 | v3 | v31: 306.48 (72 steps); v32: 319.27 (80 steps) | 12.79 | control | | |
|  | gains | gains1: 170.14 (76 steps); gains2: 185.90 (76 steps) | 15.76 | -134.86 ms | -43.1 % | gain |
|  | off | off1: 303.56 (75 steps); off2: 304.26 (75 steps) | 0.70 | -8.97 ms | -2.9 % | negligible |

### The step on pass 2 (the same request again), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `v3`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `v3` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | v3 | v31: 123.66 (75 steps); v32: 123.70 (75 steps) | 0.04 | control | | |
|  | gains | gains1: 108.42 (85 steps); gains2: 108.80 (85 steps) | 0.38 | -15.07 ms | -12.2 % | gain |
|  | off | off1: 121.64 (75 steps); off2: 122.03 (75 steps) | 0.38 | -1.85 ms | -1.5 % | gain |
| 8,192 | v3 | v31: 158.48 (67 steps); v32: 152.65 (71 steps) | 5.83 | control | | |
|  | gains | gains1: 127.22 (84 steps); gains2: 126.80 (84 steps) | 0.42 | -28.55 ms | -18.4 % | gain |
|  | off | off1: 152.27 (69 steps); off2: 152.73 (69 steps) | 0.46 | -3.07 ms | -2.0 % | negligible |
| 16,384 | v3 | v31: 184.07 (71 steps); v32: 185.79 (70 steps) | 1.72 | control | | |
|  | gains | gains1: 130.06 (75 steps); gains2: 130.33 (75 steps) | 0.27 | -54.74 ms | -29.6 % | gain |
|  | off | off1: 181.30 (77 steps); off2: 181.18 (77 steps) | 0.11 | -3.69 ms | -2.0 % | gain |
| 32,768 | v3 | v31: 244.09 (87 steps); v32: 244.11 (93 steps) | 0.02 | control | | |
|  | gains | gains1: 140.05 (90 steps); gains2: 140.58 (90 steps) | 0.53 | -103.78 ms | -42.5 % | gain |
|  | off | off1: 240.89 (87 steps); off2: 241.08 (87 steps) | 0.19 | -3.11 ms | -1.3 % | gain |
| 49,152 | v3 | v31: 302.25 (72 steps); v32: 302.37 (79 steps) | 0.12 | control | | |
|  | gains | gains1: 149.81 (76 steps); gains2: 150.10 (76 steps) | 0.29 | -152.36 ms | -50.4 % | gain |
|  | off | off1: 299.38 (75 steps); off2: 299.53 (75 steps) | 0.15 | -2.86 ms | -0.9 % | gain |

### Prompt reading, tok/s

Per load: the tokens read at that depth over their time (a prime request reads a chunk on top of the cache; without primes each decode request reads its prompt), with the tokens read and the tokens reused. Requests that read under 32 tokens are left out.

| Depth (tokens) | Arm | Per load | Spread | Against `v3` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | v3 | v31: 32.8 (64 read on 0 cached); v32: 31.7 (64 read on 0 cached) | 1.1 | control | | |
|  | gains | gains1: 53.6 (64 read on 0 cached); gains2: 52.1 (64 read on 0 cached) | 1.6 | +20.6 tok/s | +64.0 % | gain |
|  | off | off1: 33.1 (64 read on 0 cached); off2: 32.8 (64 read on 0 cached) | 0.3 | +0.7 tok/s | +2.2 % | negligible |
| 8,192 | v3 | v31: 124.6 (8128 read on 0 cached); v32: 125.7 (8128 read on 0 cached) | 1.1 | control | | |
|  | gains | gains1: 142.0 (8128 read on 0 cached); gains2: 142.0 (8128 read on 0 cached) | 0.1 | +16.8 tok/s | +13.4 % | gain |
|  | off | off1: 133.0 (8128 read on 0 cached); off2: 137.7 (8128 read on 0 cached) | 4.7 | +10.2 tok/s | +8.2 % | gain |
| 16,384 | v3 | v31: 108.5 (8708 read on 7612 cached); v32: 108.4 (8708 read on 7612 cached) | 0.1 | control | | |
|  | gains | gains1: 125.8 (8708 read on 7612 cached); gains2: 125.0 (8708 read on 7612 cached) | 0.8 | +16.9 tok/s | +15.6 % | gain |
|  | off | off1: 113.3 (8708 read on 7612 cached); off2: 113.9 (8708 read on 7612 cached) | 0.6 | +5.1 tok/s | +4.7 % | gain |
| 32,768 | v3 | v31: 90.0 (16388 read on 16316 cached); v32: 90.1 (16388 read on 16316 cached) | 0.1 | control | | |
|  | gains | gains1: 98.5 (16388 read on 16316 cached); gains2: 98.7 (16388 read on 16316 cached) | 0.2 | +8.5 tok/s | +9.5 % | gain |
|  | off | off1: 93.8 (16388 read on 16316 cached); off2: 93.3 (16388 read on 16316 cached) | 0.5 | +3.5 tok/s | +3.9 % | gain |
| 49,152 | v3 | v31: 69.3 (16900 read on 32188 cached); v32: 69.5 (16900 read on 32188 cached) | 0.2 | control | | |
|  | gains | gains1: 75.5 (16900 read on 32188 cached); gains2: 76.5 (16900 read on 32188 cached) | 1.0 | +6.6 tok/s | +9.5 % | gain |
|  | off | off1: 71.8 (16900 read on 32188 cached); off2: 71.5 (16900 read on 32188 cached) | 0.3 | +2.2 tok/s | +3.2 % | gain |

### The draft and the step

Per load, pass 1: draft tokens accepted of drafted (the share), tokens a step (generated tokens over steps, a step being one pass of the main model: generated minus accepted), and the time of a step. Pass 2 is listed where its counters differ.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | v3 | v31: 178 of 229 (0.777), 3.28 a step, 125.0 ms a step (pass 2: 181 of 220 (0.823), 3.41 a step, 123.7 ms a step); v32: 178 of 229 (0.777), 3.28 a step, 127.6 ms a step (pass 2: 181 of 220 (0.823), 3.41 a step, 123.7 ms a step) |
|  | gains | gains1: 170 of 253 (0.672), 2.98 a step, 112.2 ms a step (pass 2: 171 of 250 (0.684), 3.01 a step, 108.4 ms a step); gains2: 170 of 253 (0.672), 2.98 a step, 112.1 ms a step (pass 2: 171 of 250 (0.684), 3.01 a step, 108.8 ms a step) |
|  | off | off1: 178 of 229 (0.777), 3.28 a step, 123.3 ms a step (pass 2: 181 of 220 (0.823), 3.41 a step, 121.6 ms a step); off2: 178 of 229 (0.777), 3.28 a step, 125.9 ms a step (pass 2: 181 of 220 (0.823), 3.41 a step, 122.0 ms a step) |
| 8,192 | v3 | v31: 189 of 196 (0.964), 3.82 a step, 234.9 ms a step; v32: 184 of 210 (0.876), 3.56 a step, 217.8 ms a step (pass 2: 185 of 207 (0.894), 3.61 a step, 152.6 ms a step) |
|  | gains | gains1: 167 of 262 (0.637), 2.88 a step, 155.7 ms a step (pass 2: 172 of 247 (0.696), 3.05 a step, 127.2 ms a step); gains2: 167 of 262 (0.637), 2.88 a step, 155.6 ms a step (pass 2: 172 of 247 (0.696), 3.05 a step, 126.8 ms a step) |
|  | off | off1: 186 of 207 (0.899), 3.66 a step, 256.0 ms a step (pass 2: 187 of 204 (0.917), 3.71 a step, 152.3 ms a step); off2: 186 of 207 (0.899), 3.66 a step, 174.8 ms a step (pass 2: 187 of 204 (0.917), 3.71 a step, 152.7 ms a step) |
| 16,384 | v3 | v31: 185 of 210 (0.881), 3.61 a step, 208.7 ms a step; v32: 186 of 207 (0.899), 3.66 a step, 254.4 ms a step |
|  | gains | gains1: 179 of 226 (0.792), 3.32 a step, 134.4 ms a step (pass 2: 181 of 220 (0.823), 3.41 a step, 130.1 ms a step); gains2: 179 of 226 (0.792), 3.32 a step, 135.0 ms a step (pass 2: 181 of 220 (0.823), 3.41 a step, 130.3 ms a step) |
|  | off | off1: 178 of 231 (0.771), 3.28 a step, 214.6 ms a step (pass 2: 179 of 228 (0.785), 3.32 a step, 181.3 ms a step); off2: 178 of 231 (0.771), 3.28 a step, 231.9 ms a step (pass 2: 179 of 228 (0.785), 3.32 a step, 181.2 ms a step) |
| 32,768 | v3 | v31: 169 of 257 (0.658), 2.94 a step, 264.3 ms a step; v32: 162 of 277 (0.585), 2.72 a step, 261.3 ms a step (pass 2: 163 of 274 (0.595), 2.75 a step, 244.1 ms a step) |
|  | gains | gains1: 164 of 273 (0.601), 2.78 a step, 145.3 ms a step (pass 2: 166 of 267 (0.622), 2.84 a step, 140.1 ms a step); gains2: 164 of 273 (0.601), 2.78 a step, 145.4 ms a step (pass 2: 166 of 267 (0.622), 2.84 a step, 140.6 ms a step) |
|  | off | off1: 168 of 257 (0.654), 2.91 a step, 269.5 ms a step (pass 2: 169 of 254 (0.665), 2.94 a step, 240.9 ms a step); off2: 168 of 257 (0.654), 2.91 a step, 265.5 ms a step (pass 2: 169 of 254 (0.665), 2.94 a step, 241.1 ms a step) |
| 49,152 | v3 | v31: 184 of 210 (0.876), 3.56 a step, 306.5 ms a step; v32: 176 of 234 (0.752), 3.20 a step, 319.3 ms a step (pass 2: 177 of 231 (0.766), 3.24 a step, 302.4 ms a step) |
|  | gains | gains1: 180 of 222 (0.811), 3.37 a step, 170.1 ms a step; gains2: 180 of 222 (0.811), 3.37 a step, 185.9 ms a step |
|  | off | off1: 181 of 219 (0.826), 3.41 a step, 303.6 ms a step; off2: 181 of 219 (0.826), 3.41 a step, 304.3 ms a step |

### The sampler pass

One request per depth and load with the server line's own sampler and a fixed seed: tok/s, the draft counters and the step.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | v3 | v31: 17.03 tok/s, 146 of 324 (0.451), 2.33 a step, 136.6 ms a step; v32: 16.12 tok/s, 146 of 324 (0.451), 2.33 a step, 144.4 ms a step |
|  | gains | gains1: 19.08 tok/s, 142 of 339 (0.419), 2.25 a step, 117.7 ms a step; gains2: 19.09 tok/s, 142 of 339 (0.419), 2.25 a step, 117.6 ms a step |
|  | off | off1: 16.65 tok/s, 146 of 324 (0.451), 2.33 a step, 139.7 ms a step; off2: 16.65 tok/s, 146 of 324 (0.451), 2.33 a step, 139.7 ms a step |
| 8,192 | v3 | v31: 19.33 tok/s, 174 of 243 (0.716), 3.12 a step, 161.5 ms a step; v32: 17.11 tok/s, 164 of 273 (0.601), 2.78 a step, 162.6 ms a step |
|  | gains | gains1: 24.66 tok/s, 176 of 237 (0.743), 3.20 a step, 129.8 ms a step; gains2: 24.64 tok/s, 176 of 237 (0.743), 3.20 a step, 129.9 ms a step |
|  | off | off1: 16.10 tok/s, 156 of 295 (0.529), 2.56 a step, 159.0 ms a step; off2: 15.83 tok/s, 156 of 295 (0.529), 2.56 a step, 161.7 ms a step |
| 16,384 | v3 | v31: 17.57 tok/s, 178 of 231 (0.771), 3.28 a step, 186.8 ms a step; v32: 16.10 tok/s, 177 of 233 (0.760), 3.24 a step, 201.2 ms a step |
|  | gains | gains1: 27.15 tok/s, 185 of 209 (0.885), 3.61 a step, 132.8 ms a step; gains2: 27.18 tok/s, 185 of 209 (0.885), 3.61 a step, 132.7 ms a step |
|  | off | off1: 17.86 tok/s, 178 of 228 (0.781), 3.28 a step, 183.8 ms a step; off2: 17.53 tok/s, 178 of 228 (0.781), 3.28 a step, 187.3 ms a step |
| 32,768 | v3 | v31: 10.42 tok/s, 158 of 287 (0.551), 2.61 a step, 250.7 ms a step; v32: 8.73 tok/s, 139 of 346 (0.402), 2.19 a step, 250.6 ms a step |
|  | gains | gains1: 20.39 tok/s, 171 of 252 (0.679), 3.01 a step, 147.7 ms a step; gains2: 20.46 tok/s, 171 of 252 (0.679), 3.01 a step, 147.2 ms a step |
|  | off | off1: 10.68 tok/s, 159 of 288 (0.552), 2.64 a step, 247.1 ms a step; off2: 10.67 tok/s, 159 of 288 (0.552), 2.64 a step, 247.3 ms a step |
| 49,152 | v3 | v31: 8.71 tok/s, 161 of 279 (0.577), 2.69 a step, 309.6 ms a step; v32: 10.57 tok/s, 177 of 232 (0.763), 3.24 a step, 306.6 ms a step |
|  | gains | gains1: 19.75 tok/s, 173 of 244 (0.709), 3.08 a step, 156.2 ms a step; gains2: 19.97 tok/s, 173 of 244 (0.709), 3.08 a step, 154.4 ms a step |
|  | off | off1: 10.91 tok/s, 179 of 227 (0.789), 3.32 a step, 304.7 ms a step; off2: 10.92 tok/s, 179 of 227 (0.789), 3.32 a step, 304.6 ms a step |

### Checks

- Times: every load started on a machine checked quiet (valid=1).
- Prompts: 64 ids at depth 0, 8192 ids at depth 8,192, 16384 ids at depth 16,384, 32768 ids at depth 32,768, 49152 ids at depth 49,152; the same ids on every load and pass.
- Tokens out: 256 on every timed request.
- Greedy reply at depth 0: **differs from `v3` in gains.**
- Greedy reply at depth 8,192: 4 different replies over the 6 loads (`v3`: 2 over its 2; `gains`: 1 over its 2; `off`: 1 over its 2); every load wrote its own reply again on pass 2.
- Greedy reply at depth 16,384: 4 different replies over the 6 loads (`v3`: 2 over its 2; `gains`: 1 over its 2; `off`: 1 over its 2); every load wrote its own reply again on pass 2.
- Greedy reply at depth 32,768: 4 different replies over the 6 loads (`v3`: 2 over its 2; `gains`: 1 over its 2; `off`: 1 over its 2); every load wrote its own reply again on pass 2.
- Greedy reply at depth 49,152: 4 different replies over the 6 loads (`v3`: 2 over its 2; `gains`: 1 over its 2; `off`: 1 over its 2); every load wrote its own reply again on pass 2.
- Primes (read on cached, the cache's source): 8,192: 8128 on 0 (none); 16,384: 8708 on 7612 (checkpoint); 32,768: 16388 on 16316 (checkpoint); 49,152: 16900 on 32188 (checkpoint).
