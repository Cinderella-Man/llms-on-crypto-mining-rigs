## The deep lane: rel, v4c

Model file `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf`, llama-server version: 0.8.0-dev (build 12358, commit 1551f5fe1).

256 tokens per request at temperature 0, each request twice per load (pass 1 is the first request at a new depth, pass 2 the same request again). Timed loads, in their order: rel1, v4c1, rel2, v4c2.

| Arm | Loads | Draft | Context | Its own variables and flags |
|---|---|---|---:|---|
| rel (control) | rel1, rel2 | on | 65536 | — |
| v4c | v4c1, v4c2 | on | 65536 | `GGML_CUDA_FATTN_SPARSE_PASCAL=1 GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=1.0 GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=5`, `--spec-draft-sampling probabilistic --spec-draft-p-min 0 --ctx-checkpoints-vram 4 --ubatch-taper on --spec-draft-n-ubatch 512` |

### Generation, tok/s

Per load: pass 1, pass 2 → the load, with the load's time per token. A load's speed is its two passes' tokens over their time. Spread: the difference between the arm's fastest and slowest load. Class against `rel`: gain when the mean is above the control's by more than the larger of the two spreads, loss when below by more than it, else negligible.

| Depth (tokens) | Arm | Per load | Spread | Against `rel` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | rel | rel1: 26.18, 27.68 → **26.91** (37.16 ms a token); rel2: 23.35, 27.59 → **25.29** (39.54 ms a token) | 1.62 | control | | |
|  | v4c | v4c1: 25.73, 27.60 → **26.63** (37.55 ms a token); v4c2: 25.74, 27.59 → **26.64** (37.54 ms a token) | 0.00 | +0.54 tok/s | +2.1 % | negligible |
| 8,192 | rel | rel1: 12.47, 20.05 → **15.37** (65.05 ms a token); rel2: 22.45, 25.01 → **23.66** (42.26 ms a token) | 8.29 | control | | |
|  | v4c | v4c1: 20.08, 24.72 → **22.16** (45.13 ms a token); v4c2: 18.48, 23.31 → **20.62** (48.50 ms a token) | 1.54 | +1.87 tok/s | +9.6 % | negligible |
| 16,384 | rel | rel1: 16.35, 19.42 → **17.75** (56.33 ms a token); rel2: 19.16, 19.39 → **19.27** (51.88 ms a token) | 1.52 | control | | |
|  | v4c | v4c1: 20.14, 22.66 → **21.32** (46.90 ms a token); v4c2: 23.87, 24.11 → **23.99** (41.68 ms a token) | 2.67 | +4.14 tok/s | +22.4 % | gain |
| 32,768 | rel | rel1: 11.65, 12.29 → **11.96** (83.60 ms a token); rel2: 11.01, 11.66 → **11.32** (88.32 ms a token) | 0.64 | control | | |
|  | v4c | v4c1: 16.34, 17.00 → **16.66** (60.01 ms a token); v4c2: 14.67, 15.42 → **15.03** (66.51 ms a token) | 1.63 | +4.21 tok/s | +36.1 % | gain |
| 49,152 | rel | rel1: 10.27, 10.54 → **10.40** (96.11 ms a token); rel2: 10.55, 10.85 → **10.70** (93.48 ms a token) | 0.29 | control | | |
|  | v4c | v4c1: 18.93, 19.25 → **19.08** (52.40 ms a token); v4c2: 18.14, 18.54 → **18.34** (54.53 ms a token) | 0.75 | +8.16 tok/s | +77.3 % | gain |

### Generation over all depths, tok/s

Every depth's tokens over their time, per load.

| Depth (tokens) | Arm | Per load | Spread | Against `rel` | | Class |
|---:|---|---|---:|---:|---:|---|
| all | rel | rel1: **14.78**; rel2: **15.85** | 1.07 | control | | |
|  | v4c | v4c1: **20.66**; v4c2: **20.10** | 0.56 | +5.07 tok/s | +33.1 % | gain |

### The step on pass 1 (the first request at a new depth), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `rel`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `rel` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | rel | rel1: 125.38 (78 steps); rel2: 140.58 (78 steps) | 15.21 | control | | |
|  | v4c | v4c1: 127.54 (78 steps); v4c2: 127.49 (78 steps) | 0.06 | -5.47 ms | -4.1 % | negligible |
| 8,192 | rel | rel1: 233.36 (88 steps); rel2: 170.19 (67 steps) | 63.17 | control | | |
|  | v4c | v4c1: 174.62 (73 steps); v4c2: 177.58 (78 steps) | 2.95 | -25.68 ms | -12.7 % | negligible |
| 16,384 | rel | rel1: 214.52 (73 steps); rel2: 185.60 (72 steps) | 28.92 | control | | |
|  | v4c | v4c1: 169.51 (75 steps); v4c2: 153.21 (70 steps) | 16.29 | -38.70 ms | -19.3 % | gain |
| 32,768 | rel | rel1: 249.66 (88 steps); rel2: 250.08 (93 steps) | 0.42 | control | | |
|  | v4c | v4c1: 170.26 (92 steps); v4c2: 172.80 (101 steps) | 2.54 | -78.34 ms | -31.4 % | gain |
| 49,152 | rel | rel1: 307.76 (81 steps); rel2: 307.18 (79 steps) | 0.58 | control | | |
|  | v4c | v4c1: 185.29 (73 steps); v4c2: 185.72 (76 steps) | 0.43 | -121.96 ms | -39.7 % | gain |

### The step on pass 2 (the same request again), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `rel`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `rel` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | rel | rel1: 123.31 (75 steps); rel2: 123.74 (75 steps) | 0.43 | control | | |
|  | v4c | v4c1: 123.68 (75 steps); v4c2: 123.70 (75 steps) | 0.02 | +0.17 ms | +0.1 % | negligible |
| 8,192 | rel | rel1: 153.86 (83 steps); rel2: 152.76 (67 steps) | 1.10 | control | | |
|  | v4c | v4c1: 143.85 (72 steps); v4c2: 144.48 (76 steps) | 0.62 | -9.14 ms | -6.0 % | gain |
| 16,384 | rel | rel1: 183.08 (72 steps); rel2: 183.34 (72 steps) | 0.26 | control | | |
|  | v4c | v4c1: 150.66 (75 steps); v4c2: 151.67 (70 steps) | 1.01 | -32.05 ms | -17.5 % | gain |
| 32,768 | rel | rel1: 245.10 (85 steps); rel2: 244.01 (90 steps) | 1.09 | control | | |
|  | v4c | v4c1: 165.51 (91 steps); v4c2: 166.02 (100 steps) | 0.51 | -78.79 ms | -32.2 % | gain |
| 49,152 | rel | rel1: 303.51 (80 steps); rel2: 302.47 (78 steps) | 1.03 | control | | |
|  | v4c | v4c1: 182.21 (73 steps); v4c2: 181.66 (76 steps) | 0.56 | -121.05 ms | -40.0 % | gain |

### Prompt reading, tok/s

Per load: the tokens read at that depth over their time (a prime request reads a chunk on top of the cache; without primes each decode request reads its prompt), with the tokens read and the tokens reused. Requests that read under 32 tokens are left out.

| Depth (tokens) | Arm | Per load | Spread | Against `rel` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | rel | rel1: 32.9 (64 read on 0 cached); rel2: 31.2 (64 read on 0 cached) | 1.6 | control | | |
|  | v4c | v4c1: 48.1 (64 read on 0 cached); v4c2: 48.3 (64 read on 0 cached) | 0.2 | +16.2 tok/s | +50.4 % | gain |
| 8,192 | rel | rel1: 128.5 (8128 read on 0 cached); rel2: 130.1 (8128 read on 0 cached) | 1.6 | control | | |
|  | v4c | v4c1: 127.9 (8128 read on 0 cached); v4c2: 130.0 (8128 read on 0 cached) | 2.1 | -0.4 tok/s | -0.3 % | negligible |
| 16,384 | rel | rel1: 109.6 (8708 read on 7612 cached); rel2: 110.0 (8708 read on 7612 cached) | 0.4 | control | | |
|  | v4c | v4c1: 123.9 (8708 read on 7612 cached); v4c2: 128.6 (8708 read on 7612 cached) | 4.7 | +16.5 tok/s | +15.0 % | gain |
| 32,768 | rel | rel1: 90.8 (16388 read on 16316 cached); rel2: 90.5 (16388 read on 16316 cached) | 0.2 | control | | |
|  | v4c | v4c1: 103.6 (16388 read on 16316 cached); v4c2: 103.7 (16388 read on 16316 cached) | 0.2 | +13.0 tok/s | +14.4 % | gain |
| 49,152 | rel | rel1: 69.9 (16900 read on 32188 cached); rel2: 69.8 (16900 read on 32188 cached) | 0.2 | control | | |
|  | v4c | v4c1: 83.5 (16900 read on 32188 cached); v4c2: 83.1 (16900 read on 32188 cached) | 0.4 | +13.4 tok/s | +19.2 % | gain |

### The draft and the step

Per load, pass 1: draft tokens accepted of drafted (the share), tokens a step (generated tokens over steps, a step being one pass of the main model: generated minus accepted), and the time of a step. Pass 2 is listed where its counters differ.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | rel | rel1: 178 of 229 (0.777), 3.28 a step, 125.4 ms a step (pass 2: 181 of 220 (0.823), 3.41 a step, 123.3 ms a step); rel2: 178 of 229 (0.777), 3.28 a step, 140.6 ms a step (pass 2: 181 of 220 (0.823), 3.41 a step, 123.7 ms a step) |
|  | v4c | v4c1: 178 of 229 (0.777), 3.28 a step, 127.5 ms a step (pass 2: 181 of 220 (0.823), 3.41 a step, 123.7 ms a step); v4c2: 178 of 229 (0.777), 3.28 a step, 127.5 ms a step (pass 2: 181 of 220 (0.823), 3.41 a step, 123.7 ms a step) |
| 8,192 | rel | rel1: 168 of 259 (0.649), 2.91 a step, 233.4 ms a step (pass 2: 173 of 244 (0.709), 3.08 a step, 153.9 ms a step); rel2: 189 of 196 (0.964), 3.82 a step, 170.2 ms a step |
|  | v4c | v4c1: 183 of 216 (0.847), 3.51 a step, 174.6 ms a step (pass 2: 184 of 213 (0.864), 3.56 a step, 143.9 ms a step); v4c2: 178 of 231 (0.771), 3.28 a step, 177.6 ms a step (pass 2: 180 of 225 (0.800), 3.37 a step, 144.5 ms a step) |
| 16,384 | rel | rel1: 183 of 214 (0.855), 3.51 a step, 214.5 ms a step (pass 2: 184 of 211 (0.872), 3.56 a step, 183.1 ms a step); rel2: 184 of 213 (0.864), 3.56 a step, 185.6 ms a step |
|  | v4c | v4c1: 181 of 219 (0.826), 3.41 a step, 169.5 ms a step; v4c2: 186 of 206 (0.903), 3.66 a step, 153.2 ms a step |
| 32,768 | rel | rel1: 168 of 259 (0.649), 2.91 a step, 249.7 ms a step (pass 2: 171 of 248 (0.690), 3.01 a step, 245.1 ms a step); rel2: 163 of 274 (0.595), 2.75 a step, 250.1 ms a step (pass 2: 166 of 265 (0.626), 2.84 a step, 244.0 ms a step) |
|  | v4c | v4c1: 164 of 268 (0.612), 2.78 a step, 170.3 ms a step (pass 2: 165 of 265 (0.623), 2.81 a step, 165.5 ms a step); v4c2: 155 of 298 (0.520), 2.53 a step, 172.8 ms a step (pass 2: 156 of 295 (0.529), 2.56 a step, 166.0 ms a step) |
| 49,152 | rel | rel1: 175 of 240 (0.729), 3.16 a step, 307.8 ms a step (pass 2: 176 of 237 (0.743), 3.20 a step, 303.5 ms a step); rel2: 177 of 232 (0.763), 3.24 a step, 307.2 ms a step (pass 2: 178 of 229 (0.777), 3.28 a step, 302.5 ms a step) |
|  | v4c | v4c1: 183 of 216 (0.847), 3.51 a step, 185.3 ms a step; v4c2: 180 of 224 (0.804), 3.37 a step, 185.7 ms a step |

### The sampler pass

One request per depth and load with the server line's own sampler and a fixed seed: tok/s, the draft counters and the step.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | rel | rel1: 17.04 tok/s, 146 of 324 (0.451), 2.33 a step, 136.6 ms a step; rel2: 15.22 tok/s, 146 of 324 (0.451), 2.33 a step, 152.9 ms a step |
|  | v4c | v4c1: 15.22 tok/s, 138 of 350 (0.394), 2.17 a step, 142.5 ms a step; v4c2: 15.27 tok/s, 138 of 350 (0.394), 2.17 a step, 142.1 ms a step |
| 8,192 | rel | rel1: 17.34 tok/s, 170 of 254 (0.669), 2.98 a step, 171.7 ms a step; rel2: 17.83 tok/s, 167 of 264 (0.633), 2.88 a step, 161.4 ms a step |
|  | v4c | v4c1: 19.89 tok/s, 171 of 250 (0.684), 3.01 a step, 151.4 ms a step; v4c2: 18.87 tok/s, 173 of 242 (0.715), 3.08 a step, 163.5 ms a step |
| 16,384 | rel | rel1: 16.49 tok/s, 173 of 246 (0.703), 3.08 a step, 187.0 ms a step; rel2: 18.98 tok/s, 183 of 216 (0.847), 3.51 a step, 184.7 ms a step |
|  | v4c | v4c1: 24.11 tok/s, 186 of 204 (0.912), 3.66 a step, 151.7 ms a step; v4c2: 21.93 tok/s, 180 of 224 (0.804), 3.37 a step, 153.6 ms a step |
| 32,768 | rel | rel1: 11.27 tok/s, 164 of 270 (0.607), 2.78 a step, 247.0 ms a step; rel2: 10.11 tok/s, 154 of 298 (0.517), 2.51 a step, 248.2 ms a step |
|  | v4c | v4c1: 15.12 tok/s, 158 of 290 (0.545), 2.61 a step, 172.8 ms a step; v4c2: 16.22 tok/s, 164 of 272 (0.603), 2.78 a step, 171.5 ms a step |
| 49,152 | rel | rel1: 9.65 tok/s, 169 of 255 (0.663), 2.94 a step, 304.9 ms a step; rel2: 9.78 tok/s, 171 of 252 (0.679), 3.01 a step, 308.0 ms a step |
|  | v4c | v4c1: 17.84 tok/s, 178 of 228 (0.781), 3.28 a step, 184.0 ms a step; v4c2: 18.56 tok/s, 181 of 220 (0.823), 3.41 a step, 183.9 ms a step |

### Checks

- Times: every load started on a machine checked quiet (valid=1).
- Prompts: 64 ids at depth 0, 8192 ids at depth 8,192, 16384 ids at depth 16,384, 32768 ids at depth 32,768, 49152 ids at depth 49,152; the same ids on every load and pass.
- Tokens out: 256 on every timed request.
- Greedy reply at depth 0: the same on every pass and load of every arm.
- Greedy reply at depth 8,192: 4 different replies over the 4 loads (`rel`: 2 over its 2; `v4c`: 2 over its 2); every load wrote its own reply again on pass 2.
- Greedy reply at depth 16,384: 4 different replies over the 4 loads (`rel`: 2 over its 2; `v4c`: 2 over its 2); every load wrote its own reply again on pass 2.
- Greedy reply at depth 32,768: 5 different replies over the 4 loads (`rel`: 3 over its 2; `v4c`: 2 over its 2); **rel1 wrote another reply on pass 2 than on pass 1.**
- Greedy reply at depth 49,152: 4 different replies over the 4 loads (`rel`: 2 over its 2; `v4c`: 2 over its 2); every load wrote its own reply again on pass 2.
- Primes (read on cached, the cache's source): 8,192: 8128 on 0 (none); 16,384: 8708 on 7612 (checkpoint); 32,768: 16388 on 16316 (checkpoint); 49,152: 16900 on 32188 (checkpoint).
