## The deep lane: kv, dev, kvdev, check

Model file `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf`, llama-server version: 0.9.0-dev (build 12661, commit d097d6d88a).

256 tokens per request at temperature 0, each request twice per load (pass 1 is the first request at a new depth, pass 2 the same request again). Timed loads, in their order: kv1, dev1, kvdev1, kvdev2, dev2, kv2, check.

| Arm | Loads | Draft | Context | Its own variables and flags |
|---|---|---|---:|---|
| kv (control) | kv1, kv2 | on | 65536 | `LLAMA_MTP_KV_ONLY=1` |
| dev | dev1, dev2 | on | 65536 | `LLAMA_QSA_KPOOL_DEV=1 LLAMA_QSA_KPOOL_STAGE=1` |
| kvdev | kvdev1, kvdev2 | on | 65536 | `LLAMA_MTP_KV_ONLY=1 LLAMA_QSA_KPOOL_DEV=1 LLAMA_QSA_KPOOL_STAGE=1` |
| check | check | on | 65536 | `LLAMA_MTP_KV_ONLY=1 LLAMA_QSA_KPOOL_DEV=1 LLAMA_QSA_KPOOL_STAGE=1 LLAMA_QSA_KPOOL_DEV_CHECK=1`, `-lv 4` |

### Generation, tok/s

Per load: pass 1, pass 2 → the load, with the load's time per token. A load's speed is its two passes' tokens over their time. Spread: the difference between the arm's fastest and slowest load. Class against `kv`: gain when the mean is above the control's by more than the larger of the two spreads, loss when below by more than it, else negligible.

| Depth (tokens) | Arm | Per load | Spread | Against `kv` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | kv | kv1: 26.29, 28.30 → **27.26** (36.69 ms a token); kv2: 26.36, 28.30 → **27.29** (36.64 ms a token) | 0.03 | control | | |
|  | dev | dev1: 25.93, 27.99 → **26.92** (37.15 ms a token); dev2: 26.13, 28.05 → **27.06** (36.96 ms a token) | 0.14 | -0.29 tok/s | -1.0 % | loss |
|  | kvdev | kvdev1: 26.34, 28.36 → **27.31** (36.61 ms a token); kvdev2: 26.34, 28.35 → **27.31** (36.62 ms a token) | 0.00 | +0.04 tok/s | +0.1 % | gain |
|  | check | check: 22.69, 23.89 → **23.27** (42.97 ms a token) | — | -4.00 tok/s | -14.7 % | no class (one load) |
| 16,384 | kv | kv1: 19.56, 24.13 → **21.61** (46.29 ms a token); kv2: 17.03, 24.11 → **19.96** (50.11 ms a token) | 1.65 | control | | |
|  | dev | dev1: 20.33, 24.25 → **22.12** (45.22 ms a token); dev2: 21.52, 24.26 → **22.81** (43.84 ms a token) | 0.69 | +1.68 tok/s | +8.1 % | gain |
|  | kvdev | kvdev1: 17.30, 24.41 → **20.25** (49.38 ms a token); kvdev2: 21.67, 24.44 → **22.97** (43.53 ms a token) | 2.72 | +0.83 tok/s | +4.0 % | negligible |
|  | check | check: 19.24, 21.13 → **20.14** (49.66 ms a token) | — | -0.64 tok/s | -3.1 % | no class (one load) |
| 49,152 | kv | kv1: 15.50, 19.35 → **17.21** (58.10 ms a token); kv2: 17.02, 19.35 → **18.11** (55.22 ms a token) | 0.90 | control | | |
|  | dev | dev1: 16.47, 19.64 → **17.92** (55.81 ms a token); dev2: 16.56, 19.44 → **17.88** (55.91 ms a token) | 0.03 | +0.24 tok/s | +1.4 % | negligible |
|  | kvdev | kvdev1: 16.02, 19.61 → **17.63** (56.71 ms a token); kvdev2: 16.62, 19.94 → **18.13** (55.16 ms a token) | 0.50 | +0.22 tok/s | +1.2 % | negligible |
|  | check | check: 15.12, 17.50 → **16.22** (61.64 ms a token) | — | -1.44 tok/s | -8.1 % | no class (one load) |

### Generation over all depths, tok/s

Every depth's tokens over their time, per load.

| Depth (tokens) | Arm | Per load | Spread | Against `kv` | | Class |
|---:|---|---|---:|---:|---:|---|
| all | kv | kv1: **21.27**; kv2: **21.13** | 0.13 | control | | |
|  | dev | dev1: **21.71**; dev2: **21.94** | 0.23 | +0.63 tok/s | +3.0 % | gain |
|  | kvdev | kvdev1: **21.02**; kvdev2: **22.17** | 1.15 | +0.40 tok/s | +1.9 % | negligible |
|  | check | check: **19.45** | — | -1.75 tok/s | -8.3 % | no class (one load) |

### The step on pass 1 (the first request at a new depth), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `kv`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `kv` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | kv | kv1: 124.82 (78 steps); kv2: 124.53 (78 steps) | 0.29 | control | | |
|  | dev | dev1: 126.57 (78 steps); dev2: 125.59 (78 steps) | 0.97 | +1.41 ms | +1.1 % | loss |
|  | kvdev | kvdev1: 124.61 (78 steps); kvdev2: 124.58 (78 steps) | 0.03 | -0.08 ms | -0.1 % | negligible |
|  | check | check: 144.63 (78 steps) | — | +19.96 ms | +16.0 % | no class (one load) |
| 16,384 | kv | kv1: 181.80 (72 steps); kv2: 208.84 (72 steps) | 27.04 | control | | |
|  | dev | dev1: 174.91 (72 steps); dev2: 165.22 (72 steps) | 9.68 | -25.25 ms | -12.9 % | negligible |
|  | kvdev | kvdev1: 205.49 (72 steps); kvdev2: 164.09 (72 steps) | 41.40 | -10.52 ms | -5.4 % | negligible |
|  | check | check: 184.82 (72 steps) | — | -10.50 ms | -5.4 % | no class (one load) |
| 49,152 | kv | kv1: 223.21 (74 steps); kv2: 203.24 (74 steps) | 19.97 | control | | |
|  | dev | dev1: 210.02 (74 steps); dev2: 208.86 (74 steps) | 1.16 | -3.78 ms | -1.8 % | negligible |
|  | kvdev | kvdev1: 215.96 (74 steps); kvdev2: 208.16 (74 steps) | 7.80 | -1.16 ms | -0.5 % | negligible |
|  | check | check: 228.76 (74 steps) | — | +15.54 ms | +7.3 % | no class (one load) |

### The step on pass 2 (the same request again), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `kv`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `kv` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | kv | kv1: 120.63 (75 steps); kv2: 120.62 (75 steps) | 0.01 | control | | |
|  | dev | dev1: 121.95 (75 steps); dev2: 121.69 (75 steps) | 0.26 | +1.19 ms | +1.0 % | loss |
|  | kvdev | kvdev1: 120.35 (75 steps); kvdev2: 120.40 (75 steps) | 0.05 | -0.25 ms | -0.2 % | gain |
|  | check | check: 142.90 (75 steps) | — | +22.27 ms | +18.5 % | no class (one load) |
| 16,384 | kv | kv1: 147.34 (72 steps); kv2: 147.48 (72 steps) | 0.14 | control | | |
|  | dev | dev1: 146.63 (72 steps); dev2: 146.54 (72 steps) | 0.09 | -0.82 ms | -0.6 % | gain |
|  | kvdev | kvdev1: 145.67 (72 steps); kvdev2: 145.47 (72 steps) | 0.20 | -1.83 ms | -1.2 % | gain |
|  | check | check: 168.28 (72 steps) | — | +20.88 ms | +14.2 % | no class (one load) |
| 49,152 | kv | kv1: 178.75 (74 steps); kv2: 178.80 (74 steps) | 0.05 | control | | |
|  | dev | dev1: 176.12 (74 steps); dev2: 178.00 (74 steps) | 1.87 | -1.71 ms | -1.0 % | negligible |
|  | kvdev | kvdev1: 176.41 (74 steps); kvdev2: 173.47 (74 steps) | 2.93 | -3.84 ms | -2.1 % | gain |
|  | check | check: 197.71 (74 steps) | — | +18.94 ms | +10.6 % | no class (one load) |

### Prompt reading, tok/s

Per load: the tokens read at that depth over their time (a prime request reads a chunk on top of the cache; without primes each decode request reads its prompt), with the tokens read and the tokens reused. Requests that read under 32 tokens are left out.

| Depth (tokens) | Arm | Per load | Spread | Against `kv` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | kv | kv1: 32.8 (64 read on 0 cached); kv2: 32.8 (64 read on 0 cached) | 0.0 | control | | |
|  | dev | dev1: 32.0 (64 read on 0 cached); dev2: 32.4 (64 read on 0 cached) | 0.4 | -0.6 tok/s | -1.7 % | loss |
|  | kvdev | kvdev1: 32.4 (64 read on 0 cached); kvdev2: 32.8 (64 read on 0 cached) | 0.3 | -0.2 tok/s | -0.5 % | negligible |
|  | check | check: 32.1 (64 read on 0 cached) | — | -0.6 tok/s | -1.9 % | no class (one load) |
| 16,384 | kv | kv1: 141.4 (16320 read on 0 cached); kv2: 141.2 (16320 read on 0 cached) | 0.2 | control | | |
|  | dev | dev1: 129.0 (16320 read on 0 cached); dev2: 131.1 (16320 read on 0 cached) | 2.2 | -11.3 tok/s | -8.0 % | loss |
|  | kvdev | kvdev1: 141.7 (16320 read on 0 cached); kvdev2: 142.6 (16320 read on 0 cached) | 0.9 | +0.8 tok/s | +0.6 % | negligible |
|  | check | check: 72.4 (16320 read on 0 cached) | — | -69.0 tok/s | -48.8 % | no class (one load) |
| 49,152 | kv | kv1: 92.8 (33284 read on 15804 cached); kv2: 92.8 (33284 read on 15804 cached) | 0.0 | control | | |
|  | dev | dev1: 82.7 (33284 read on 15804 cached); dev2: 82.7 (33284 read on 15804 cached) | 0.1 | -10.1 tok/s | -10.9 % | loss |
|  | kvdev | kvdev1: 92.8 (33284 read on 15804 cached); kvdev2: 92.1 (33284 read on 15804 cached) | 0.6 | -0.3 tok/s | -0.4 % | negligible |
|  | check | check: 45.5 (33284 read on 15804 cached) | — | -47.3 tok/s | -51.0 % | no class (one load) |

### The draft and the step

Per load, pass 1: draft tokens accepted of drafted (the share), tokens a step (generated tokens over steps, a step being one pass of the main model: generated minus accepted), and the time of a step. Pass 2 is listed where its counters differ.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | kv | kv1: 178 of 229 (0.777), 3.28 a step, 124.8 ms a step (pass 2: 181 of 220 (0.823), 3.41 a step, 120.6 ms a step); kv2: 178 of 229 (0.777), 3.28 a step, 124.5 ms a step (pass 2: 181 of 220 (0.823), 3.41 a step, 120.6 ms a step) |
|  | dev | dev1: 178 of 229 (0.777), 3.28 a step, 126.6 ms a step (pass 2: 181 of 220 (0.823), 3.41 a step, 121.9 ms a step); dev2: 178 of 229 (0.777), 3.28 a step, 125.6 ms a step (pass 2: 181 of 220 (0.823), 3.41 a step, 121.7 ms a step) |
|  | kvdev | kvdev1: 178 of 229 (0.777), 3.28 a step, 124.6 ms a step (pass 2: 181 of 220 (0.823), 3.41 a step, 120.4 ms a step); kvdev2: 178 of 229 (0.777), 3.28 a step, 124.6 ms a step (pass 2: 181 of 220 (0.823), 3.41 a step, 120.4 ms a step) |
|  | check | check: 178 of 229 (0.777), 3.28 a step, 144.6 ms a step (pass 2: 181 of 220 (0.823), 3.41 a step, 142.9 ms a step) |
| 16,384 | kv | kv1: 184 of 210 (0.876), 3.56 a step, 181.8 ms a step; kv2: 184 of 210 (0.876), 3.56 a step, 208.8 ms a step |
|  | dev | dev1: 184 of 210 (0.876), 3.56 a step, 174.9 ms a step; dev2: 184 of 210 (0.876), 3.56 a step, 165.2 ms a step |
|  | kvdev | kvdev1: 184 of 210 (0.876), 3.56 a step, 205.5 ms a step; kvdev2: 184 of 210 (0.876), 3.56 a step, 164.1 ms a step |
|  | check | check: 184 of 210 (0.876), 3.56 a step, 184.8 ms a step |
| 49,152 | kv | kv1: 182 of 218 (0.835), 3.46 a step, 223.2 ms a step; kv2: 182 of 218 (0.835), 3.46 a step, 203.2 ms a step |
|  | dev | dev1: 182 of 218 (0.835), 3.46 a step, 210.0 ms a step; dev2: 182 of 218 (0.835), 3.46 a step, 208.9 ms a step |
|  | kvdev | kvdev1: 182 of 218 (0.835), 3.46 a step, 216.0 ms a step; kvdev2: 182 of 218 (0.835), 3.46 a step, 208.2 ms a step |
|  | check | check: 182 of 218 (0.835), 3.46 a step, 228.8 ms a step |

### Checks

- Times: every load started on a machine checked quiet (valid=1).
- Prompts: 64 ids at depth 0, 16384 ids at depth 16,384, 49152 ids at depth 49,152; the same ids on every load and pass.
- Tokens out: 256 on every timed request.
- Greedy reply at depth 0: the same on every pass and load of every arm.
- Greedy reply at depth 16,384: the same on every pass and load of every arm.
- Greedy reply at depth 49,152: the same on every pass and load of every arm.
- Primes (read on cached, the cache's source): 16,384: 16320 on 0 (none); 49,152: 33284 on 15804 (checkpoint).
