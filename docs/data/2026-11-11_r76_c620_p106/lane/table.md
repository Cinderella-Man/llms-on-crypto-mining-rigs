## The deep lane: draft, kvonly, draftb

Model file `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf`, llama-server version: 0.9.0-dev (build 12581, commit c408dabcb3).

16 tokens per request at temperature 0, each request twice per load (pass 1 is the first request at a new depth, pass 2 the same request again). Timed loads, in their order: draft1, kvonly1, draft2.

| Arm | Loads | Draft | Context | Its own variables and flags |
|---|---|---|---:|---|
| draft (control) | draft1 | on | 65536 | `LLAMA_SPEC_DUMP=$WORK/lane/out/dump-draft.jsonl` |
| kvonly | kvonly1 | on | 65536 | `LLAMA_MTP_KV_ONLY=1 LLAMA_SPEC_DUMP=$WORK/lane/out/dump-kvonly.jsonl` |
| draftb | draft2 | on | 65536 | `LLAMA_SPEC_DUMP=$WORK/lane/out/dump-draftb.jsonl` |

### Generation, tok/s

Per load: pass 1, pass 2 → the load, with the load's time per token. A load's speed is its two passes' tokens over their time. Spread: the difference between the arm's fastest and slowest load. Class against `draft`: gain when the mean is above the control's by more than the larger of the two spreads, loss when below by more than it, else negligible.

| Depth (tokens) | Arm | Per load | Spread | Against `draft` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | draft | draft1: 6.70, 13.93 → **9.05** (110.56 ms a token) | — | control | | |
|  | kvonly | kvonly1: 6.69, 14.00 → **9.05** (110.47 ms a token) | — | +0.01 tok/s | +0.1 % | no class (one load) |
|  | draftb | draft2: 6.69, 13.92 → **9.04** (110.65 ms a token) | — | -0.01 tok/s | -0.1 % | no class (one load) |
| 32,768 | draft | draft1: 6.47, 12.89 → **8.62** (116.05 ms a token) | — | control | | |
|  | kvonly | kvonly1: 5.18, 13.04 → **7.42** (134.81 ms a token) | — | -1.20 tok/s | -13.9 % | no class (one load) |
|  | draftb | draft2: 6.47, 12.87 → **8.61** (116.12 ms a token) | — | -0.01 tok/s | -0.1 % | no class (one load) |

### Generation over all depths, tok/s

Every depth's tokens over their time, per load.

| Depth (tokens) | Arm | Per load | Spread | Against `draft` | | Class |
|---:|---|---|---:|---:|---:|---|
| all | draft | draft1: **8.83** | — | control | | |
|  | kvonly | kvonly1: **8.15** | — | -0.67 tok/s | -7.6 % | no class (one load) |
|  | draftb | draft2: **8.82** | — | -0.01 tok/s | -0.1 % | no class (one load) |

### The step on pass 1 (the first request at a new depth), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `draft`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `draft` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | draft | draft1: 265.49 (9 steps) | — | control | | |
|  | kvonly | kvonly1: 265.82 (9 steps) | — | +0.34 ms | +0.1 % | no class (one load) |
|  | draftb | draft2: 265.76 (9 steps) | — | +0.27 ms | +0.1 % | no class (one load) |
| 32,768 | draft | draft1: 309.00 (8 steps) | — | control | | |
|  | kvonly | kvonly1: 385.86 (8 steps) | — | +76.87 ms | +24.9 % | no class (one load) |
|  | draftb | draft2: 309.04 (8 steps) | — | +0.05 ms | +0.0 % | no class (one load) |

### The step on pass 2 (the same request again), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `draft`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `draft` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | draft | draft1: 127.60 (9 steps) | — | control | | |
|  | kvonly | kvonly1: 126.98 (9 steps) | — | -0.62 ms | -0.5 % | no class (one load) |
|  | draftb | draft2: 127.67 (9 steps) | — | +0.07 ms | +0.1 % | no class (one load) |
| 32,768 | draft | draft1: 155.19 (8 steps) | — | control | | |
|  | kvonly | kvonly1: 153.39 (8 steps) | — | -1.79 ms | -1.2 % | no class (one load) |
|  | draftb | draft2: 155.45 (8 steps) | — | +0.27 ms | +0.2 % | no class (one load) |

### Prompt reading, tok/s

Per load: the tokens read at that depth over their time (a prime request reads a chunk on top of the cache; without primes each decode request reads its prompt), with the tokens read and the tokens reused. Requests that read under 32 tokens are left out.

| Depth (tokens) | Arm | Per load | Spread | Against `draft` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | draft | draft1: 32.5 (64 read on 0 cached) | — | control | | |
|  | kvonly | kvonly1: 33.7 (64 read on 0 cached) | — | +1.2 tok/s | +3.8 % | no class (one load) |
|  | draftb | draft2: 33.4 (64 read on 0 cached) | — | +0.9 tok/s | +2.9 % | no class (one load) |
| 32,768 | draft | draft1: 111.7 (32704 read on 0 cached) | — | control | | |
|  | kvonly | kvonly1: 120.2 (32704 read on 0 cached) | — | +8.5 tok/s | +7.6 % | no class (one load) |
|  | draftb | draft2: 111.5 (32704 read on 0 cached) | — | -0.2 tok/s | -0.2 % | no class (one load) |

### The draft and the step

Per load, pass 1: draft tokens accepted of drafted (the share), tokens a step (generated tokens over steps, a step being one pass of the main model: generated minus accepted), and the time of a step. Pass 2 is listed where its counters differ.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | draft | draft1: 7 of 23 (0.304), 1.78 a step, 265.5 ms a step |
|  | kvonly | kvonly1: 7 of 23 (0.304), 1.78 a step, 265.8 ms a step |
|  | draftb | draft2: 7 of 23 (0.304), 1.78 a step, 265.8 ms a step |
| 32,768 | draft | draft1: 8 of 18 (0.444), 2.00 a step, 309.0 ms a step (pass 2: 8 of 16 (0.500), 2.00 a step, 155.2 ms a step) |
|  | kvonly | kvonly1: 8 of 18 (0.444), 2.00 a step, 385.9 ms a step (pass 2: 8 of 16 (0.500), 2.00 a step, 153.4 ms a step) |
|  | draftb | draft2: 8 of 18 (0.444), 2.00 a step, 309.0 ms a step (pass 2: 8 of 16 (0.500), 2.00 a step, 155.5 ms a step) |

### Checks

- Times: every load started on a machine checked quiet (valid=1).
- Prompts: 64 ids at depth 0, 32768 ids at depth 32,768; the same ids on every load and pass.
- Tokens out: 16 on every timed request.
- Greedy reply at depth 0: the same on every pass and load of every arm.
- Greedy reply at depth 32,768: 2 different replies over the 3 loads (`draft`: 2 over its 1; `kvonly`: 2 over its 1; `draftb`: 2 over its 1); **draft1, kvonly1, draft2 wrote another reply on pass 2 than on pass 1.**
- Primes (read on cached, the cache's source): 32,768: 32704 on 0 (none).
