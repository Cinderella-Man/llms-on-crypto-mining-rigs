## The deep lane: draft, overlap

Model file `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf`, llama-server version: 0.9.0-dev (build 12661, commit d097d6d88a).

16 tokens per request at temperature 0, each request twice per load (pass 1 is the first request at a new depth, pass 2 the same request again). Timed loads, in their order: draft1, overlap1, draft2.

| Arm | Loads | Draft | Context | Its own variables and flags |
|---|---|---|---:|---|
| draft (control) | draft1, draft2 | on | 65536 | `LLAMA_SPEC_DUMP=$WORK/dump-draft.jsonl` |
| overlap | overlap1 | on | 65536 | `LLAMA_MTP_CATCHUP_OVERLAP=1 LLAMA_SPEC_DUMP=$WORK/dump-overlap.jsonl` |

### Generation, tok/s

Per load: pass 1, pass 2 → the load, with the load's time per token. A load's speed is its two passes' tokens over their time. Spread: the difference between the arm's fastest and slowest load. Class against `draft`: gain when the mean is above the control's by more than the larger of the two spreads, loss when below by more than it, else negligible.

| Depth (tokens) | Arm | Per load | Spread | Against `draft` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | draft | draft1: 6.66, 13.80 → **8.98** (111.33 ms a token); draft2: 6.65, 13.78 → **8.97** (111.43 ms a token) | 0.01 | control | | |
|  | overlap | overlap1: 6.68, 13.79 → **9.00** (111.11 ms a token) | — | +0.02 tok/s | +0.2 % | no class (one load) |
| 32,768 | draft | draft1: 4.60, 11.30 → **6.53** (153.03 ms a token); draft2: 6.45, 12.78 → **8.57** (116.66 ms a token) | 2.04 | control | | |
|  | overlap | overlap1: 4.95, 12.76 → **7.13** (140.28 ms a token) | — | -0.42 tok/s | -5.6 % | no class (one load) |

### Generation over all depths, tok/s

Every depth's tokens over their time, per load.

| Depth (tokens) | Arm | Per load | Spread | Against `draft` | | Class |
|---:|---|---|---:|---:|---:|---|
| all | draft | draft1: **7.57**; draft2: **8.77** | 1.20 | control | | |
|  | overlap | overlap1: **7.96** | — | -0.21 tok/s | -2.6 % | no class (one load) |

### The step on pass 1 (the first request at a new depth), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `draft`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `draft` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | draft | draft1: 267.01 (9 steps); draft2: 267.20 (9 steps) | 0.19 | control | | |
|  | overlap | overlap1: 266.13 (9 steps) | — | -0.98 ms | -0.4 % | no class (one load) |
| 32,768 | draft | draft1: 435.22 (8 steps); draft2: 310.08 (8 steps) | 125.14 | control | | |
|  | overlap | overlap1: 404.32 (8 steps) | — | +31.67 ms | +8.5 % | no class (one load) |

### The step on pass 2 (the same request again), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `draft`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `draft` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | draft | draft1: 128.82 (9 steps); draft2: 128.98 (9 steps) | 0.16 | control | | |
|  | overlap | overlap1: 128.94 (9 steps) | — | +0.04 ms | +0.0 % | no class (one load) |
| 32,768 | draft | draft1: 176.92 (8 steps); draft2: 156.55 (8 steps) | 20.37 | control | | |
|  | overlap | overlap1: 156.79 (8 steps) | — | -9.94 ms | -6.0 % | no class (one load) |

### Prompt reading, tok/s

Per load: the tokens read at that depth over their time (a prime request reads a chunk on top of the cache; without primes each decode request reads its prompt), with the tokens read and the tokens reused. Requests that read under 32 tokens are left out.

| Depth (tokens) | Arm | Per load | Spread | Against `draft` | | Class |
|---:|---|---|---:|---:|---:|---|
| 0 | draft | draft1: 32.9 (64 read on 0 cached); draft2: 33.5 (64 read on 0 cached) | 0.5 | control | | |
|  | overlap | overlap1: 32.9 (64 read on 0 cached) | — | -0.3 tok/s | -1.0 % | no class (one load) |
| 32,768 | draft | draft1: 111.4 (32704 read on 0 cached); draft2: 111.5 (32704 read on 0 cached) | 0.1 | control | | |
|  | overlap | overlap1: 157.3 (32704 read on 0 cached) | — | +45.9 tok/s | +41.1 % | no class (one load) |

### The draft and the step

Per load, pass 1: draft tokens accepted of drafted (the share), tokens a step (generated tokens over steps, a step being one pass of the main model: generated minus accepted), and the time of a step. Pass 2 is listed where its counters differ.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | draft | draft1: 7 of 23 (0.304), 1.78 a step, 267.0 ms a step; draft2: 7 of 23 (0.304), 1.78 a step, 267.2 ms a step |
|  | overlap | overlap1: 7 of 23 (0.304), 1.78 a step, 266.1 ms a step |
| 32,768 | draft | draft1: 8 of 18 (0.444), 2.00 a step, 435.2 ms a step (pass 2: 8 of 16 (0.500), 2.00 a step, 176.9 ms a step); draft2: 8 of 18 (0.444), 2.00 a step, 310.1 ms a step (pass 2: 8 of 16 (0.500), 2.00 a step, 156.6 ms a step) |
|  | overlap | overlap1: 8 of 18 (0.444), 2.00 a step, 404.3 ms a step (pass 2: 8 of 16 (0.500), 2.00 a step, 156.8 ms a step) |

### Checks

- Times: every load started on a machine checked quiet (valid=1).
- Prompts: 64 ids at depth 0, 32768 ids at depth 32,768; the same ids on every load and pass.
- Tokens out: 16 on every timed request.
- Greedy reply at depth 0: the same on every pass and load of every arm.
- Greedy reply at depth 32,768: 2 different replies over the 3 loads (`draft`: 2 over its 2; `overlap`: 2 over its 1); **draft1, overlap1, draft2 wrote another reply on pass 2 than on pass 1.**
- Primes (read on cached, the cache's source): 32,768: 32704 on 0 (none).
