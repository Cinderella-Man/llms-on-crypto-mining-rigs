## The deep lane: draft, nodraft

Model file `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf`, llama-server version: 0.8.0-dev (build 12358, commit 1551f5fe1).

256 tokens per request at temperature 0, each request twice per load (pass 1 is the first request at a new depth, pass 2 the same request again). Timed loads, in their order: draft1, nodraft1.

| Arm | Loads | Draft | Context | Its own variables and flags |
|---|---|---|---:|---|
| draft (control) | draft1 | on | 196608 | — |
| nodraft | nodraft1 | off | 196608 | — |

### Generation, tok/s

Per load: pass 1, pass 2 → the load, with the load's time per token. A load's speed is its two passes' tokens over their time. Spread: the difference between the arm's fastest and slowest load. Class against `draft`: gain when the mean is above the control's by more than the larger of the two spreads, loss when below by more than it, else negligible.

| Depth (tokens) | Arm | Per load | Spread | Against `draft` | | Class |
|---:|---|---|---:|---:|---:|---|
| 120,000 | draft | draft1: 12.08, 13.18 → **12.61** (79.31 ms a token) | — | control | | |
|  | nodraft | nodraft1: 9.65, 11.06 → **10.31** (97.03 ms a token) | — | -2.30 tok/s | -18.3 % | no class (one load) |
| 196,000 | draft | draft1: 10.50, 10.62 → **10.56** (94.71 ms a token) | — | control | | |
|  | nodraft | nodraft1: 8.75, 8.86 → **8.80** (113.58 ms a token) | — | -1.75 tok/s | -16.6 % | no class (one load) |

### Generation over all depths, tok/s

Every depth's tokens over their time, per load.

| Depth (tokens) | Arm | Per load | Spread | Against `draft` | | Class |
|---:|---|---|---:|---:|---:|---|
| all | draft | draft1: **11.49** | — | control | | |
|  | nodraft | nodraft1: **9.50** | — | -2.00 tok/s | -17.4 % | no class (one load) |

### The step on pass 1 (the first request at a new depth), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `draft`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `draft` | | Class |
|---:|---|---|---:|---:|---:|---|
| 120,000 | draft | draft1: 268.23 (79 steps) | — | control | | |
|  | nodraft | nodraft1: 103.65 (256 steps) | — | -164.58 ms | -61.4 % | no class (the draft differs) |
| 196,000 | draft | draft1: 325.15 (75 steps) | — | control | | |
|  | nodraft | nodraft1: 114.24 (256 steps) | — | -210.91 ms | -64.9 % | no class (the draft differs) |

### The step on pass 2 (the same request again), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `draft`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `draft` | | Class |
|---:|---|---|---:|---:|---:|---|
| 120,000 | draft | draft1: 248.95 (78 steps) | — | control | | |
|  | nodraft | nodraft1: 90.42 (256 steps) | — | -158.53 ms | -63.7 % | no class (the draft differs) |
| 196,000 | draft | draft1: 321.43 (75 steps) | — | control | | |
|  | nodraft | nodraft1: 112.92 (256 steps) | — | -208.52 ms | -64.9 % | no class (the draft differs) |

### Prompt reading, tok/s

Per load: the tokens read at that depth over their time (a prime request reads a chunk on top of the cache; without primes each decode request reads its prompt), with the tokens read and the tokens reused. Requests that read under 32 tokens are left out.

| Depth (tokens) | Arm | Per load | Spread | Against `draft` | | Class |
|---:|---|---|---:|---:|---:|---|
| 120,000 | draft | draft1: 65.4 (119876 read on 60 cached) | — | control | | |
|  | nodraft | nodraft1: 97.9 (119876 read on 60 cached) | — | +32.5 tok/s | +49.6 % | no class (one load) |
| 196,000 | draft | draft1: 32.9 (76260 read on 119676 cached) | — | control | | |
|  | nodraft | nodraft1: 50.3 (76260 read on 119676 cached) | — | +17.4 tok/s | +53.0 % | no class (one load) |

### The draft and the step

Per load, pass 1: draft tokens accepted of drafted (the share), tokens a step (generated tokens over steps, a step being one pass of the main model: generated minus accepted), and the time of a step. Pass 2 is listed where its counters differ.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 120,000 | draft | draft1: 177 of 231 (0.766), 3.24 a step, 268.2 ms a step (pass 2: 178 of 228 (0.781), 3.28 a step, 248.9 ms a step) |
|  | nodraft | nodraft1: no draft, 1.00 a step, 103.6 ms a step |
| 196,000 | draft | draft1: 181 of 222 (0.815), 3.41 a step, 325.2 ms a step |
|  | nodraft | nodraft1: no draft, 1.00 a step, 114.2 ms a step |

### Checks

- Times: every load started on a machine checked quiet (valid=1).
- Prompts: 120000 ids at depth 120,000, 196000 ids at depth 196,000; the same ids on every load and pass.
- Tokens out: 256 on every timed request.
- Greedy reply at depth 120,000: **differs from `draft` in nodraft.**
- Greedy reply at depth 196,000: **differs from `draft` in nodraft.**
- Primes (read on cached, the cache's source): 120,000: 119876 on 60 (checkpoint); 196,000: 76260 on 119676 (checkpoint).
