## The deep lane: host

Model file `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf`, llama-server version: 0.8.0-dev (build 12358, commit 1551f5fe1).

256 tokens per request at temperature 0, each request twice per load (pass 1 is the first request at a new depth, pass 2 the same request again). Timed loads, in their order: host.

| Arm | Loads | Draft | Context | Its own variables and flags |
|---|---|---|---:|---|
| host (control) | host | on | 65536 | `LLAMA_QSA_INPUT_STATS=1 GGML_CUDA_PROFILE_HOST=$OUT/host.csv` |

### Generation, tok/s

Per load: pass 1, pass 2 → the load, with the load's time per token. A load's speed is its two passes' tokens over their time. Spread: the difference between the arm's fastest and slowest load. Class against `host`: gain when the mean is above the control's by more than the larger of the two spreads, loss when below by more than it, else negligible.

| Depth (tokens) | Arm | Per load | Spread | Against `host` | | Class |
|---:|---|---|---:|---:|---:|---|
| 16,384 | host | host: 17.27, 24.15 → **20.14** (49.66 ms a token) | — | control | | |
| 49,152 | host | host: 18.63, 19.06 → **18.84** (53.07 ms a token) | — | control | | |

### Generation over all depths, tok/s

Every depth's tokens over their time, per load.

| Depth (tokens) | Arm | Per load | Spread | Against `host` | | Class |
|---:|---|---|---:|---:|---:|---|
| all | host | host: **19.47** | — | control | | |

### The step on pass 1 (the first request at a new depth), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `host`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `host` | | Class |
|---:|---|---|---:|---:|---:|---|
| 16,384 | host | host: 211.80 (70 steps) | — | control | | |
| 49,152 | host | host: 185.71 (74 steps) | — | control | | |

### The step on pass 2 (the same request again), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `host`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `host` | | Class |
|---:|---|---|---:|---:|---:|---|
| 16,384 | host | host: 151.44 (70 steps) | — | control | | |
| 49,152 | host | host: 181.48 (74 steps) | — | control | | |

### Prompt reading, tok/s

Per load: the tokens read at that depth over their time (a prime request reads a chunk on top of the cache; without primes each decode request reads its prompt), with the tokens read and the tokens reused. Requests that read under 32 tokens are left out.

| Depth (tokens) | Arm | Per load | Spread | Against `host` | | Class |
|---:|---|---|---:|---:|---:|---|
| 16,384 | host | host: 107.5 (16260 read on 60 cached) | — | control | | |
| 49,152 | host | host: 71.7 (33284 read on 15804 cached) | — | control | | |

### The draft and the step

Per load, pass 1: draft tokens accepted of drafted (the share), tokens a step (generated tokens over steps, a step being one pass of the main model: generated minus accepted), and the time of a step. Pass 2 is listed where its counters differ.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 16,384 | host | host: 186 of 206 (0.903), 3.66 a step, 211.8 ms a step |
| 49,152 | host | host: 182 of 218 (0.835), 3.46 a step, 185.7 ms a step |

### Checks

- Times: every load started on a machine checked quiet (valid=1).
- Prompts: 16384 ids at depth 16,384, 49152 ids at depth 49,152; the same ids on every load and pass.
- Tokens out: 256 on every timed request.
- Greedy reply at depth 16,384: the same on every pass and load.
- Greedy reply at depth 49,152: the same on every pass and load.
- Primes (read on cached, the cache's source): 16,384: 16260 on 60 (checkpoint); 49,152: 33284 on 15804 (checkpoint).
