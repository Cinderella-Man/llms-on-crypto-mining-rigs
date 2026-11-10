## The deep lane: prof

Model file `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf`, llama-server version: 0.8.0-dev (build 12358, commit 1551f5fe1).

256 tokens per request at temperature 0, each request twice per load (pass 1 is the first request at a new depth, pass 2 the same request again). Timed loads, in their order: prof.

| Arm | Loads | Draft | Context | Its own variables and flags |
|---|---|---|---:|---|
| prof (control) | prof | on | 65536 | `GGML_CUDA_PROFILE=$OUT/prof-depth.csv GGML_CUDA_PROFILE_MODE=group` |

### Generation, tok/s

Per load: pass 1, pass 2 → the load, with the load's time per token. A load's speed is its two passes' tokens over their time. Spread: the difference between the arm's fastest and slowest load. Class against `prof`: gain when the mean is above the control's by more than the larger of the two spreads, loss when below by more than it, else negligible.

| Depth (tokens) | Arm | Per load | Spread | Against `prof` | | Class |
|---:|---|---|---:|---:|---:|---|
| 16,384 | prof | prof: 17.47, 18.93 → **18.17** (55.03 ms a token) | — | control | | |
| 32,768 | prof | prof: 14.57, 15.04 → **14.80** (67.57 ms a token) | — | control | | |
| 49,152 | prof | prof: 12.48, 14.43 → **13.38** (74.71 ms a token) | — | control | | |

### Generation over all depths, tok/s

Every depth's tokens over their time, per load.

| Depth (tokens) | Arm | Per load | Spread | Against `prof` | | Class |
|---:|---|---|---:|---:|---:|---|
| all | prof | prof: **15.20** | — | control | | |

### The step on pass 1 (the first request at a new depth), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `prof`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `prof` | | Class |
|---:|---|---|---:|---:|---:|---|
| 16,384 | prof | prof: 206.35 (71 steps) | — | control | | |
| 32,768 | prof | prof: 216.91 (81 steps) | — | control | | |
| 49,152 | prof | prof: 262.99 (78 steps) | — | control | | |

### The step on pass 2 (the same request again), ms

A step is one pass of the main model: it yields one token of its own and the draft tokens it accepts, so a request's steps are its generated tokens minus the accepted ones, and the time of a step is the request's generation time over its steps. Loads that write different replies accept different shares of the draft, so their tok/s differ by more than their steps do. Per load: the request of this pass, with its steps. Class against `prof`: gain when the mean is below the control's by more than the larger of the two spreads, loss when above by more than it, else negligible; none for an arm whose draft is not in the control's state, since its steps yield other tokens.

| Depth (tokens) | Arm | Per load | Spread | Against `prof` | | Class |
|---:|---|---|---:|---:|---:|---|
| 16,384 | prof | prof: 190.52 (71 steps) | — | control | | |
| 32,768 | prof | prof: 210.18 (81 steps) | — | control | | |
| 49,152 | prof | prof: 230.37 (77 steps) | — | control | | |

### Prompt reading, tok/s

Per load: the tokens read at that depth over their time (a prime request reads a chunk on top of the cache; without primes each decode request reads its prompt), with the tokens read and the tokens reused. Requests that read under 32 tokens are left out.

| Depth (tokens) | Arm | Per load | Spread | Against `prof` | | Class |
|---:|---|---|---:|---:|---:|---|
| 16,384 | prof | prof: 62.0 (16260 read on 60 cached) | — | control | | |
| 32,768 | prof | prof: 46.4 (16900 read on 15804 cached) | — | control | | |
| 49,152 | prof | prof: 36.0 (16900 read on 32188 cached) | — | control | | |

### The draft and the step

Per load, pass 1: draft tokens accepted of drafted (the share), tokens a step (generated tokens over steps, a step being one pass of the main model: generated minus accepted), and the time of a step. Pass 2 is listed where its counters differ.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 16,384 | prof | prof: 185 of 210 (0.881), 3.61 a step, 206.4 ms a step |
| 32,768 | prof | prof: 175 of 240 (0.729), 3.16 a step, 216.9 ms a step |
| 49,152 | prof | prof: 178 of 226 (0.788), 3.28 a step, 263.0 ms a step (pass 2: 179 of 223 (0.803), 3.32 a step, 230.4 ms a step) |

### Checks

- Times: every load started on a machine checked quiet (valid=1).
- Prompts: 16384 ids at depth 16,384, 32768 ids at depth 32,768, 49152 ids at depth 49,152; the same ids on every load and pass.
- Tokens out: 256 on every timed request.
- Greedy reply at depth 16,384: the same on every pass and load.
- Greedy reply at depth 32,768: the same on every pass and load.
- Greedy reply at depth 49,152: the same on every pass and load.
- Primes (read on cached, the cache's source): 16,384: 16260 on 60 (checkpoint); 32,768: 16900 on 15804 (checkpoint); 49,152: 16900 on 32188 (checkpoint).
