## The deep lane: v4s

Model file `flint-near-q8-p065.gguf`, llama-server version: 1.0.0-dev (build 12832, commit eb312b7bce).

256 tokens per request at temperature 0, each request twice per load (pass 1 is the first request at a new depth, pass 2 the same request again). Timed loads, in their order: fit.

| Arm | Loads | Draft | Context | Its own variables and flags |
|---|---|---|---:|---|
| v4s (control) | fit | off | 262144 | `LLAMA_KV_SEQ_RM_BOUND=1 LLAMA_SERVER_YIELD=2 LLAMA_SAMPLING_TOP_K_ROWS=1 GGML_CUDA_KEEPALIVE=decode GGML_CUDA_KEEPALIVE_EAGER=1 GGML_CUDA_HOP_STAGED=1 GGML_CUDA_GRAPHS_FORCE=1`, `--ctx-checkpoints-vram 4 --ctx-checkpoints-incall --ubatch-taper short=600:64` |

### Generation, tok/s

Per load: pass 1, pass 2 → the load, with the load's time per token. A load's speed is its two passes' tokens over their time. Spread: the difference between the arm's fastest and slowest load. Class against `v4s`: gain when the mean is above the control's by more than the larger of the two spreads, loss when below by more than it, else negligible.

| Depth (tokens) | Arm | Per load | Spread | Against `v4s` | | Class |
|---:|---|---|---:|---:|---:|---|
| 49,152 | v4s | fit: 22.32, 23.24 → **22.77** (43.92 ms a token) | — | control | | |

### Generation over all depths, tok/s

Every depth's tokens over their time, per load.

| Depth (tokens) | Arm | Per load | Spread | Against `v4s` | | Class |
|---:|---|---|---:|---:|---:|---|
| all | v4s | fit: **22.77** | — | control | | |

### Prompt reading, tok/s

Per load: the tokens read at that depth over their time (a prime request reads a chunk on top of the cache; without primes each decode request reads its prompt), with the tokens read and the tokens reused. Requests that read under 32 tokens are left out.

| Depth (tokens) | Arm | Per load | Spread | Against `v4s` | | Class |
|---:|---|---|---:|---:|---:|---|
| 49,152 | v4s | fit: 271.8 (49028 read on 60 cached) | — | control | | |

### The draft and the step

Per load, pass 1: draft tokens accepted of drafted (the share), tokens a step (generated tokens over steps, a step being one pass of the main model: generated minus accepted), and the time of a step. Pass 2 is listed where its counters differ.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 49,152 | v4s | fit: no draft, 1.00 a step, 44.8 ms a step |

### The sampler pass

One request per depth and load with the server line's own sampler and a fixed seed: tok/s, the draft counters and the step.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 49,152 | v4s | fit: 29.28 tok/s, no draft, 1.00 a step, 34.2 ms a step |

### Checks

- Times: every load started on a machine checked quiet (valid=1).
- Prompts: 49152 ids at depth 49,152; the same ids on every load and pass.
- Tokens out: 256 on every timed request.
- Greedy reply at depth 49,152: the same on every pass and load.
- Primes (read on cached, the cache's source): 49,152: 49028 on 60 (checkpoint).
- Graph reallocations of the target, requests with one per load: fit: 0 of 5 requests.
