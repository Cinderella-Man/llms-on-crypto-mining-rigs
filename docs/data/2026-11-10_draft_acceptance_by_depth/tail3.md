## The deep lane: draft

Model file `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf`.

256 tokens per request at temperature 0, each request twice per load (pass 1 is the first request at a new depth, pass 2 the same request again). Timed loads, in their order: draft1.

| Arm | Loads | Draft | Context | Its own variables and flags |
|---|---|---|---:|---|
| draft (control) | draft1 | on | 65536 | — |

**The times of this run are not measurements** (the quiet checks were off, valid=0), so no speed is printed. The counts below do not depend on the machine.

### The draft and the step

Per load, pass 1: draft tokens accepted of drafted (the share), tokens a step (generated tokens over steps, a step being one pass of the main model: generated minus accepted). Pass 2 is listed where its counters differ.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | draft | draft1: 169 of 256 (0.660), 2.94 a step (pass 2: 171 of 250 (0.684), 3.01 a step) |
| 8,192 | draft | draft1: 170 of 255 (0.667), 2.98 a step (pass 2: 171 of 252 (0.679), 3.01 a step) |
| 16,384 | draft | draft1: 181 of 219 (0.826), 3.41 a step (pass 2: 182 of 216 (0.843), 3.46 a step) |
| 32,768 | draft | draft1: 184 of 208 (0.885), 3.56 a step |
| 49,152 | draft | draft1: 180 of 223 (0.807), 3.37 a step (pass 2: 181 of 220 (0.823), 3.41 a step) |

### The sampler pass

One request per depth and load with the server line's own sampler and a fixed seed: the draft counters and the step.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | draft | draft1: 162 of 277 (0.585), 2.72 a step |
| 8,192 | draft | draft1: 163 of 273 (0.597), 2.75 a step |
| 16,384 | draft | draft1: 167 of 263 (0.635), 2.88 a step |
| 32,768 | draft | draft1: 180 of 225 (0.800), 3.37 a step |
| 49,152 | draft | draft1: 175 of 238 (0.735), 3.16 a step |

### Checks

- Times: not measurements (valid=0).
- Prompts: 64 ids at depth 0, 8192 ids at depth 8,192, 16384 ids at depth 16,384, 32768 ids at depth 32,768, 49152 ids at depth 49,152; the same ids on every load and pass.
- Tokens out: 256 on every timed request.
- Greedy reply at depth 0: the same on every pass and load.
- Greedy reply at depth 8,192: the same on every pass and load.
- Greedy reply at depth 16,384: the same on every pass and load.
- Greedy reply at depth 32,768: the same on every pass and load.
- Greedy reply at depth 49,152: the same on every pass and load.
- Primes (read on cached, the cache's source): 8,192: 8128 on 0 (none); 16,384: 8708 on 7612 (checkpoint); 32,768: 16388 on 16316 (checkpoint); 49,152: 16900 on 32188 (checkpoint).
