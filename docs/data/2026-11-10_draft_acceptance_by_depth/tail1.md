## The deep lane: draft, nodraft

Model file `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf`.

256 tokens per request at temperature 0, each request twice per load (pass 1 is the first request at a new depth, pass 2 the same request again). Timed loads, in their order: draft1, nodraft1.

| Arm | Loads | Draft | Context | Its own variables and flags |
|---|---|---|---:|---|
| draft (control) | draft1 | on | 65536 | — |
| nodraft | nodraft1 | off | 65536 | — |

**The times of this run are not measurements** (the quiet checks were off, valid=0), so no speed is printed. The counts below do not depend on the machine.

### The draft and the step

Per load, pass 1: draft tokens accepted of drafted (the share), tokens a step (generated tokens over steps, a step being one pass of the main model: generated minus accepted). Pass 2 is listed where its counters differ.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | draft | draft1: 171 of 252 (0.679), 3.01 a step |
|  | nodraft | nodraft1: no draft, 1.00 a step |
| 8,192 | draft | draft1: 186 of 205 (0.907), 3.66 a step (pass 2: 187 of 202 (0.926), 3.71 a step) |
|  | nodraft | nodraft1: no draft, 1.00 a step |
| 16,384 | draft | draft1: 186 of 207 (0.899), 3.66 a step |
|  | nodraft | nodraft1: no draft, 1.00 a step |
| 32,768 | draft | draft1: 167 of 262 (0.637), 2.88 a step (pass 2: 168 of 259 (0.649), 2.91 a step) |
|  | nodraft | nodraft1: no draft, 1.00 a step |
| 49,152 | draft | draft1: 177 of 234 (0.756), 3.24 a step (pass 2: 178 of 231 (0.771), 3.28 a step) |
|  | nodraft | nodraft1: no draft, 1.00 a step |

### The sampler pass

One request per depth and load with the server line's own sampler and a fixed seed: the draft counters and the step.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 0 | draft | draft1: 143 of 333 (0.429), 2.27 a step |
|  | nodraft | nodraft1: no draft, 1.00 a step |
| 8,192 | draft | draft1: 162 of 277 (0.585), 2.72 a step |
|  | nodraft | nodraft1: no draft, 1.00 a step |
| 16,384 | draft | draft1: 184 of 210 (0.876), 3.56 a step |
|  | nodraft | nodraft1: no draft, 1.00 a step |
| 32,768 | draft | draft1: 153 of 302 (0.507), 2.49 a step |
|  | nodraft | nodraft1: no draft, 1.00 a step |
| 49,152 | draft | draft1: 173 of 245 (0.706), 3.08 a step |
|  | nodraft | nodraft1: no draft, 1.00 a step |

### Checks

- Times: not measurements (valid=0).
- Prompts: 64 ids at depth 0, 8192 ids at depth 8,192, 16384 ids at depth 16,384, 32768 ids at depth 32,768, 49152 ids at depth 49,152; the same ids on every load and pass.
- Tokens out: 256 on every timed request.
- Greedy reply at depth 0: **differs from `draft` in nodraft.**
- Greedy reply at depth 8,192: **differs from `draft` in nodraft.**
- Greedy reply at depth 16,384: **differs from `draft` in nodraft.**
- Greedy reply at depth 32,768: **differs from `draft` in nodraft.**
- Greedy reply at depth 49,152: **differs from `draft` in nodraft.**
- Primes (read on cached, the cache's source): 8,192: 8128 on 0 (none); 16,384: 8708 on 7612 (checkpoint); 32,768: 16388 on 16316 (checkpoint); 49,152: 16900 on 32188 (checkpoint).
