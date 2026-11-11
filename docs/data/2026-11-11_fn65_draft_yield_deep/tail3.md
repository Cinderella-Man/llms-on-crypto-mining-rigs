## The deep lane: draft

Model file `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf`.

256 tokens per request at temperature 0, each request twice per load (pass 1 is the first request at a new depth, pass 2 the same request again). Timed loads, in their order: draft1.

| Arm | Loads | Draft | Context | Its own variables and flags |
|---|---|---|---:|---|
| draft (control) | draft1 | on | 131072 | — |

**The times of this run are not measurements** (the quiet checks were off, valid=0), so no speed is printed. The counts below do not depend on the machine.

### The draft and the step

Per load, pass 1: draft tokens accepted of drafted (the share), tokens a step (generated tokens over steps, a step being one pass of the main model: generated minus accepted). Pass 2 is listed where its counters differ.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 65,536 | draft | draft1: 189 of 197 (0.959), 3.82 a step |
| 120,000 | draft | draft1: 190 of 193 (0.984), 3.88 a step |

### The sampler pass

One request per depth and load with the server line's own sampler and a fixed seed: the draft counters and the step.

| Depth (tokens) | Arm | Per load |
|---:|---|---|
| 65,536 | draft | draft1: 162 of 276 (0.587), 2.72 a step |
| 120,000 | draft | draft1: 164 of 270 (0.607), 2.78 a step |

### Checks

- Times: not measurements (valid=0).
- Prompts: 65536 ids at depth 65,536, 120000 ids at depth 120,000; the same ids on every load and pass.
- Tokens out: 256 on every timed request.
- Greedy reply at depth 65,536: the same on every pass and load.
- Greedy reply at depth 120,000: the same on every pass and load.
- Primes (read on cached, the cache's source): 65,536: 65412 on 60 (checkpoint); 120,000: 54724 on 65212 (checkpoint).
