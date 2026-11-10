## The draft against no draft: `draft` (draft1) and `nodraft` (nodraft1)

### Generation on both passes: tok/s (ms a token)

Per load: pass 1, pass 2 → the load, its two requests' tokens over their time. Spread: the difference between an arm's loads. Class of `draft` against `nodraft`: gain when its mean is above `nodraft`'s by more than the larger spread, loss when below by more than it, else negligible.

| Depth (tokens) | `draft1` | Spread | `nodraft1` | Spread | `draft` against `nodraft` | | Class |
|---:|---|---:|---|---:|---:|---:|---|
| 49,152 | 15.01, 18.05 → **16.39** (61.01) | — | 14.12, 14.55 → **14.33** (69.78) | — | +2.06 tok/s | +14.4 % | no class (one load) |
| 65,536 | 17.09, 18.82 → **17.92** (55.82) | — | 13.39, 13.55 → **13.47** (74.25) | — | +4.45 tok/s | +33.0 % | no class (one load) |
| 120,000 | 15.20, 15.29 → **15.24** (65.60) | — | 10.85, 11.07 → **10.96** (91.25) | — | +4.28 tok/s | +39.1 % | no class (one load) |
| all depths | **16.44** (60.81) | — | **12.75** (78.42) | — | +3.69 tok/s | +29.0 % | no class (one load) |

### Generation on pass 1: tok/s (ms a token)

Per load. Spread: the difference between an arm's loads. Class of `draft` against `nodraft`: gain when its mean is above `nodraft`'s by more than the larger spread, loss when below by more than it, else negligible.

| Depth (tokens) | `draft1` | Spread | `nodraft1` | Spread | `draft` against `nodraft` | | Class |
|---:|---|---:|---|---:|---:|---:|---|
| 49,152 | **15.01** (66.63) | — | **14.12** (70.84) | — | +0.89 tok/s | +6.3 % | no class (one load) |
| 65,536 | **17.09** (58.51) | — | **13.39** (74.68) | — | +3.70 tok/s | +27.6 % | no class (one load) |
| 120,000 | **15.20** (65.78) | — | **10.85** (92.15) | — | +4.35 tok/s | +40.1 % | no class (one load) |

### Generation on pass 2: tok/s (ms a token)

Per load. Spread: the difference between an arm's loads. Class of `draft` against `nodraft`: gain when its mean is above `nodraft`'s by more than the larger spread, loss when below by more than it, else negligible.

| Depth (tokens) | `draft1` | Spread | `nodraft1` | Spread | `draft` against `nodraft` | | Class |
|---:|---|---:|---|---:|---:|---:|---|
| 49,152 | **18.05** (55.40) | — | **14.55** (68.72) | — | +3.50 tok/s | +24.1 % | no class (one load) |
| 65,536 | **18.82** (53.13) | — | **13.55** (73.81) | — | +5.27 tok/s | +38.9 % | no class (one load) |
| 120,000 | **15.29** (65.41) | — | **11.07** (90.34) | — | +4.22 tok/s | +38.1 % | no class (one load) |

### The step: what it costs and what it must yield

A step of `draft` is one pass of the main model over the last token and the drafted ones, plus the draft's own passes; it yields one token and the draft tokens it accepts. A step of `nodraft` is one token. The draft pays when a step yields more tokens than its time is in tokens of the plain line: that ratio is the break-even, from the means of each arm's loads. The share to break even is the break-even minus one, over the draft tokens the arm offered a step.

| Depth (tokens) | Pass | Step of `draft`, ms: `draft1` | Token of `nodraft`, ms: `nodraft1` | Tokens a step to break even | Share accepted to break even | Tokens a step yielded: `draft1` | Share accepted: `draft1` |
|---:|---|---|---|---:|---:|---|---|
| 49,152 | 1 | 218.69 | 70.84 | 3.09 | 0.705 | 3.28 | 0.771 |
|  | 2 | 181.81 | 68.72 | 2.65 | 0.556 | 3.28 | 0.771 |
|  | both | 200.25 | 69.78 | 2.87 | 0.631 | 3.28 | 0.771 |
| 65,536 | 1 | 217.07 | 74.68 | 2.91 | 0.655 | 3.71 | 0.930 |
|  | 2 | 197.12 | 73.81 | 2.67 | 0.573 | 3.71 | 0.930 |
|  | both | 207.09 | 74.25 | 2.79 | 0.614 | 3.71 | 0.930 |
| 120,000 | 1 | 251.36 | 92.15 | 2.73 | 0.591 | 3.82 | 0.964 |
|  | 2 | 249.94 | 90.34 | 2.77 | 0.604 | 3.82 | 0.964 |
|  | both | 250.65 | 91.25 | 2.75 | 0.597 | 3.82 | 0.964 |

### The draft against no draft at another yield (est.)

Tokens a step measured elsewhere, at this run's step times: the yield over the break-even, minus one. An estimate, not a measured speed.

| Depth (tokens) | Yield | Tokens a step | Break-even on pass 1 | Est. on pass 1 | Break-even on pass 2 | Est. on pass 2 |
|---:|---|---:|---:|---:|---:|---:|
| 49,152 | the sampler off the rig, at 49,152 | 2.88 | 3.09 | -6.8 % | 2.65 | +8.7 % |
| 65,536 | the sampler off the rig, at 49,152 | 2.88 | 2.91 | -1.1 % | 2.67 | +7.7 % |
| 120,000 | the sampler off the rig, at 49,152 | 2.88 | 2.73 | +5.4 % | 2.77 | +4.0 % |

### Prompt reading: tok/s (ms a prompt token)

Per load: the tokens the prime request read at that depth over their time (at a depth without a prime, the first decode request). The draft adds: the difference between the arms' mean ms a prompt token, and its share of `draft`'s.

| Depth (tokens) | Read, on cached | `draft1` | `nodraft1` | `draft` against `nodraft` | Class | The draft adds, ms a prompt token | Share of the prompt time |
|---:|---|---|---|---:|---|---:|---:|
| 49,152 | 49028 on 60 | **100.5** (9.95) | **146.2** (6.84) | -31.3 % | no class (one load) | +3.11 | 31.3 % |
| 65,536 | 16644 on 48828 | **66.2** (15.10) | **93.7** (10.68) | -29.3 % | no class (one load) | +4.42 | 29.3 % |
| 120,000 | 54724 on 65212 | **49.0** (20.41) | **73.6** (13.58) | -33.5 % | no class (one load) | +6.83 | 33.5 % |

### A whole turn: the prompt it reads against the tokens it writes

A turn reads new prompt tokens and then generates. The draft saves time per generated token (the plain line's ms a token minus the draft line's, on pass 1, the first request at a depth) and costs time per prompt token (the table above). The last column is the saving over the cost: the new prompt tokens per generated token up to which the draft pays in a turn at that depth. For this run's loads the saving is between the arms' mean ms a token; for a yield measured elsewhere it is the plain token minus the step over that yield (est.).

| Depth (tokens) | The draft adds, ms a prompt token | Yield | Tokens a step | Saved, ms a generated token | Prompt tokens per generated token to break even |
|---:|---:|---|---:|---:|---:|
| 49,152 | +3.11 | this run's loads | 3.28 | +4.21 | 1.35 |
|  |  | the sampler off the rig, at 49,152 (est.) | 2.88 | -5.19 | none: nothing saved |
| 65,536 | +4.42 | this run's loads | 3.71 | +16.17 | 3.66 |
|  |  | the sampler off the rig, at 49,152 (est.) | 2.88 | -0.80 | none: nothing saved |
| 120,000 | +6.83 | this run's loads | 3.82 | +26.37 | 3.86 |
|  |  | the sampler off the rig, at 49,152 (est.) | 2.88 | +4.76 | 0.70 |

