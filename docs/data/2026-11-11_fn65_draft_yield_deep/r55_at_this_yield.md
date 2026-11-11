## The draft against no draft: `draft` (draft1, draft2) and `nodraft` (nodraft1, nodraft2)

### Generation on both passes: tok/s (ms a token)

Per load: pass 1, pass 2 → the load, its two requests' tokens over their time. Spread: the difference between an arm's loads. Class of `draft` against `nodraft`: gain when its mean is above `nodraft`'s by more than the larger spread, loss when below by more than it, else negligible.

| Depth (tokens) | `draft1` | `draft2` | Spread | `nodraft1` | `nodraft2` | Spread | `draft` against `nodraft` | | Class |
|---:|---|---|---:|---|---|---:|---:|---:|---|
| 49,152 | 15.01, 18.05 → **16.39** (61.01) | 19.03, 19.44 → **19.23** (51.99) | 2.85 | 14.12, 14.55 → **14.33** (69.78) | 14.18, 14.51 → **14.34** (69.72) | 0.01 | +3.48 tok/s | +24.2 % | gain |
| 65,536 | 17.09, 18.82 → **17.92** (55.82) | 16.91, 18.50 → **17.67** (56.60) | 0.25 | 13.39, 13.55 → **13.47** (74.25) | 13.29, 13.51 → **13.40** (74.65) | 0.07 | +4.36 tok/s | +32.4 % | gain |
| 120,000 | 15.20, 15.29 → **15.24** (65.60) | 12.62, 12.80 → **12.71** (78.69) | 2.54 | 10.85, 11.07 → **10.96** (91.25) | 10.78, 11.03 → **10.90** (91.74) | 0.06 | +3.05 tok/s | +27.9 % | gain |
| all depths | **16.44** (60.81) | **16.02** (62.43) | 0.43 | **12.75** (78.42) | **12.71** (78.71) | 0.05 | +3.50 tok/s | +27.5 % | gain |

### Generation on pass 1: tok/s (ms a token)

Per load. Spread: the difference between an arm's loads. Class of `draft` against `nodraft`: gain when its mean is above `nodraft`'s by more than the larger spread, loss when below by more than it, else negligible.

| Depth (tokens) | `draft1` | `draft2` | Spread | `nodraft1` | `nodraft2` | Spread | `draft` against `nodraft` | | Class |
|---:|---|---|---:|---|---|---:|---:|---:|---|
| 49,152 | **15.01** (66.63) | **19.03** (52.54) | 4.02 | **14.12** (70.84) | **14.18** (70.54) | 0.06 | +2.87 tok/s | +20.3 % | negligible |
| 65,536 | **17.09** (58.51) | **16.91** (59.14) | 0.18 | **13.39** (74.68) | **13.29** (75.25) | 0.10 | +3.66 tok/s | +27.4 % | gain |
| 120,000 | **15.20** (65.78) | **12.62** (79.26) | 2.58 | **10.85** (92.15) | **10.78** (92.79) | 0.07 | +3.09 tok/s | +28.6 % | gain |

### Generation on pass 2: tok/s (ms a token)

Per load. Spread: the difference between an arm's loads. Class of `draft` against `nodraft`: gain when its mean is above `nodraft`'s by more than the larger spread, loss when below by more than it, else negligible.

| Depth (tokens) | `draft1` | `draft2` | Spread | `nodraft1` | `nodraft2` | Spread | `draft` against `nodraft` | | Class |
|---:|---|---|---:|---|---|---:|---:|---:|---|
| 49,152 | **18.05** (55.40) | **19.44** (51.43) | 1.39 | **14.55** (68.72) | **14.51** (68.91) | 0.04 | +4.22 tok/s | +29.0 % | gain |
| 65,536 | **18.82** (53.13) | **18.50** (54.07) | 0.33 | **13.55** (73.81) | **13.51** (74.05) | 0.04 | +5.13 tok/s | +37.9 % | gain |
| 120,000 | **15.29** (65.41) | **12.80** (78.11) | 2.49 | **11.07** (90.34) | **11.03** (90.70) | 0.04 | +3.00 tok/s | +27.1 % | gain |

### The step: what it costs and what it must yield

A step of `draft` is one pass of the main model over the last token and the drafted ones, plus the draft's own passes; it yields one token and the draft tokens it accepts. A step of `nodraft` is one token. The draft pays when a step yields more tokens than its time is in tokens of the plain line: that ratio is the break-even, from the means of each arm's loads. The share to break even is the break-even minus one, over the draft tokens the arm offered a step.

| Depth (tokens) | Pass | Step of `draft`, ms: `draft1`, `draft2` | Token of `nodraft`, ms: `nodraft1`, `nodraft2` | Tokens a step to break even | Share accepted to break even | Tokens a step yielded: `draft1`, `draft2` | Share accepted: `draft1`, `draft2` |
|---:|---|---|---|---:|---:|---|---|
| 49,152 | 1 | 218.69, 186.82 | 70.84, 70.54 | 2.87 | 0.634 | 3.28, 3.56 | 0.771, 0.872 |
|  | 2 | 181.81, 182.88 | 68.72, 68.91 | 2.65 | 0.560 | 3.28, 3.56 | 0.771, 0.872 |
|  | both | 200.25, 184.85 | 69.78, 69.72 | 2.76 | 0.598 | 3.28, 3.56 | 0.771, 0.872 |
| 65,536 | 1 | 217.07, 216.29 | 74.68, 75.25 | 2.89 | 0.649 | 3.71, 3.66 | 0.930, 0.912 |
|  | 2 | 197.12, 197.73 | 73.81, 74.05 | 2.67 | 0.573 | 3.71, 3.66 | 0.930, 0.912 |
|  | both | 207.09, 207.01 | 74.25, 74.65 | 2.78 | 0.611 | 3.71, 3.66 | 0.930, 0.912 |
| 120,000 | 1 | 251.36, 253.63 | 92.15, 92.79 | 2.73 | 0.588 | 3.82, 3.20 | 0.964, 0.743 |
|  | 2 | 249.94, 249.96 | 90.34, 90.70 | 2.76 | 0.598 | 3.82, 3.20 | 0.964, 0.743 |
|  | both | 250.65, 251.79 | 91.25, 91.74 | 2.75 | 0.593 | 3.82, 3.20 | 0.964, 0.743 |

### The draft against no draft at another yield (est.)

Tokens a step measured elsewhere, at this run's step times: the yield over the break-even, minus one. An estimate, not a measured speed.

| Depth (tokens) | Yield | Tokens a step | Break-even on pass 1 | Est. on pass 1 | Break-even on pass 2 | Est. on pass 2 |
|---:|---|---:|---:|---:|---:|---:|
| 49,152 | the sampler off the rig, at 49,152 (Q1172) | 2.88 | 2.87 | +0.3 % | 2.65 | +8.5 % |
| 65,536 | the sampler off the rig (Q1251) | 2.93 | 2.89 | +1.4 % | 2.67 | +9.8 % |
|  | greedy off the rig (Q1251) | 3.46 | 2.89 | +19.7 % | 2.67 | +29.5 % |
| 120,000 | the sampler off the rig (Q1251) | 2.69 | 2.73 | -1.3 % | 2.76 | -2.4 % |
|  | greedy off the rig (Q1251) | 3.35 | 2.73 | +22.8 % | 2.76 | +21.5 % |

### `draft1` against `draft2`: the text and the step

Tok/s is the tokens a step over the step's time, so the difference between two loads of one setting is the product of two factors. Both passes of each load together.

| Depth (tokens) | tok/s: `draft1`, `draft2` | `draft1` against `draft2` | Tokens a step: `draft1`, `draft2` | Factor | Step, ms: `draft1`, `draft2` | Factor | Replies |
|---:|---|---:|---|---:|---|---:|---|
| 49,152 | 16.39, 19.23 | -14.8 % | 3.28, 3.56 | -7.7 % | 200.25, 184.85 | -7.7 % | different |
| 65,536 | 17.92, 17.67 | +1.4 % | 3.71, 3.66 | +1.4 % | 207.09, 207.01 | -0.0 % | different |
| 120,000 | 15.24, 12.71 | +19.9 % | 3.82, 3.20 | +19.4 % | 250.65, 251.79 | +0.5 % | different |

### Prompt reading: tok/s (ms a prompt token)

Per load: the tokens the prime request read at that depth over their time (at a depth without a prime, the first decode request). The draft adds: the difference between the arms' mean ms a prompt token, and its share of `draft`'s.

| Depth (tokens) | Read, on cached | `draft1` | `draft2` | `nodraft1` | `nodraft2` | `draft` against `nodraft` | Class | The draft adds, ms a prompt token | Share of the prompt time |
|---:|---|---|---|---|---|---:|---|---:|---:|
| 49,152 | 49028 on 60 | **100.5** (9.95) | **100.0** (10.00) | **146.2** (6.84) | **145.6** (6.87) | -31.3 % | loss | +3.12 | 31.3 % |
| 65,536 | 16644 on 48828 | **66.2** (15.10) | **65.4** (15.30) | **93.7** (10.68) | **93.2** (10.73) | -29.6 % | loss | +4.49 | 29.6 % |
| 120,000 | 54724 on 65212 | **49.0** (20.41) | **48.9** (20.46) | **73.6** (13.58) | **73.2** (13.66) | -33.3 % | loss | +6.81 | 33.3 % |

### A whole turn: the prompt it reads against the tokens it writes

A turn reads new prompt tokens and then generates. The draft saves time per generated token (the plain line's ms a token minus the draft line's, on pass 1, the first request at a depth) and costs time per prompt token (the table above). The last column is the saving over the cost: the new prompt tokens per generated token up to which the draft pays in a turn at that depth. For this run's loads the saving is between the arms' mean ms a token; for a yield measured elsewhere it is the plain token minus the step over that yield (est.).

| Depth (tokens) | The draft adds, ms a prompt token | Yield | Tokens a step | Saved, ms a generated token | Prompt tokens per generated token to break even |
|---:|---:|---|---:|---:|---:|
| 49,152 | +3.12 | this run's loads | 3.42 | +11.10 | 3.55 |
|  |  | the sampler off the rig, at 49,152 (Q1172) (est.) | 2.88 | +0.19 | 0.06 |
| 65,536 | +4.49 | this run's loads | 3.68 | +16.14 | 3.59 |
|  |  | the sampler off the rig (Q1251) (est.) | 2.93 | +1.04 | 0.23 |
|  |  | greedy off the rig (Q1251) (est.) | 3.46 | +12.32 | 2.74 |
| 120,000 | +6.81 | this run's loads | 3.51 | +19.95 | 2.93 |
|  |  | the sampler off the rig (Q1251) (est.) | 2.69 | -1.22 | none: nothing saved |
|  |  | greedy off the rig (Q1251) (est.) | 3.35 | +17.19 | 2.52 |

