## The draft against no draft: `draft` (draft1) and `nodraft` (nodraft1)

### Generation on both passes: tok/s (ms a token)

Per load: pass 1, pass 2 → the load, its two requests' tokens over their time. Spread: the difference between an arm's loads. Class of `draft` against `nodraft`: gain when its mean is above `nodraft`'s by more than the larger spread, loss when below by more than it, else negligible.

| Depth (tokens) | `draft1` | Spread | `nodraft1` | Spread | `draft` against `nodraft` | | Class |
|---:|---|---:|---|---:|---:|---:|---|
| 120,000 | 12.08, 13.18 → **12.61** (79.31) | — | 9.65, 11.06 → **10.31** (97.03) | — | +2.30 tok/s | +22.3 % | no class (one load) |
| 196,000 | 10.50, 10.62 → **10.56** (94.71) | — | 8.75, 8.86 → **8.80** (113.58) | — | +1.75 tok/s | +19.9 % | no class (one load) |
| all depths | **11.49** (87.01) | — | **9.50** (105.31) | — | +2.00 tok/s | +21.0 % | no class (one load) |

### Generation on pass 1: tok/s (ms a token)

Per load. Spread: the difference between an arm's loads. Class of `draft` against `nodraft`: gain when its mean is above `nodraft`'s by more than the larger spread, loss when below by more than it, else negligible.

| Depth (tokens) | `draft1` | Spread | `nodraft1` | Spread | `draft` against `nodraft` | | Class |
|---:|---|---:|---|---:|---:|---:|---|
| 120,000 | **12.08** (82.77) | — | **9.65** (103.65) | — | +2.43 tok/s | +25.2 % | no class (one load) |
| 196,000 | **10.50** (95.26) | — | **8.75** (114.24) | — | +1.74 tok/s | +19.9 % | no class (one load) |

### Generation on pass 2: tok/s (ms a token)

Per load. Spread: the difference between an arm's loads. Class of `draft` against `nodraft`: gain when its mean is above `nodraft`'s by more than the larger spread, loss when below by more than it, else negligible.

| Depth (tokens) | `draft1` | Spread | `nodraft1` | Spread | `draft` against `nodraft` | | Class |
|---:|---|---:|---|---:|---:|---:|---|
| 120,000 | **13.18** (75.85) | — | **11.06** (90.42) | — | +2.12 tok/s | +19.2 % | no class (one load) |
| 196,000 | **10.62** (94.17) | — | **8.86** (112.92) | — | +1.76 tok/s | +19.9 % | no class (one load) |

### The step: what it costs and what it must yield

A step of `draft` is one pass of the main model over the last token and the drafted ones, plus the draft's own passes; it yields one token and the draft tokens it accepts. A step of `nodraft` is one token. The draft pays when a step yields more tokens than its time is in tokens of the plain line: that ratio is the break-even, from the means of each arm's loads. The share to break even is the break-even minus one, over the draft tokens the arm offered a step.

| Depth (tokens) | Pass | Step of `draft`, ms: `draft1` | Token of `nodraft`, ms: `nodraft1` | Tokens a step to break even | Share accepted to break even | Tokens a step yielded: `draft1` | Share accepted: `draft1` |
|---:|---|---|---|---:|---:|---|---|
| 120,000 | 1 | 268.23 | 103.65 | 2.59 | 0.543 | 3.24 | 0.766 |
|  | 2 | 248.95 | 90.42 | 2.75 | 0.600 | 3.28 | 0.781 |
|  | both | 258.65 | 97.03 | 2.67 | 0.570 | 3.26 | 0.773 |
| 196,000 | 1 | 325.15 | 114.24 | 2.85 | 0.624 | 3.41 | 0.815 |
|  | 2 | 321.43 | 112.92 | 2.85 | 0.624 | 3.41 | 0.815 |
|  | both | 323.29 | 113.58 | 2.85 | 0.624 | 3.41 | 0.815 |

### The draft against no draft at another yield (est.)

Tokens a step measured elsewhere, at this run's step times: the yield over the break-even, minus one. An estimate, not a measured speed.

| Depth (tokens) | Yield | Tokens a step | Break-even on pass 1 | Est. on pass 1 | Break-even on pass 2 | Est. on pass 2 |
|---:|---|---:|---:|---:|---:|---:|
| 120,000 | the sampler off the rig, at 49,152 | 2.88 | 2.59 | +11.1 % | 2.75 | +4.5 % |
| 196,000 | the sampler off the rig, at 49,152 | 2.88 | 2.85 | +1.0 % | 2.85 | +1.0 % |

### Prompt reading: tok/s (ms a prompt token)

Per load: the tokens the prime request read at that depth over their time (at a depth without a prime, the first decode request). The draft adds: the difference between the arms' mean ms a prompt token, and its share of `draft`'s.

| Depth (tokens) | Read, on cached | `draft1` | `nodraft1` | `draft` against `nodraft` | Class | The draft adds, ms a prompt token | Share of the prompt time |
|---:|---|---|---|---:|---|---:|---:|
| 120,000 | 119876 on 60 | **65.4** (15.29) | **97.9** (10.22) | -33.2 % | no class (one load) | +5.07 | 33.2 % |
| 196,000 | 76260 on 119676 | **32.9** (30.44) | **50.3** (19.89) | -34.6 % | no class (one load) | +10.55 | 34.6 % |

### A whole turn: the prompt it reads against the tokens it writes

A turn reads new prompt tokens and then generates. The draft saves time per generated token (the plain line's ms a token minus the draft line's, on pass 1, the first request at a depth) and costs time per prompt token (the table above). The last column is the saving over the cost: the new prompt tokens per generated token up to which the draft pays in a turn at that depth. For this run's loads the saving is between the arms' mean ms a token; for a yield measured elsewhere it is the plain token minus the step over that yield (est.).

| Depth (tokens) | The draft adds, ms a prompt token | Yield | Tokens a step | Saved, ms a generated token | Prompt tokens per generated token to break even |
|---:|---:|---|---:|---:|---:|
| 120,000 | +5.07 | this run's loads | 3.24 | +20.87 | 4.12 |
|  |  | the sampler off the rig, at 49,152 (est.) | 2.88 | +10.38 | 2.05 |
| 196,000 | +10.55 | this run's loads | 3.41 | +18.98 | 1.80 |
|  |  | the sampler off the rig, at 49,152 (est.) | 2.88 | +1.18 | 0.11 |

