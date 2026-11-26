## The draft against no draft: `draft` (draft1, draft2) and `nodraft` (nodraft1, nodraft2)

### Generation on both passes: tok/s (ms a token)

Per load: pass 1, pass 2 → the load, its two requests' tokens over their time. Spread: the difference between an arm's loads. Class of `draft` against `nodraft`: gain when its mean is above `nodraft`'s by more than the larger spread, loss when below by more than it, else negligible.

| Depth (tokens) | `draft1` | `draft2` | Spread | `nodraft1` | `nodraft2` | Spread | `draft` against `nodraft` | | Class |
|---:|---|---|---:|---|---|---:|---:|---:|---|
| 49,152 | 28.07, 29.20 → **28.63** (34.93) | 26.19, 29.16 → **27.60** (36.24) | 1.03 | 19.01, 19.39 → **19.20** (52.08) | 18.98, 19.37 → **19.17** (52.16) | 0.03 | +8.92 tok/s | +46.5 % | gain |
| 65,536 | 30.58, 31.12 → **30.85** (32.41) | 30.59, 31.10 → **30.84** (32.42) | 0.01 | 18.83, 19.13 → **18.98** (52.69) | 18.81, 19.09 → **18.95** (52.77) | 0.03 | +11.88 tok/s | +62.7 % | gain |
| 120,000 | 23.53, 24.45 → **23.98** (41.70) | 23.56, 24.43 → **23.99** (41.69) | 0.01 | 17.61, 18.17 → **17.88** (55.92) | 17.53, 18.16 → **17.84** (56.06) | 0.04 | +6.12 tok/s | +34.3 % | gain |
| all depths | **27.51** (36.35) | **27.19** (36.78) | 0.32 | **18.67** (53.57) | **18.63** (53.66) | 0.03 | +8.70 tok/s | +46.6 % | gain |

### Generation on pass 1: tok/s (ms a token)

Per load. Spread: the difference between an arm's loads. Class of `draft` against `nodraft`: gain when its mean is above `nodraft`'s by more than the larger spread, loss when below by more than it, else negligible.

| Depth (tokens) | `draft1` | `draft2` | Spread | `nodraft1` | `nodraft2` | Spread | `draft` against `nodraft` | | Class |
|---:|---|---|---:|---|---|---:|---:|---:|---|
| 49,152 | **28.07** (35.62) | **26.19** (38.19) | 1.89 | **19.01** (52.60) | **18.98** (52.69) | 0.03 | +8.13 tok/s | +42.8 % | gain |
| 65,536 | **30.58** (32.70) | **30.59** (32.69) | 0.01 | **18.83** (53.10) | **18.81** (53.18) | 0.03 | +11.77 tok/s | +62.5 % | gain |
| 120,000 | **23.53** (42.50) | **23.56** (42.45) | 0.03 | **17.61** (56.80) | **17.53** (57.05) | 0.08 | +5.98 tok/s | +34.0 % | gain |

### Generation on pass 2: tok/s (ms a token)

Per load. Spread: the difference between an arm's loads. Class of `draft` against `nodraft`: gain when its mean is above `nodraft`'s by more than the larger spread, loss when below by more than it, else negligible.

| Depth (tokens) | `draft1` | `draft2` | Spread | `nodraft1` | `nodraft2` | Spread | `draft` against `nodraft` | | Class |
|---:|---|---|---:|---|---|---:|---:|---:|---|
| 49,152 | **29.20** (34.24) | **29.16** (34.29) | 0.04 | **19.39** (51.57) | **19.37** (51.63) | 0.02 | +9.80 tok/s | +50.6 % | gain |
| 65,536 | **31.12** (32.13) | **31.10** (32.15) | 0.02 | **19.13** (52.29) | **19.09** (52.37) | 0.03 | +12.00 tok/s | +62.8 % | gain |
| 120,000 | **24.45** (40.90) | **24.43** (40.93) | 0.01 | **18.17** (55.05) | **18.16** (55.07) | 0.01 | +6.28 tok/s | +34.6 % | gain |

### The step: what it costs and what it must yield

A step of `draft` is one pass of the main model over the last token and the drafted ones, plus the draft's own passes; it yields one token and the draft tokens it accepts. A step of `nodraft` is one token. The draft pays when a step yields more tokens than its time is in tokens of the plain line: that ratio is the break-even, from the means of each arm's loads. The share to break even is the break-even minus one, over the draft tokens the arm offered a step.

| Depth (tokens) | Pass | Step of `draft`, ms: `draft1`, `draft2` | Token of `nodraft`, ms: `nodraft1`, `nodraft2` | Tokens a step to break even | Share accepted to break even | Tokens a step yielded: `draft1`, `draft2` | Share accepted: `draft1`, `draft2` |
|---:|---|---|---|---:|---:|---|---|
| 49,152 | 1 | 116.91, 125.33 | 52.60, 52.69 | 2.30 | 0.439 | 3.28, 3.28 | 0.771, 0.771 |
|  | 2 | 112.38, 112.54 | 51.57, 51.63 | 2.18 | 0.398 | 3.28, 3.28 | 0.771, 0.771 |
|  | both | 114.65, 118.94 | 52.08, 52.16 | 2.24 | 0.419 | 3.28, 3.28 | 0.771, 0.771 |
| 65,536 | 1 | 117.90, 117.87 | 53.10, 53.18 | 2.22 | 0.412 | 3.61, 3.61 | 0.881, 0.881 |
|  | 2 | 115.85, 115.93 | 52.29, 52.37 | 2.21 | 0.411 | 3.61, 3.61 | 0.881, 0.881 |
|  | both | 116.88, 116.90 | 52.69, 52.77 | 2.22 | 0.411 | 3.61, 3.61 | 0.881, 0.881 |
| 120,000 | 1 | 129.52, 129.36 | 56.80, 57.05 | 2.27 | 0.430 | 3.05, 3.05 | 0.691, 0.691 |
|  | 2 | 126.16, 126.23 | 55.05, 55.07 | 2.29 | 0.436 | 3.08, 3.08 | 0.703, 0.703 |
|  | both | 127.85, 127.80 | 55.92, 56.06 | 2.28 | 0.433 | 3.07, 3.07 | 0.697, 0.697 |

### The draft against no draft at another yield (est.)

Tokens a step measured elsewhere, at this run's step times: the yield over the break-even, minus one. An estimate, not a measured speed.

| Depth (tokens) | Yield | Tokens a step | Break-even on pass 1 | Est. on pass 1 | Break-even on pass 2 | Est. on pass 2 |
|---:|---|---:|---:|---:|---:|---:|
| 49,152 | this run's sampler pass (one draw a depth) | 3.37 | 2.30 | +46.5 % | 2.18 | +54.6 % |
|  | the preset's sampler off the rig (Q1172, Q1251) | 2.88 | 2.30 | +25.0 % | 2.18 | +32.0 % |
| 65,536 | this run's sampler pass (one draw a depth) | 3.12 | 2.22 | +40.6 % | 2.21 | +40.9 % |
|  | the preset's sampler off the rig (Q1172, Q1251) | 2.93 | 2.22 | +32.1 % | 2.21 | +32.3 % |
| 120,000 | this run's sampler pass (one draw a depth) | 3.71 | 2.27 | +63.2 % | 2.29 | +61.9 % |
|  | the preset's sampler off the rig (Q1172, Q1251) | 2.69 | 2.27 | +18.3 % | 2.29 | +17.4 % |

### `draft1` against `draft2`: the text and the step

Tok/s is the tokens a step over the step's time, so the difference between two loads of one setting is the product of two factors. Both passes of each load together.

| Depth (tokens) | tok/s: `draft1`, `draft2` | `draft1` against `draft2` | Tokens a step: `draft1`, `draft2` | Factor | Step, ms: `draft1`, `draft2` | Factor | Replies |
|---:|---|---:|---|---:|---|---:|---|
| 49,152 | 28.63, 27.60 | +3.7 % | 3.28, 3.28 | +0.0 % | 114.65, 118.94 | +3.7 % | the same |
| 65,536 | 30.85, 30.84 | +0.0 % | 3.61, 3.61 | +0.0 % | 116.88, 116.90 | +0.0 % | the same |
| 120,000 | 23.98, 23.99 | -0.0 % | 3.07, 3.07 | +0.0 % | 127.85, 127.80 | -0.0 % | the same |

### Prompt reading: tok/s (ms a prompt token)

Per load: the tokens the prime request read at that depth over their time (at a depth without a prime, the first decode request). The draft adds: the difference between the arms' mean ms a prompt token, and its share of `draft`'s.

| Depth (tokens) | Read, on cached | `draft1` | `draft2` | `nodraft1` | `nodraft2` | `draft` against `nodraft` | Class | The draft adds, ms a prompt token | Share of the prompt time |
|---:|---|---|---|---|---|---:|---|---:|---:|
| 49,152 | 49028 on 60 | **187.3** (5.34) | **187.1** (5.35) | **232.9** (4.29) | **233.0** (4.29) | -19.7 % | loss | +1.05 | 19.7 % |
| 65,536 | 16644 on 48828 | **128.8** (7.76) | **129.1** (7.75) | **154.1** (6.49) | **154.3** (6.48) | -16.4 % | loss | +1.27 | 16.4 % |
| 120,000 | 54724 on 65212 | **94.5** (10.59) | **94.3** (10.61) | **118.1** (8.47) | **118.0** (8.47) | -20.1 % | loss | +2.13 | 20.1 % |

### A whole turn: the prompt it reads against the tokens it writes

A turn reads new prompt tokens and then generates. The draft saves time per generated token (the plain line's ms a token minus the draft line's, on pass 1, the first request at a depth) and costs time per prompt token (the table above). The last column is the saving over the cost: the new prompt tokens per generated token up to which the draft pays in a turn at that depth. For this run's loads the saving is between the arms' mean ms a token; for a yield measured elsewhere it is the plain token minus the step over that yield (est.).

| Depth (tokens) | The draft adds, ms a prompt token | Yield | Tokens a step | Saved, ms a generated token | Prompt tokens per generated token to break even |
|---:|---:|---|---:|---:|---:|
| 49,152 | +1.05 | this run's loads | 3.28 | +15.74 | 14.98 |
|  |  | this run's sampler pass (one draw a depth) (est.) | 3.37 | +16.70 | 15.90 |
|  |  | the preset's sampler off the rig (Q1172, Q1251) (est.) | 2.88 | +10.53 | 10.02 |
| 65,536 | +1.27 | this run's loads | 3.61 | +20.44 | 16.09 |
|  |  | this run's sampler pass (one draw a depth) (est.) | 3.12 | +15.35 | 12.08 |
|  |  | the preset's sampler off the rig (Q1172, Q1251) (est.) | 2.93 | +12.90 | 10.15 |
| 120,000 | +2.13 | this run's loads | 3.05 | +14.45 | 6.80 |
|  |  | this run's sampler pass (one draw a depth) (est.) | 3.71 | +22.03 | 10.37 |
|  |  | the preset's sampler off the rig (Q1172, Q1251) (est.) | 2.69 | +8.81 | 4.14 |

### The same line twice: `draft` here and `draft` in v3.0's 131,072-token line (S38, R55)

Per load: tok/s on pass 1, pass 2 → the load with its ms a token; the share accepted and the tokens a step on pass 1; the step on pass 1 and on pass 2; the first six characters of the reply's hash on pass 1 (loads with the same characters wrote the same reply).

| Depth (tokens) | Load | tok/s (ms a token) | Share accepted | Tokens a step | Step on pass 1, ms | Step on pass 2, ms | Reply |
|---:|---|---|---:|---:|---:|---:|---|
| 49,152 | `draft1` | 28.07, 29.20 → **28.63** (34.93) | 0.771 | 3.28 | 116.91 | 112.38 | 55e7e0 |
|  | `draft2` | 26.19, 29.16 → **27.60** (36.24) | 0.771 | 3.28 | 125.33 | 112.54 | 55e7e0 |
|  | `draft1` (v3.0's 131,072-token line (S38, R55)) | 15.01, 18.05 → **16.39** (61.01) | 0.771 | 3.28 | 218.69 | 181.81 | 55e7e0 |
|  | `draft2` (v3.0's 131,072-token line (S38, R55)) | 19.03, 19.44 → **19.23** (51.99) | 0.872 | 3.56 | 186.82 | 182.88 | e79182 |
| | all 4 loads | fastest − slowest: 12.24 tok/s | | | 101.78 | 70.50 | |
| 65,536 | `draft1` | 30.58, 31.12 → **30.85** (32.41) | 0.881 | 3.61 | 117.90 | 115.85 | c5b3a7 |
|  | `draft2` | 30.59, 31.10 → **30.84** (32.42) | 0.881 | 3.61 | 117.87 | 115.93 | c5b3a7 |
|  | `draft1` (v3.0's 131,072-token line (S38, R55)) | 17.09, 18.82 → **17.92** (55.82) | 0.930 | 3.71 | 217.07 | 197.12 | 8a31b1 |
|  | `draft2` (v3.0's 131,072-token line (S38, R55)) | 16.91, 18.50 → **17.67** (56.60) | 0.912 | 3.66 | 216.29 | 197.73 | 51f0d9 |
| | all 4 loads | fastest − slowest: 13.18 tok/s | | | 99.20 | 81.87 | |
| 120,000 | `draft1` | 23.53, 24.45 → **23.98** (41.70) | 0.691 | 3.05 | 129.52 | 126.16 | a08411 |
|  | `draft2` | 23.56, 24.43 → **23.99** (41.69) | 0.691 | 3.05 | 129.36 | 126.23 | a08411 |
|  | `draft1` (v3.0's 131,072-token line (S38, R55)) | 15.20, 15.29 → **15.24** (65.60) | 0.964 | 3.82 | 251.36 | 249.94 | 619e14 |
|  | `draft2` (v3.0's 131,072-token line (S38, R55)) | 12.62, 12.80 → **12.71** (78.69) | 0.743 | 3.20 | 253.63 | 249.96 | 81ef51 |
| | all 4 loads | fastest − slowest: 11.28 tok/s | | | 124.27 | 123.80 | |

