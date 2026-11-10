## The draft against no draft: `draft` (draft1, draft2) and `nodraft` (nodraft1, nodraft2)

### Generation on both passes: tok/s (ms a token)

Per load: pass 1, pass 2 → the load, its two requests' tokens over their time. Spread: the difference between an arm's loads. Class of `draft` against `nodraft`: gain when its mean is above `nodraft`'s by more than the larger spread, loss when below by more than it, else negligible.

| Depth (tokens) | `draft1` | `draft2` | Spread | `nodraft1` | `nodraft2` | Spread | `draft` against `nodraft` | | Class |
|---:|---|---|---:|---|---|---:|---:|---:|---|
| 0 | 26.38, 27.71 → **27.03** (37.00) | 25.79, 27.69 → **26.71** (37.44) | 0.32 | 19.04, 19.49 → **19.26** (51.92) | 19.04, 19.48 → **19.26** (51.92) | 0.00 | +7.61 tok/s | +39.5 % | gain |
| 8,192 | 11.57, 20.16 → **14.70** (68.02) | 17.16, 24.11 → **20.05** (49.88) | 5.35 | 16.03, 17.82 → **16.87** (59.26) | 16.22, 17.82 → **16.98** (58.89) | 0.11 | +0.45 tok/s | +2.6 % | negligible |
| 16,384 | 18.43, 24.06 → **20.87** (47.91) | 20.61, 23.92 → **22.14** (45.16) | 1.27 | 16.66, 16.94 → **16.80** (59.53) | 16.67, 16.96 → **16.81** (59.47) | 0.02 | +4.70 tok/s | +28.0 % | gain |
| 32,768 | 16.80, 18.86 → **17.77** (56.27) | 12.76, 15.45 → **13.98** (71.55) | 3.79 | 15.26, 15.65 → **15.45** (64.71) | 14.96, 15.43 → **15.19** (65.82) | 0.26 | +0.55 tok/s | +3.6 % | negligible |
| 49,152 | 17.86, 17.98 → **17.92** (55.80) | 16.83, 17.45 → **17.13** (58.36) | 0.79 | 14.03, 14.53 → **14.28** (70.04) | 13.76, 14.54 → **14.14** (70.73) | 0.14 | +3.32 tok/s | +23.4 % | gain |
| all depths | **18.87** (53.00) | **19.06** (52.48) | 0.19 | **16.37** (61.09) | **16.30** (61.37) | 0.07 | +2.63 tok/s | +16.1 % | gain |

### Generation on pass 1: tok/s (ms a token)

Per load. Spread: the difference between an arm's loads. Class of `draft` against `nodraft`: gain when its mean is above `nodraft`'s by more than the larger spread, loss when below by more than it, else negligible.

| Depth (tokens) | `draft1` | `draft2` | Spread | `nodraft1` | `nodraft2` | Spread | `draft` against `nodraft` | | Class |
|---:|---|---|---:|---|---|---:|---:|---:|---|
| 0 | **26.38** (37.91) | **25.79** (38.77) | 0.58 | **19.04** (52.52) | **19.04** (52.53) | 0.00 | +7.05 tok/s | +37.0 % | gain |
| 8,192 | **11.57** (86.44) | **17.16** (58.27) | 5.59 | **16.03** (62.39) | **16.22** (61.66) | 0.19 | -1.76 tok/s | -10.9 % | negligible |
| 16,384 | **18.43** (54.26) | **20.61** (48.52) | 2.18 | **16.66** (60.01) | **16.67** (59.99) | 0.00 | +2.85 tok/s | +17.1 % | gain |
| 32,768 | **16.80** (59.51) | **12.76** (78.36) | 4.04 | **15.26** (65.54) | **14.96** (66.84) | 0.30 | -0.33 tok/s | -2.2 % | negligible |
| 49,152 | **17.86** (55.98) | **16.83** (59.42) | 1.03 | **14.03** (71.27) | **13.76** (72.66) | 0.27 | +3.45 tok/s | +24.8 % | gain |

### Generation on pass 2: tok/s (ms a token)

Per load. Spread: the difference between an arm's loads. Class of `draft` against `nodraft`: gain when its mean is above `nodraft`'s by more than the larger spread, loss when below by more than it, else negligible.

| Depth (tokens) | `draft1` | `draft2` | Spread | `nodraft1` | `nodraft2` | Spread | `draft` against `nodraft` | | Class |
|---:|---|---|---:|---|---|---:|---:|---:|---|
| 0 | **27.71** (36.09) | **27.69** (36.11) | 0.01 | **19.49** (51.32) | **19.48** (51.32) | 0.00 | +8.21 tok/s | +42.2 % | gain |
| 8,192 | **20.16** (49.60) | **24.11** (41.48) | 3.95 | **17.82** (56.13) | **17.82** (56.13) | 0.00 | +4.32 tok/s | +24.2 % | gain |
| 16,384 | **24.06** (41.57) | **23.92** (41.80) | 0.13 | **16.94** (59.05) | **16.96** (58.95) | 0.03 | +7.04 tok/s | +41.5 % | gain |
| 32,768 | **18.86** (53.03) | **15.45** (64.73) | 3.41 | **15.65** (63.88) | **15.43** (64.80) | 0.22 | +1.61 tok/s | +10.4 % | negligible |
| 49,152 | **17.98** (55.62) | **17.45** (57.30) | 0.53 | **14.53** (68.81) | **14.54** (68.80) | 0.00 | +3.18 tok/s | +21.9 % | gain |

### The step: what it costs and what it must yield

A step of `draft` is one pass of the main model over the last token and the drafted ones, plus the draft's own passes; it yields one token and the draft tokens it accepts. A step of `nodraft` is one token. The draft pays when a step yields more tokens than its time is in tokens of the plain line: that ratio is the break-even, from the means of each arm's loads. The share to break even is the break-even minus one, over the draft tokens the arm offered a step.

| Depth (tokens) | Pass | Step of `draft`, ms: `draft1`, `draft2` | Token of `nodraft`, ms: `nodraft1`, `nodraft2` | Tokens a step to break even | Share accepted to break even | Tokens a step yielded: `draft1`, `draft2` | Share accepted: `draft1`, `draft2` |
|---:|---|---|---|---:|---:|---|---|
| 0 | 1 | 124.42, 127.24 | 52.52, 52.53 | 2.40 | 0.475 | 3.28, 3.28 | 0.777, 0.777 |
|  | 2 | 123.20, 123.26 | 51.32, 51.32 | 2.40 | 0.478 | 3.41, 3.41 | 0.823, 0.823 |
|  | both | 123.82, 125.29 | 51.92, 51.92 | 2.40 | 0.477 | 3.35, 3.35 | 0.800, 0.800 |
| 8,192 | 1 | 248.63, 207.18 | 62.39, 61.66 | 3.67 | 0.913 | 2.88, 3.56 | 0.637, 0.876 |
|  | 2 | 151.17, 149.56 | 56.13, 56.13 | 2.68 | 0.573 | 3.05, 3.61 | 0.696, 0.894 |
|  | both | 201.31, 178.57 | 59.26, 58.89 | 3.22 | 0.756 | 2.96, 3.58 | 0.666, 0.885 |
| 16,384 | 1 | 195.63, 174.94 | 60.01, 59.99 | 3.09 | 0.711 | 3.61, 3.61 | 0.894, 0.881 |
|  | 2 | 149.89, 150.72 | 59.05, 58.95 | 2.55 | 0.527 | 3.61, 3.61 | 0.894, 0.881 |
|  | both | 172.76, 162.83 | 59.53, 59.47 | 2.82 | 0.620 | 3.61, 3.61 | 0.894, 0.881 |
| 32,768 | 1 | 185.78, 194.76 | 65.54, 66.84 | 2.87 | 0.634 | 3.12, 2.49 | 0.719, 0.502 |
|  | 2 | 165.57, 165.72 | 63.88, 64.80 | 2.57 | 0.533 | 3.12, 2.56 | 0.719, 0.527 |
|  | both | 175.68, 180.45 | 64.71, 65.82 | 2.73 | 0.585 | 3.12, 2.52 | 0.719, 0.514 |
| 49,152 | 1 | 186.13, 187.79 | 71.27, 72.66 | 2.60 | 0.542 | 3.32, 3.16 | 0.792, 0.729 |
|  | 2 | 184.90, 181.11 | 68.81, 68.80 | 2.66 | 0.563 | 3.32, 3.16 | 0.792, 0.729 |
|  | both | 185.52, 184.45 | 70.04, 70.73 | 2.63 | 0.552 | 3.32, 3.16 | 0.792, 0.729 |

### The draft against no draft at another yield (est.)

Tokens a step measured elsewhere, at this run's step times: the yield over the break-even, minus one. An estimate, not a measured speed.

| Depth (tokens) | Yield | Tokens a step | Break-even on pass 1 | Est. on pass 1 | Break-even on pass 2 | Est. on pass 2 |
|---:|---|---:|---:|---:|---:|---:|
| 0 | the sampler off the rig | 2.23 | 2.40 | -7.1 % | 2.40 | -7.3 % |
|  | greedy off the rig | 2.96 | 2.40 | +23.8 % | 2.40 | +23.5 % |
| 8,192 | the sampler off the rig | 2.87 | 3.67 | -22.0 % | 2.68 | +7.0 % |
|  | greedy off the rig | 2.95 | 3.67 | -19.6 % | 2.68 | +10.3 % |
| 16,384 | the sampler off the rig | 3.24 | 3.09 | +5.0 % | 2.55 | +27.2 % |
|  | greedy off the rig | 3.54 | 3.09 | +14.6 % | 2.55 | +38.9 % |
| 32,768 | the sampler off the rig | 2.84 | 2.87 | -1.1 % | 2.57 | +10.5 % |
|  | greedy off the rig | 3.20 | 2.87 | +11.3 % | 2.57 | +24.3 % |
| 49,152 | the sampler off the rig | 2.88 | 2.60 | +10.7 % | 2.66 | +8.1 % |
|  | greedy off the rig | 3.41 | 2.60 | +31.4 % | 2.66 | +28.3 % |

### `draft1` against `draft2`: the text and the step

Tok/s is the tokens a step over the step's time, so the difference between two loads of one setting is the product of two factors. Both passes of each load together.

| Depth (tokens) | tok/s: `draft1`, `draft2` | `draft1` against `draft2` | Tokens a step: `draft1`, `draft2` | Factor | Step, ms: `draft1`, `draft2` | Factor | Replies |
|---:|---|---:|---|---:|---|---:|---|
| 0 | 27.03, 26.71 | +1.2 % | 3.35, 3.35 | +0.0 % | 123.82, 125.29 | +1.2 % | the same |
| 8,192 | 14.70, 20.05 | -26.7 % | 2.96, 3.58 | -17.3 % | 201.31, 178.57 | -11.3 % | different |
| 16,384 | 20.87, 22.14 | -5.7 % | 3.61, 3.61 | +0.0 % | 172.76, 162.83 | -5.7 % | different |
| 32,768 | 17.77, 13.98 | +27.1 % | 3.12, 2.52 | +23.8 % | 175.68, 180.45 | +2.7 % | different |
| 49,152 | 17.92, 17.13 | +4.6 % | 3.32, 3.16 | +5.2 % | 185.52, 184.45 | -0.6 % | different |

### Prompt reading: tok/s (ms a prompt token)

Per load: the tokens the prime request read at that depth over their time (at a depth without a prime, the first decode request). The draft adds: the difference between the arms' mean ms a prompt token, and its share of `draft`'s.

| Depth (tokens) | Read, on cached | `draft1` | `draft2` | `nodraft1` | `nodraft2` | `draft` against `nodraft` | Class | The draft adds, ms a prompt token | Share of the prompt time |
|---:|---|---|---|---|---|---:|---|---:|---:|
| 0 | 64 on 0 | **32.9** (30.41) | **32.7** (30.59) | **33.4** (29.93) | **33.1** (30.26) | -1.3 % | loss | +0.41 | 1.3 % |
| 8,192 | 8128 on 0 | **127.7** (7.83) | **127.2** (7.86) | **179.4** (5.57) | **176.6** (5.66) | -28.4 % | loss | +2.23 | 28.4 % |
| 16,384 | 8708 on 7612 | **107.6** (9.29) | **109.3** (9.15) | **164.9** (6.07) | **166.6** (6.00) | -34.6 % | loss | +3.19 | 34.6 % |
| 32,768 | 16388 on 16316 | **90.3** (11.07) | **91.0** (10.99) | **148.3** (6.74) | **148.1** (6.75) | -38.8 % | loss | +4.28 | 38.8 % |
| 49,152 | 16900 on 32188 | **69.9** (14.30) | **70.3** (14.22) | **119.5** (8.37) | **119.4** (8.37) | -41.3 % | loss | +5.89 | 41.3 % |

### A whole turn: the prompt it reads against the tokens it writes

A turn reads new prompt tokens and then generates. The draft saves time per generated token (the plain line's ms a token minus the draft line's, on pass 1, the first request at a depth) and costs time per prompt token (the table above). The last column is the saving over the cost: the new prompt tokens per generated token up to which the draft pays in a turn at that depth. For this run's loads the saving is between the arms' mean ms a token; for a yield measured elsewhere it is the plain token minus the step over that yield (est.).

| Depth (tokens) | The draft adds, ms a prompt token | Yield | Tokens a step | Saved, ms a generated token | Prompt tokens per generated token to break even |
|---:|---:|---|---:|---:|---:|
| 0 | +0.41 | this run's loads | 3.28 | +14.18 | 34.76 |
|  |  | the sampler off the rig (est.) | 2.23 | -4.00 | none: nothing saved |
|  |  | greedy off the rig (est.) | 2.96 | +10.08 | 24.71 |
| 8,192 | +2.23 | this run's loads | 3.22 | -10.33 | none: nothing saved |
|  |  | the sampler off the rig (est.) | 2.87 | -17.50 | none: nothing saved |
|  |  | greedy off the rig (est.) | 2.95 | -15.13 | none: nothing saved |
| 16,384 | +3.19 | this run's loads | 3.61 | +8.61 | 2.70 |
|  |  | the sampler off the rig (est.) | 3.24 | +2.83 | 0.89 |
|  |  | greedy off the rig (est.) | 3.54 | +7.65 | 2.40 |
| 32,768 | +4.28 | this run's loads | 2.80 | -2.74 | none: nothing saved |
|  |  | the sampler off the rig (est.) | 2.84 | -0.71 | none: nothing saved |
|  |  | greedy off the rig (est.) | 3.20 | +6.73 | 1.57 |
| 49,152 | +5.89 | this run's loads | 3.24 | +14.26 | 2.42 |
|  |  | the sampler off the rig (est.) | 2.88 | +6.96 | 1.18 |
|  |  | greedy off the rig (est.) | 3.41 | +17.18 | 2.92 |

### The same line twice: `draft` here and `sparse` in the first pair

Per load: tok/s on pass 1, pass 2 → the load with its ms a token; the share accepted and the tokens a step on pass 1; the step on pass 1 and on pass 2; the first six characters of the reply's hash on pass 1 (loads with the same characters wrote the same reply).

| Depth (tokens) | Load | tok/s (ms a token) | Share accepted | Tokens a step | Step on pass 1, ms | Step on pass 2, ms | Reply |
|---:|---|---|---:|---:|---:|---:|---|
| 0 | `draft1` | 26.38, 27.71 → **27.03** (37.00) | 0.777 | 3.28 | 124.42 | 123.20 | 08569d |
|  | `draft2` | 25.79, 27.69 → **26.71** (37.44) | 0.777 | 3.28 | 127.24 | 123.26 | 08569d |
|  | `sparse1` (the first pair) | 25.81, 27.67 → **26.71** (37.44) | 0.777 | 3.28 | 127.14 | 123.36 | 08569d |
|  | `sparse2` (the first pair) | 25.78, 27.67 → **26.69** (37.47) | 0.777 | 3.28 | 127.32 | 123.38 | 08569d |
| | all 4 loads | fastest − slowest: 0.34 tok/s | | | 2.90 | 0.18 | |
| 8,192 | `draft1` | 11.57, 20.16 → **14.70** (68.02) | 0.637 | 2.88 | 248.63 | 151.17 | 28577f |
|  | `draft2` | 17.16, 24.11 → **20.05** (49.88) | 0.876 | 3.56 | 207.18 | 149.56 | 8b3879 |
|  | `sparse1` (the first pair) | 17.29, 25.40 → **20.58** (48.60) | 0.964 | 3.82 | 221.00 | 150.40 | a84769 |
|  | `sparse2` (the first pair) | 17.50, 25.42 → **20.73** (48.25) | 0.964 | 3.82 | 218.37 | 150.34 | a84769 |
| | all 4 loads | fastest − slowest: 6.02 tok/s | | | 41.45 | 1.60 | |
| 16,384 | `draft1` | 18.43, 24.06 → **20.87** (47.91) | 0.894 | 3.61 | 195.63 | 149.89 | e766b3 |
|  | `draft2` | 20.61, 23.92 → **22.14** (45.16) | 0.881 | 3.61 | 174.94 | 150.72 | 41dce8 |
|  | `sparse1` (the first pair) | 18.19, 23.75 → **20.60** (48.54) | 0.876 | 3.56 | 195.46 | 149.73 | 625c60 |
|  | `sparse2` (the first pair) | 18.51, 24.25 → **20.99** (47.64) | 0.881 | 3.61 | 194.84 | 150.80 | 58515e |
| | all 4 loads | fastest − slowest: 1.54 tok/s | | | 20.70 | 1.07 | |
| 32,768 | `draft1` | 16.80, 18.86 → **17.77** (56.27) | 0.719 | 3.12 | 185.78 | 165.57 | 71c146 |
|  | `draft2` | 12.76, 15.45 → **13.98** (71.55) | 0.502 | 2.49 | 194.76 | 165.72 | a940e1 |
|  | `sparse1` (the first pair) | 14.79, 19.29 → **16.75** (59.71) | 0.746 | 3.20 | 216.30 | 165.86 | 76507e |
|  | `sparse2` (the first pair) | 17.07, 19.31 → **18.12** (55.18) | 0.732 | 3.16 | 185.10 | 165.74 | f22b9e |
| | all 4 loads | fastest − slowest: 4.15 tok/s | | | 31.20 | 0.29 | |
| 49,152 | `draft1` | 17.86, 17.98 → **17.92** (55.80) | 0.792 | 3.32 | 186.13 | 184.90 | 5dfb04 |
|  | `draft2` | 16.83, 17.45 → **17.13** (58.36) | 0.729 | 3.16 | 187.79 | 181.11 | 22ebf9 |
|  | `sparse1` (the first pair) | 17.06, 18.63 → **17.81** (56.13) | 0.807 | 3.37 | 197.40 | 180.77 | 5fe0ff |
|  | `sparse2` (the first pair) | 13.91, 15.55 → **14.68** (68.11) | 0.616 | 2.81 | 202.27 | 180.96 | 30320e |
| | all 4 loads | fastest − slowest: 3.24 tok/s | | | 16.14 | 4.14 | |

