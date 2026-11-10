# Probabilistic verification on the rig: time (66 requests in 12 sessions, 4 loads)

## Per load

| Arm | Load | Generated | Accepted of drafted | Share accepted | Steps | Tokens a step | Generation, s | Ms a step | Generation tok/s | Prompt tokens processed | Prompt tok/s | Replies at the limit | Rejected by the grammar |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pv0 | pv0-1 | 19,589 | 13,294 of 18,631 | 0.7135 | 6,295 | 3.112 | 950.7 | 152.63 | 20.60 | 63,823 | 75.57 | 33 | 0 |
| pv0 | pv0-2 | 19,536 | 13,298 of 18,508 | 0.7185 | 6,238 | 3.132 | 943.7 | 152.89 | 20.70 | 63,823 | 75.90 | 29 | 0 |
| pv1 | pv1-1 | 17,764 | 12,277 of 16,354 | 0.7507 | 5,487 | 3.237 | 833.4 | 153.74 | 21.31 | 63,887 | 75.75 | 29 | 15 |
| pv1 | pv1-2 | 20,977 | 14,539 of 19,178 | 0.7581 | 6,438 | 3.258 | 988.5 | 155.13 | 21.22 | 63,887 | 75.89 | 33 | 11 |
| recorded | the recorded session | 27,067 | 18,447 of 25,704 | 0.7177 | 8,620 | 3.140 | 1,326.8 | 155.10 | 20.40 | 39,863 | 69.42 |  |  |

## Each arm against `pv0`

| Metric | Better | A | B | A, each load | B, each load | A − B | % | Spread of A | Spread of B | Class (B's spread) | Class (the larger spread) | 95 % (requests) | 95 % (sessions) | Per request: mean A − B | 95 % | Requests higher / lower / equal |
|---|---|---|---|---|---|---:|---:|---:|---:|---|---|---|---|---:|---|---|
| generation tok/s | higher | pv1 | pv0 | 21.31, 21.22 | 20.60, 20.70 | +0.61 | +2.97 | 0.09 | 0.10 | gain | gain | +0.24 to +1.03 | +0.37 to +0.81 | +0.28 | -0.21 to +0.80 | 37 / 29 / 0 |
| ms a step | lower | pv1 | pv0 | 153.74, 155.13 | 152.63, 152.89 | +1.68 | +1.10 | 1.39 | 0.26 | loss | loss | +0.79 to +2.72 | +1.05 to +2.62 | +3.89 | +1.78 to +6.36 | 42 / 24 / 0 |
| tokens a step | higher | pv1 | pv0 | 3.237, 3.258 | 3.112, 3.132 | +0.126 | +4.04 | 0.021 | 0.020 | gain | gain | +0.069 to +0.186 | +0.088 to +0.161 | +0.125 | +0.060 to +0.191 | 47 / 17 / 2 |
| share accepted | higher | pv1 | pv0 | 0.7507, 0.7581 | 0.7135, 0.7185 | +0.0384 | +5.36 | 0.0074 | 0.0050 | gain | gain | +0.0190 to +0.0581 | +0.0261 to +0.0490 | +0.0305 | +0.0089 to +0.0531 | 45 / 19 / 2 |
| prompt tok/s | higher | pv1 | pv0 | 75.75, 75.89 | 75.57, 75.90 | +0.08 | +0.11 | 0.13 | 0.33 | negligible | negligible | -0.12 to +0.27 | -0.16 to +0.28 | +0.03 | -0.13 to +0.19 | 34 / 32 / 0 |

## One arm, its second load against its first

| Metric | Better | A | B | A, each load | B, each load | A − B | % | 95 % (requests) | 95 % (sessions) | Per request: mean A − B | 95 % | Requests higher / lower / equal |
|---|---|---|---|---|---|---:|---:|---|---|---:|---|---|
| generation tok/s | higher | pv0-2 | pv0-1 | 20.70 | 20.60 | +0.10 | +0.48 | -0.19 to +0.43 | -0.14 to +0.46 | +0.04 | -0.24 to +0.35 | 36 / 30 / 0 |
| ms a step | lower | pv0-2 | pv0-1 | 152.89 | 152.63 | +0.26 | +0.17 | -0.90 to +1.69 | -0.56 to +1.02 | +0.85 | -1.25 to +3.76 | 23 / 43 / 0 |
| tokens a step | higher | pv0-2 | pv0-1 | 3.132 | 3.112 | +0.020 | +0.64 | -0.015 to +0.064 | -0.015 to +0.071 | +0.018 | -0.023 to +0.062 | 18 / 20 / 28 |
| share accepted | higher | pv0-2 | pv0-1 | 0.7185 | 0.7135 | +0.0050 | +0.69 | -0.0070 to +0.0190 | -0.0065 to +0.0214 | +0.0028 | -0.0115 to +0.0177 | 16 / 24 / 26 |
| prompt tok/s | higher | pv0-2 | pv0-1 | 75.90 | 75.57 | +0.33 | +0.43 | +0.04 to +0.60 | +0.08 to +0.62 | +0.28 | +0.05 to +0.51 | 54 / 12 / 0 |
| generation tok/s | higher | pv1-2 | pv1-1 | 21.22 | 21.31 | -0.09 | -0.44 | -0.49 to +0.12 | -0.41 to +0.10 | -0.22 | -0.70 to +0.23 | 27 / 39 / 0 |
| ms a step | lower | pv1-2 | pv1-1 | 155.13 | 153.74 | +1.39 | +0.90 | -0.74 to +3.31 | -0.64 to +2.53 | +1.96 | -1.43 to +6.23 | 33 / 33 / 0 |
| tokens a step | higher | pv1-2 | pv1-1 | 3.258 | 3.237 | +0.021 | +0.64 | -0.055 to +0.060 | -0.047 to +0.062 | -0.017 | -0.073 to +0.037 | 15 / 30 / 21 |
| share accepted | higher | pv1-2 | pv1-1 | 0.7581 | 0.7507 | +0.0074 | +0.99 | -0.0166 to +0.0200 | -0.0146 to +0.0215 | -0.0038 | -0.0218 to +0.0138 | 18 / 28 / 20 |
| prompt tok/s | higher | pv1-2 | pv1-1 | 75.89 | 75.75 | +0.13 | +0.17 | -0.10 to +0.40 | -0.03 to +0.33 | +0.11 | -0.09 to +0.34 | 35 / 31 / 0 |

## By reply length

The requests in two halves by their reply limit (the recorded reply's length, the same in every load); pooled within a half.

| Half | Reply limit, tokens | Requests | Metric | Arm | Its loads | `pv0`, each load | Difference | % |
|---|---|---:|---|---|---|---|---:|---:|
| the shorter half | 43 to 172 | 33 | generation tok/s | pv1 | 22.02, 21.64 | 21.44, 21.28 | +0.47 | +2.20 |
| the shorter half | 43 to 172 | 33 | ms a step | pv1 | 157.15, 159.32 | 151.37, 154.90 | +5.10 | +3.33 |
| the shorter half | 43 to 172 | 33 | tokens a step | pv1 | 3.336, 3.320 | 3.134, 3.181 | +0.171 | +5.40 |
| the longer half | 183 to 8,747 | 33 | generation tok/s | pv1 | 21.17, 21.15 | 20.46, 20.60 | +0.63 | +3.08 |
| the longer half | 183 to 8,747 | 33 | ms a step | pv1 | 153.08, 154.48 | 152.85, 152.55 | +1.08 | +0.71 |
| the longer half | 183 to 8,747 | 33 | tokens a step | pv1 | 3.218, 3.248 | 3.108, 3.123 | +0.118 | +3.78 |

## A reply's time as a cost per reply and a cost per step

Per arm, least squares over every request in each of its loads: generation ms = a reply's cost + timed steps × (a step at depth 0 + its growth per 1,000 tokens of context × depth), and a second model with a cost per grammar rejection added (for an arm whose rows have any; a control without rejections keeps the first model, and the difference covers the shared terms). Depth is the prompt's tokens plus half the generated ones; the mean depth over all steps is 5,969 tokens. Intervals: 95 % bootstrap over requests, 2,000 resamples.

| Model | Arm | Rows | Term | Value | 95 % | Against `pv0` | 95 % |
|---|---|---:|---|---:|---|---:|---|
| steps and depth | pv0 | 132 | ms a reply | 223.38 | 49.15 to 340.96 |  | |
| steps and depth | pv0 | 132 | ms a step at depth 0 | 130.21 | 126.58 to 134.56 |  | |
| steps and depth | pv0 | 132 | ms a step per 1,000 tokens of context | 3.41 | 2.71 to 4.02 |  | |
| steps and depth | pv0 | 132 | ms a step at the mean depth | 150.54 | 149.90 to 152.98 |  | |
| steps and depth | pv1 | 132 | ms a reply | 301.47 | 161.60 to 427.26 | +78.09 | +19.13 to +171.03 |
| steps and depth | pv1 | 132 | ms a step at depth 0 | 132.42 | 127.96 to 134.66 | +2.21 | -2.05 to +3.58 |
| steps and depth | pv1 | 132 | ms a step per 1,000 tokens of context | 3.11 | 2.80 to 3.87 | -0.30 | -0.52 to +0.37 |
| steps and depth | pv1 | 132 | ms a step at the mean depth | 150.97 | 149.83 to 153.00 | +0.44 | -0.85 to +0.74 |
| steps, depth and grammar rejections | pv1 | 132 | ms a reply | 218.38 | 73.11 to 313.19 | -5.00 | -124.26 to +109.92 |
| steps, depth and grammar rejections | pv1 | 132 | ms a step at depth 0 | 132.38 | 127.57 to 134.70 | +2.17 | -2.33 to +3.51 |
| steps, depth and grammar rejections | pv1 | 132 | ms a step per 1,000 tokens of context | 3.13 | 2.81 to 3.98 | -0.28 | -0.50 to +0.45 |
| steps, depth and grammar rejections | pv1 | 132 | ms a step at the mean depth | 151.04 | 149.87 to 153.23 | +0.51 | -0.39 to +0.96 |
| steps, depth and grammar rejections | pv1 | 132 | ms a grammar rejection | 389.65 | 174.61 to 771.03 |  | |

## The same reply in two loads

| Loads | Kind | Same reply | Same counters |
|---|---|---|---|
| pv0-1, pv0-2 | one arm, two loads | 30 of 66 | 26 of 66 |
| pv0-1, pv1-1 | arm against the control | 2 of 66 | 2 of 66 |
| pv0-1, pv1-2 | arm against the control | 1 of 66 | 3 of 66 |
| pv0-2, pv1-1 | arm against the control | 1 of 66 | 2 of 66 |
| pv0-2, pv1-2 | arm against the control | 1 of 66 | 3 of 66 |
| pv1-1, pv1-2 | one arm, two loads | 18 of 66 | 19 of 66 |

## The server logs

| Load | Switches on its line | Generated | Requests with a grammar | Their tokens: sampled, drawn on the device | Rejected (after a device draw) | Accepted of drafted: the log's request lines; the replay's timings; the warm-up request | Checkpoint restores at a prompt | Error lines | Watchdog lines |
|---|---|---:|---:|---|---|---|---:|---:|---:|
| pv0-1 |  | 19,589 | 66 | 19,594, 19,594 | 0 (0) | 13,541 of 19,024; 13,294 of 18,631; 247 of 393 | 64 | 0 | 0 |
| pv0-2 |  | 19,536 | 66 | 19,550, 19,550 | 0 (0) | 13,557 of 18,895; 13,298 of 18,508; 259 of 387 | 64 | 0 | 0 |
| pv1-1 | --spec-draft-sampling probabilistic --spec-draft-p-min 0 | 17,764 | 66 | 17,809, 17,809 | 15 (15) | 12,422 of 16,549; 12,277 of 16,354; 145 of 195 | 65 | 0 | 0 |
| pv1-2 | --spec-draft-sampling probabilistic --spec-draft-p-min 0 | 20,977 | 66 | 21,017, 21,017 | 11 (11) | 14,873 of 19,589; 14,539 of 19,178; 334 of 411 | 65 | 0 | 0 |

## The loads' conditions

| Load | Replay | Minutes | Cards: mean °C | Hottest card, mean °C | Mean SM clock, MHz | Lowest SM clock, MHz | Mean power, W | Mean utilization, % | Memory used per card, MiB | Lowest MemAvailable, MiB | Swap used at the replay's start, end, highest, MiB | Swapped out, in during the replay, MiB | Watchdog kills |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---|---:|
| pv0-1 | 11:13:04 to 11:43:04 | 30.0 | 40.1 | 43.3 | 1862 | 1354 | 37.0 | 69.1 | 5101 5137 5005 5153 4913 5021 4423 5033 4625 | 76 | 20, 342, 342 | 342.7, 98.9 | 0 |
| pv0-2 | 12:22:48 to 12:52:37 | 29.8 | 36.8 | 40.4 | 1869 | 1354 | 36.7 | 67.9 | 5101 5137 5005 5153 4913 5021 4423 5033 4625 | 135 | 24, 439, 518 | 457.0, 134.9 | 0 |
| pv1-1 | 11:48:47 to 12:16:48 | 28.0 | 37.5 | 41.0 | 1859 | 1354 | 36.8 | 64.4 | 5101 5137 5005 5153 4913 5021 4423 5033 4625 | 68 | 24, 335, 335 | 337.7, 103.2 | 0 |
| pv1-2 | 12:58:34 to 13:29:10 | 30.6 | 37.2 | 40.6 | 1864 | 1354 | 36.2 | 69.2 | 5101 5137 5005 5153 4913 5021 4423 5033 4625 | 72 | 25, 335, 406 | 350.4, 132.7 | 0 |
