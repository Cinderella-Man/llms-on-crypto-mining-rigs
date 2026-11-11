trace: 3187 evaluations in 5 pieces (cut at gaps over 2000 ms): 9 evaluations and 1 main passes, 2 evaluations and 0 main passes, 1478 evaluations and 88 main passes, 90 evaluations and 6 main passes, 1608 evaluations and 94 main passes
evaluations between main passes, by card: card 4: 714, card 8: 714; the draft's last card: 8

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps |
|---|---:|---:|---:|---|---:|
| greedy | 88 | 3 | 84 | 4: 84 | 80 |
| long | 94 | 0 | 93 | 4: 93 | 90 |

**trace, all requests: 170 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 138.96 | 137.47 | 128.76 | 143.64 |
| verify_span | 110.09 | 112.18 | 105.24 | 114.26 |
|   verify_gpu (the nine cards) | 105.68 | 107.69 | 101.03 | 109.71 |
|   verify_rest (crossings, rows back, acceptance, server) | 4.40 | 4.45 | 4.23 | 4.57 |
| pass 0 (catch-up): lead | 0.66 | 0.68 | 0.60 | 0.71 |
| pass 0: gpu | 4.04 | 4.22 | 3.67 | 4.37 |
| pass 0: turn | 0.34 | 0.29 | 0.28 | 0.31 |
| pass 1 (draft 1): lead | 0.63 | 0.65 | 0.58 | 0.67 |
| pass 1: gpu | 3.39 | 3.02 | 2.53 | 6.39 |
| pass 1: turn | 0.53 | 0.52 | 0.51 | 0.55 |
| pass 2 (draft 2): lead | 0.62 | 0.64 | 0.57 | 0.67 |
| pass 2: gpu | 3.47 | 2.97 | 2.53 | 6.37 |
| pass 2: turn | 0.52 | 0.51 | 0.47 | 0.57 |
| pass 3 (draft 3): lead | 0.62 | 0.65 | 0.57 | 0.68 |
| pass 3: gpu | 3.31 | 2.96 | 2.52 | 6.33 |
| pass 3: turn (the server before the next verify) | 10.75 | 7.74 | 6.25 | 10.22 |
| the draft's span (all passes) | 28.88 | 24.65 | 22.32 | 36.42 |
|   the draft's GPU time on its last card | 14.20 | 13.22 | 11.28 | 22.80 |
|   the leads (crossings) | 2.53 | 2.64 | 2.33 | 2.71 |
|   the turns of the passes before the last | 1.39 | 1.33 | 1.27 | 1.44 |
|   the draft's GPU time on the embedding card | 0.08 | 0.08 | 0.08 | 0.09 |
| the host's launch of the main pass (inside verify_span) | 4.75 | 4.59 | 3.82 | 4.75 |

verify_gpu per card, median ms: 12.94, 12.24, 11.76, 12.41, 10.14, 11.86, 10.63, 11.87, 13.56

**trace, greedy: 80 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 131.93 | 130.25 | 127.28 | 141.49 |
| verify_span | 106.01 | 105.91 | 104.92 | 107.00 |
|   verify_gpu (the nine cards) | 101.76 | 101.65 | 100.61 | 102.72 |
|   verify_rest (crossings, rows back, acceptance, server) | 4.26 | 4.26 | 4.20 | 4.31 |
| pass 0 (catch-up): lead | 0.61 | 0.61 | 0.59 | 0.63 |
| pass 0: gpu | 3.71 | 3.70 | 3.65 | 3.75 |
| pass 0: turn | 0.35 | 0.28 | 0.28 | 0.43 |
| pass 1 (draft 1): lead | 0.59 | 0.59 | 0.58 | 0.60 |
| pass 1: gpu | 3.29 | 2.56 | 2.51 | 6.38 |
| pass 1: turn | 0.52 | 0.51 | 0.50 | 0.54 |
| pass 2 (draft 2): lead | 0.58 | 0.58 | 0.57 | 0.59 |
| pass 2: gpu | 3.42 | 2.56 | 2.49 | 6.34 |
| pass 2: turn | 0.51 | 0.49 | 0.47 | 0.56 |
| pass 3 (draft 3): lead | 0.57 | 0.57 | 0.56 | 0.59 |
| pass 3: gpu | 3.21 | 2.55 | 2.51 | 6.32 |
| pass 3: turn (the server before the next verify) | 8.57 | 8.43 | 6.41 | 10.63 |
| the draft's span (all passes) | 25.91 | 24.16 | 21.92 | 35.03 |
|   the draft's GPU time on its last card | 13.62 | 11.37 | 11.26 | 22.73 |
|   the leads (crossings) | 2.35 | 2.34 | 2.32 | 2.38 |
|   the turns of the passes before the last | 1.37 | 1.29 | 1.26 | 1.88 |
|   the draft's GPU time on the embedding card | 0.09 | 0.08 | 0.08 | 0.09 |
| the host's launch of the main pass (inside verify_span) | 4.22 | 3.89 | 3.80 | 4.01 |

verify_gpu per card, median ms: 12.51, 11.17, 11.40, 11.52, 9.68, 11.41, 9.57, 11.44, 12.97

**trace, long: 90 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 145.21 | 138.49 | 136.47 | 151.61 |
| verify_span | 113.71 | 113.60 | 112.57 | 114.58 |
|   verify_gpu (the nine cards) | 109.18 | 109.09 | 108.07 | 110.08 |
|   verify_rest (crossings, rows back, acceptance, server) | 4.53 | 4.53 | 4.46 | 4.58 |
| pass 0 (catch-up): lead | 0.70 | 0.70 | 0.68 | 0.72 |
| pass 0: gpu | 4.33 | 4.32 | 4.24 | 4.37 |
| pass 0: turn | 0.34 | 0.30 | 0.29 | 0.31 |
| pass 1 (draft 1): lead | 0.66 | 0.66 | 0.66 | 0.68 |
| pass 1: gpu | 3.48 | 3.05 | 2.99 | 6.70 |
| pass 1: turn | 0.53 | 0.53 | 0.52 | 0.55 |
| pass 2 (draft 2): lead | 0.66 | 0.65 | 0.64 | 0.68 |
| pass 2: gpu | 3.51 | 3.00 | 2.93 | 6.65 |
| pass 2: turn | 0.53 | 0.52 | 0.48 | 0.57 |
| pass 3 (draft 3): lead | 0.66 | 0.66 | 0.65 | 0.68 |
| pass 3: gpu | 3.40 | 3.00 | 2.95 | 6.63 |
| pass 3: turn (the server before the next verify) | 12.69 | 7.37 | 6.24 | 8.70 |
| the draft's span (all passes) | 31.51 | 24.81 | 23.35 | 37.21 |
|   the draft's GPU time on its last card | 14.72 | 13.36 | 13.13 | 24.33 |
|   the leads (crossings) | 2.69 | 2.69 | 2.66 | 2.72 |
|   the turns of the passes before the last | 1.41 | 1.35 | 1.31 | 1.41 |
|   the draft's GPU time on the embedding card | 0.08 | 0.08 | 0.08 | 0.09 |
| the host's launch of the main pass (inside verify_span) | 5.23 | 4.66 | 4.60 | 4.83 |

verify_gpu per card, median ms: 13.20, 12.42, 11.93, 12.67, 10.30, 11.99, 10.81, 12.05, 13.68

