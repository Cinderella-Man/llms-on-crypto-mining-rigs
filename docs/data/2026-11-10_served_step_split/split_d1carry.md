d1carry: 3725 evaluations in 6 pieces (cut at gaps over 2000 ms): 9 evaluations and 1 main passes, 2 evaluations and 0 main passes, 90 evaluations and 6 main passes, 308 evaluations and 18 main passes, 1495 evaluations and 89 main passes, 1821 evaluations and 107 main passes
evaluations between main passes, by card: card 4: 838, card 8: 838; the draft's last card: 8

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps |
|---|---:|---:|---:|---|---:|
| long | 18 | 1 | 16 | 4: 16 | 13 |
| greedy | 89 | 2 | 86 | 4: 86 | 83 |
| smoke | 107 | 0 | 106 | 4: 106 | 103 |

**d1carry, all requests: 199 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 132.04 | 130.06 | 126.58 | 142.37 |
| verify_span | 107.49 | 106.93 | 105.89 | 109.08 |
|   verify_gpu (the nine cards) | 102.51 | 101.99 | 100.95 | 104.08 |
|   verify_rest (crossings, rows back, acceptance, server) | 4.97 | 4.96 | 4.89 | 5.07 |
| pass 0 (catch-up): lead | 1.48 | 1.47 | 1.46 | 1.50 |
| pass 0: gpu | 3.70 | 3.65 | 3.61 | 3.73 |
| pass 0: turn | 0.45 | 0.28 | 0.28 | 0.30 |
| pass 1 (draft 1): lead | 0.61 | 0.60 | 0.59 | 0.62 |
| pass 1: gpu | 3.33 | 2.67 | 2.59 | 6.76 |
| pass 1: turn | 0.94 | 0.92 | 0.90 | 0.97 |
| pass 2 (draft 2): lead | 0.38 | 0.38 | 0.37 | 0.40 |
| pass 2: gpu | 3.21 | 2.68 | 2.64 | 6.32 |
| pass 2: turn | 0.30 | 0.29 | 0.28 | 0.32 |
| pass 3 (draft 3): lead | 0.38 | 0.37 | 0.36 | 0.40 |
| pass 3: gpu | 3.27 | 2.65 | 2.60 | 6.31 |
| pass 3: turn (the server before the next verify) | 6.51 | 6.93 | 2.20 | 7.93 |
| the draft's span (all passes) | 24.55 | 23.20 | 18.86 | 35.16 |
|   the draft's GPU time on its last card | 13.50 | 11.63 | 11.49 | 23.85 |
|   the leads (crossings) | 2.84 | 2.82 | 2.79 | 2.88 |
|   the turns of the passes before the last | 1.69 | 1.50 | 1.47 | 1.62 |
|   the draft's GPU time on the embedding card | 0.09 | 0.09 | 0.08 | 0.09 |
| the host's launch of the main pass (inside verify_span) | 4.46 | 3.86 | 3.75 | 4.15 |

verify_gpu per card, median ms: 12.63, 11.27, 11.36, 11.52, 9.70, 11.41, 9.66, 11.44, 13.01

**d1carry, long: 13 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 143.19 | 140.60 | 137.73 | 152.38 |
| verify_span | 113.51 | 113.39 | 111.85 | 114.68 |
|   verify_gpu (the nine cards) | 108.33 | 108.26 | 106.61 | 109.57 |
|   verify_rest (crossings, rows back, acceptance, server) | 5.18 | 5.18 | 5.09 | 5.24 |
| pass 0 (catch-up): lead | 1.56 | 1.56 | 1.55 | 1.59 |
| pass 0: gpu | 4.17 | 4.17 | 4.14 | 4.22 |
| pass 0: turn | 2.29 | 0.30 | 0.29 | 1.04 |
| pass 1 (draft 1): lead | 0.68 | 0.67 | 0.67 | 0.69 |
| pass 1: gpu | 3.82 | 3.00 | 2.98 | 7.12 |
| pass 1: turn | 0.94 | 0.93 | 0.92 | 0.97 |
| pass 2 (draft 2): lead | 0.45 | 0.45 | 0.44 | 0.47 |
| pass 2: gpu | 3.63 | 3.00 | 2.99 | 6.68 |
| pass 2: turn | 0.31 | 0.31 | 0.29 | 0.34 |
| pass 3 (draft 3): lead | 0.44 | 0.43 | 0.43 | 0.44 |
| pass 3: gpu | 3.72 | 2.98 | 2.94 | 6.65 |
| pass 3: turn (the server before the next verify) | 7.66 | 7.53 | 6.81 | 8.66 |
| the draft's span (all passes) | 29.69 | 26.48 | 25.14 | 39.11 |
|   the draft's GPU time on its last card | 15.34 | 13.15 | 13.07 | 25.46 |
|   the leads (crossings) | 3.14 | 3.12 | 3.11 | 3.15 |
|   the turns of the passes before the last | 3.55 | 1.55 | 1.53 | 2.30 |
|   the draft's GPU time on the embedding card | 0.09 | 0.09 | 0.08 | 0.09 |
| the host's launch of the main pass (inside verify_span) | 4.60 | 4.58 | 4.51 | 4.65 |

verify_gpu per card, median ms: 13.01, 12.28, 11.88, 12.55, 10.23, 12.01, 10.72, 11.95, 13.56

**d1carry, greedy: 83 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 132.61 | 130.61 | 128.45 | 142.73 |
| verify_span | 106.97 | 106.83 | 105.77 | 107.98 |
|   verify_gpu (the nine cards) | 102.02 | 101.88 | 100.81 | 103.01 |
|   verify_rest (crossings, rows back, acceptance, server) | 4.95 | 4.94 | 4.89 | 5.00 |
| pass 0 (catch-up): lead | 1.47 | 1.47 | 1.46 | 1.49 |
| pass 0: gpu | 3.67 | 3.65 | 3.61 | 3.69 |
| pass 0: turn | 0.34 | 0.28 | 0.28 | 0.30 |
| pass 1 (draft 1): lead | 0.60 | 0.60 | 0.59 | 0.61 |
| pass 1: gpu | 3.55 | 2.68 | 2.59 | 6.80 |
| pass 1: turn | 0.96 | 0.92 | 0.90 | 0.98 |
| pass 2 (draft 2): lead | 0.38 | 0.38 | 0.37 | 0.39 |
| pass 2: gpu | 3.36 | 2.67 | 2.60 | 6.37 |
| pass 2: turn | 0.30 | 0.29 | 0.28 | 0.32 |
| pass 3 (draft 3): lead | 0.37 | 0.37 | 0.36 | 0.38 |
| pass 3: gpu | 3.47 | 2.65 | 2.60 | 6.34 |
| pass 3: turn (the server before the next verify) | 7.18 | 7.20 | 5.99 | 8.01 |
| the draft's span (all passes) | 25.65 | 23.47 | 22.11 | 36.08 |
|   the draft's GPU time on its last card | 14.04 | 11.65 | 11.47 | 24.09 |
|   the leads (crossings) | 2.82 | 2.82 | 2.79 | 2.86 |
|   the turns of the passes before the last | 1.60 | 1.51 | 1.47 | 2.11 |
|   the draft's GPU time on the embedding card | 0.09 | 0.09 | 0.08 | 0.10 |
| the host's launch of the main pass (inside verify_span) | 4.38 | 3.82 | 3.71 | 3.96 |

verify_gpu per card, median ms: 12.60, 11.25, 11.37, 11.52, 9.71, 11.40, 9.62, 11.41, 12.96

**d1carry, smoke: 103 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 130.17 | 129.06 | 126.18 | 136.26 |
| verify_span | 107.14 | 106.80 | 105.96 | 108.39 |
|   verify_gpu (the nine cards) | 102.17 | 101.84 | 100.98 | 103.38 |
|   verify_rest (crossings, rows back, acceptance, server) | 4.97 | 4.96 | 4.90 | 5.04 |
| pass 0 (catch-up): lead | 1.47 | 1.47 | 1.46 | 1.49 |
| pass 0: gpu | 3.66 | 3.64 | 3.60 | 3.70 |
| pass 0: turn | 0.31 | 0.28 | 0.28 | 0.30 |
| pass 1 (draft 1): lead | 0.60 | 0.60 | 0.60 | 0.61 |
| pass 1: gpu | 3.10 | 2.66 | 2.59 | 3.96 |
| pass 1: turn | 0.93 | 0.92 | 0.90 | 0.95 |
| pass 2 (draft 2): lead | 0.38 | 0.38 | 0.37 | 0.39 |
| pass 2: gpu | 3.03 | 2.68 | 2.64 | 3.15 |
| pass 2: turn | 0.30 | 0.29 | 0.28 | 0.32 |
| pass 3 (draft 3): lead | 0.37 | 0.37 | 0.36 | 0.39 |
| pass 3: gpu | 3.05 | 2.64 | 2.60 | 3.82 |
| pass 3: turn (the server before the next verify) | 5.82 | 6.32 | 2.14 | 7.76 |
| the draft's span (all passes) | 23.03 | 22.37 | 18.19 | 29.26 |
|   the draft's GPU time on its last card | 12.84 | 11.61 | 11.50 | 13.69 |
|   the leads (crossings) | 2.83 | 2.83 | 2.79 | 2.86 |
|   the turns of the passes before the last | 1.53 | 1.49 | 1.47 | 1.60 |
|   the draft's GPU time on the embedding card | 0.09 | 0.09 | 0.08 | 0.09 |
| the host's launch of the main pass (inside verify_span) | 4.51 | 3.88 | 3.79 | 4.03 |

verify_gpu per card, median ms: 12.66, 11.25, 11.35, 11.50, 9.68, 11.39, 9.66, 11.42, 13.03

