d1off-2: 2756 evaluations in 6 pieces (cut at gaps over 2000 ms): 9 evaluations and 1 main passes, 2 evaluations and 0 main passes, 90 evaluations and 6 main passes, 342 evaluations and 20 main passes, 1478 evaluations and 88 main passes, 835 evaluations and 49 main passes
evaluations between main passes, by card: card 4: 610, card 8: 610; the draft's last card: 8

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps |
|---|---:|---:|---:|---|---:|
| long | 20 | 1 | 18 | 4: 18 | 15 |
| greedy | 88 | 2 | 85 | 4: 85 | 82 |
| smoke | 49 | 0 | 48 | 4: 48 | 45 |

**d1off-2, all requests: 142 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 132.22 | 130.28 | 126.88 | 141.60 |
| verify_span | 107.62 | 106.89 | 105.75 | 112.38 |
|   verify_gpu (the nine cards) | 102.60 | 101.92 | 100.76 | 107.13 |
|   verify_rest (crossings, rows back, acceptance, server) | 5.02 | 4.96 | 4.90 | 5.12 |
| pass 0 (catch-up): lead | 1.48 | 1.47 | 1.46 | 1.55 |
| pass 0: gpu | 3.69 | 3.62 | 3.57 | 4.17 |
| pass 0: turn | 0.34 | 0.30 | 0.30 | 0.32 |
| pass 1 (draft 1): lead | 0.59 | 0.58 | 0.57 | 0.65 |
| pass 1: gpu | 3.13 | 2.62 | 2.59 | 6.28 |
| pass 1: turn | 0.51 | 0.51 | 0.49 | 0.53 |
| pass 2 (draft 2): lead | 0.58 | 0.57 | 0.57 | 0.64 |
| pass 2: gpu | 3.18 | 2.60 | 2.54 | 6.22 |
| pass 2: turn | 0.52 | 0.48 | 0.46 | 0.54 |
| pass 3 (draft 3): lead | 0.59 | 0.58 | 0.56 | 0.64 |
| pass 3: gpu | 3.04 | 2.59 | 2.55 | 6.22 |
| pass 3: turn (the server before the next verify) | 6.96 | 7.40 | 3.36 | 8.16 |
| the draft's span (all passes) | 24.60 | 23.48 | 19.33 | 34.25 |
|   the draft's GPU time on its last card | 13.03 | 11.43 | 11.30 | 22.35 |
|   the leads (crossings) | 3.24 | 3.21 | 3.17 | 3.49 |
|   the turns of the passes before the last | 1.37 | 1.29 | 1.26 | 1.40 |
|   the draft's GPU time on the embedding card | 0.08 | 0.08 | 0.08 | 0.09 |
| the host's launch of the main pass (inside verify_span) | 4.40 | 3.86 | 3.75 | 4.59 |

verify_gpu per card, median ms: 12.65, 11.19, 11.38, 11.47, 9.71, 11.47, 9.64, 11.47, 12.96

**d1off-2, long: 15 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 141.04 | 139.26 | 138.35 | 150.11 |
| verify_span | 114.10 | 113.41 | 112.38 | 115.88 |
|   verify_gpu (the nine cards) | 108.90 | 108.13 | 107.13 | 110.65 |
|   verify_rest (crossings, rows back, acceptance, server) | 5.20 | 5.21 | 5.12 | 5.28 |
| pass 0 (catch-up): lead | 1.56 | 1.56 | 1.55 | 1.58 |
| pass 0: gpu | 4.23 | 4.22 | 4.17 | 4.26 |
| pass 0: turn | 0.32 | 0.32 | 0.31 | 0.33 |
| pass 1 (draft 1): lead | 0.66 | 0.66 | 0.65 | 0.66 |
| pass 1: gpu | 2.99 | 2.92 | 2.91 | 3.00 |
| pass 1: turn | 0.56 | 0.52 | 0.51 | 0.54 |
| pass 2 (draft 2): lead | 0.65 | 0.64 | 0.64 | 0.65 |
| pass 2: gpu | 2.93 | 2.89 | 2.88 | 2.97 |
| pass 2: turn | 0.56 | 0.51 | 0.49 | 0.58 |
| pass 3 (draft 3): lead | 0.65 | 0.65 | 0.64 | 0.65 |
| pass 3: gpu | 2.93 | 2.89 | 2.88 | 2.94 |
| pass 3: turn (the server before the next verify) | 8.92 | 7.91 | 7.49 | 11.11 |
| the draft's span (all passes) | 26.94 | 25.71 | 25.19 | 28.86 |
|   the draft's GPU time on its last card | 13.07 | 12.93 | 12.85 | 14.02 |
|   the leads (crossings) | 3.51 | 3.51 | 3.49 | 3.53 |
|   the turns of the passes before the last | 1.44 | 1.35 | 1.31 | 1.42 |
|   the draft's GPU time on the embedding card | 0.09 | 0.08 | 0.08 | 0.09 |
| the host's launch of the main pass (inside verify_span) | 5.79 | 4.60 | 4.50 | 4.73 |

verify_gpu per card, median ms: 12.99, 12.20, 11.95, 12.56, 10.23, 11.96, 10.78, 12.06, 13.57

**d1off-2, greedy: 82 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 132.50 | 130.51 | 128.42 | 142.13 |
| verify_span | 107.03 | 106.87 | 105.75 | 108.48 |
|   verify_gpu (the nine cards) | 102.02 | 101.91 | 100.82 | 103.00 |
|   verify_rest (crossings, rows back, acceptance, server) | 5.01 | 4.94 | 4.89 | 5.03 |
| pass 0 (catch-up): lead | 1.47 | 1.47 | 1.46 | 1.49 |
| pass 0: gpu | 3.64 | 3.62 | 3.58 | 3.69 |
| pass 0: turn | 0.35 | 0.30 | 0.30 | 0.32 |
| pass 1 (draft 1): lead | 0.58 | 0.58 | 0.58 | 0.59 |
| pass 1: gpu | 3.32 | 2.61 | 2.59 | 6.35 |
| pass 1: turn | 0.51 | 0.51 | 0.49 | 0.53 |
| pass 2 (draft 2): lead | 0.58 | 0.57 | 0.57 | 0.58 |
| pass 2: gpu | 3.42 | 2.57 | 2.54 | 6.29 |
| pass 2: turn | 0.53 | 0.48 | 0.46 | 0.54 |
| pass 3 (draft 3): lead | 0.58 | 0.58 | 0.56 | 0.60 |
| pass 3: gpu | 3.21 | 2.58 | 2.54 | 6.25 |
| pass 3: turn (the server before the next verify) | 7.27 | 7.48 | 6.27 | 8.03 |
| the draft's span (all passes) | 25.47 | 23.53 | 22.36 | 35.02 |
|   the draft's GPU time on its last card | 13.59 | 11.40 | 11.29 | 22.54 |
|   the leads (crossings) | 3.21 | 3.21 | 3.18 | 3.24 |
|   the turns of the passes before the last | 1.39 | 1.29 | 1.26 | 1.40 |
|   the draft's GPU time on the embedding card | 0.08 | 0.08 | 0.08 | 0.09 |
| the host's launch of the main pass (inside verify_span) | 4.39 | 3.82 | 3.72 | 3.96 |

verify_gpu per card, median ms: 12.66, 11.20, 11.38, 11.47, 9.71, 11.42, 9.62, 11.43, 12.91

**d1off-2, smoke: 45 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 128.78 | 127.75 | 125.09 | 132.01 |
| verify_span | 106.53 | 106.55 | 105.28 | 107.77 |
|   verify_gpu (the nine cards) | 101.56 | 101.54 | 100.29 | 102.76 |
|   verify_rest (crossings, rows back, acceptance, server) | 4.97 | 4.97 | 4.91 | 5.05 |
| pass 0 (catch-up): lead | 1.47 | 1.47 | 1.46 | 1.49 |
| pass 0: gpu | 3.61 | 3.60 | 3.56 | 3.65 |
| pass 0: turn | 0.32 | 0.30 | 0.30 | 0.31 |
| pass 1 (draft 1): lead | 0.58 | 0.58 | 0.57 | 0.59 |
| pass 1: gpu | 2.81 | 2.61 | 2.60 | 2.71 |
| pass 1: turn | 0.51 | 0.50 | 0.49 | 0.53 |
| pass 2 (draft 2): lead | 0.57 | 0.57 | 0.56 | 0.58 |
| pass 2: gpu | 2.82 | 2.61 | 2.56 | 2.66 |
| pass 2: turn | 0.48 | 0.47 | 0.46 | 0.51 |
| pass 3 (draft 3): lead | 0.58 | 0.57 | 0.56 | 0.60 |
| pass 3: gpu | 2.76 | 2.60 | 2.56 | 2.64 |
| pass 3: turn (the server before the next verify) | 5.74 | 6.31 | 2.36 | 7.90 |
| the draft's span (all passes) | 22.25 | 22.11 | 18.25 | 24.51 |
|   the draft's GPU time on its last card | 12.01 | 11.42 | 11.35 | 11.65 |
|   the leads (crossings) | 3.20 | 3.19 | 3.17 | 3.24 |
|   the turns of the passes before the last | 1.30 | 1.28 | 1.26 | 1.33 |
|   the draft's GPU time on the embedding card | 0.08 | 0.08 | 0.08 | 0.08 |
| the host's launch of the main pass (inside verify_span) | 3.96 | 3.89 | 3.78 | 4.00 |

verify_gpu per card, median ms: 12.56, 11.12, 11.28, 11.37, 9.68, 11.44, 9.60, 11.48, 12.98

