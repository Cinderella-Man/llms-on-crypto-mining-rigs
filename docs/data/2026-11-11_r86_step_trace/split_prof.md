prof: 3181 evaluations in 5 pieces (cut at gaps over 2000 ms): 9 evaluations and 1 main passes, 2 evaluations and 0 main passes, 1478 evaluations and 88 main passes, 90 evaluations and 6 main passes, 1602 evaluations and 94 main passes
evaluations between main passes, by card: card 4: 714, card 8: 714; the draft's last card: 8

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps |
|---|---:|---:|---:|---|---:|
| greedy | 88 | 3 | 84 | 4: 84 | 81 |
| long | 94 | 1 | 92 | 4: 92 | 89 |

**prof, all requests: 170 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 150.60 | 137.88 | 127.51 | 247.33 |
| verify_span | 110.15 | 112.49 | 105.27 | 114.36 |
|   verify_gpu (the nine cards) | 105.76 | 108.02 | 101.03 | 109.81 |
|   verify_rest (crossings, rows back, acceptance, server) | 4.39 | 4.47 | 4.21 | 4.54 |
| pass 0 (catch-up): lead | 0.65 | 0.68 | 0.59 | 0.71 |
| pass 0: gpu | 4.03 | 4.21 | 3.66 | 4.37 |
| pass 0: turn | 0.39 | 0.32 | 0.31 | 0.33 |
| pass 1 (draft 1): lead | 0.66 | 0.66 | 0.58 | 0.67 |
| pass 1: gpu | 3.29 | 2.97 | 2.53 | 6.28 |
| pass 1: turn | 1.20 | 0.53 | 0.51 | 0.55 |
| pass 2 (draft 2): lead | 0.64 | 0.64 | 0.57 | 0.67 |
| pass 2: gpu | 4.01 | 2.98 | 2.53 | 6.28 |
| pass 2: turn | 0.53 | 0.52 | 0.47 | 0.56 |
| pass 3 (draft 3): lead | 0.66 | 0.65 | 0.57 | 0.67 |
| pass 3: gpu | 3.22 | 2.97 | 2.53 | 6.26 |
| pass 3: turn (the server before the next verify) | 21.17 | 7.33 | 5.98 | 114.00 |
| the draft's span (all passes) | 40.45 | 24.42 | 21.89 | 134.32 |
|   the draft's GPU time on its last card | 14.55 | 13.24 | 11.28 | 22.60 |
|   the leads (crossings) | 2.61 | 2.66 | 2.32 | 2.72 |
|   the turns of the passes before the last | 2.12 | 1.37 | 1.30 | 1.48 |
|   the draft's GPU time on the embedding card | 0.11 | 0.08 | 0.08 | 0.09 |
| the host's launch of the main pass (inside verify_span) | 5.43 | 4.63 | 3.82 | 6.63 |

verify_gpu per card, median ms: 13.02, 12.21, 11.77, 12.53, 10.17, 11.86, 10.60, 11.88, 13.54

**prof, greedy: 81 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 130.75 | 128.92 | 126.76 | 140.51 |
| verify_span | 106.11 | 105.96 | 104.82 | 107.17 |
|   verify_gpu (the nine cards) | 101.86 | 101.69 | 100.55 | 102.96 |
|   verify_rest (crossings, rows back, acceptance, server) | 4.25 | 4.25 | 4.19 | 4.30 |
| pass 0 (catch-up): lead | 0.60 | 0.60 | 0.59 | 0.62 |
| pass 0: gpu | 3.72 | 3.70 | 3.65 | 3.76 |
| pass 0: turn | 0.36 | 0.31 | 0.30 | 0.33 |
| pass 1 (draft 1): lead | 0.63 | 0.59 | 0.58 | 0.60 |
| pass 1: gpu | 3.29 | 2.56 | 2.52 | 6.37 |
| pass 1: turn | 0.52 | 0.52 | 0.50 | 0.54 |
| pass 2 (draft 2): lead | 0.58 | 0.57 | 0.57 | 0.59 |
| pass 2: gpu | 3.41 | 2.56 | 2.50 | 6.33 |
| pass 2: turn | 0.51 | 0.49 | 0.47 | 0.57 |
| pass 3 (draft 3): lead | 0.58 | 0.57 | 0.57 | 0.59 |
| pass 3: gpu | 3.20 | 2.56 | 2.50 | 6.29 |
| pass 3: turn (the server before the next verify) | 7.24 | 7.25 | 5.98 | 8.28 |
| the draft's span (all passes) | 24.63 | 22.47 | 21.29 | 34.79 |
|   the draft's GPU time on its last card | 13.61 | 11.40 | 11.24 | 22.70 |
|   the leads (crossings) | 2.39 | 2.34 | 2.32 | 2.38 |
|   the turns of the passes before the last | 1.40 | 1.31 | 1.28 | 1.48 |
|   the draft's GPU time on the embedding card | 0.09 | 0.08 | 0.08 | 0.09 |
| the host's launch of the main pass (inside verify_span) | 4.43 | 3.89 | 3.79 | 4.00 |

verify_gpu per card, median ms: 12.52, 11.16, 11.36, 11.50, 9.70, 11.42, 9.57, 11.44, 12.97

**prof, long: 89 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 168.68 | 139.17 | 137.34 | 251.12 |
| verify_span | 113.82 | 113.68 | 112.71 | 114.65 |
|   verify_gpu (the nine cards) | 109.30 | 109.18 | 108.14 | 110.14 |
|   verify_rest (crossings, rows back, acceptance, server) | 4.52 | 4.51 | 4.47 | 4.56 |
| pass 0 (catch-up): lead | 0.70 | 0.70 | 0.68 | 0.72 |
| pass 0: gpu | 4.32 | 4.31 | 4.24 | 4.39 |
| pass 0: turn | 0.41 | 0.32 | 0.32 | 0.33 |
| pass 1 (draft 1): lead | 0.69 | 0.66 | 0.66 | 0.68 |
| pass 1: gpu | 3.29 | 3.03 | 2.95 | 3.39 |
| pass 1: turn | 1.82 | 0.53 | 0.52 | 0.55 |
| pass 2 (draft 2): lead | 0.69 | 0.65 | 0.65 | 0.68 |
| pass 2: gpu | 4.56 | 3.00 | 2.95 | 3.48 |
| pass 2: turn | 0.54 | 0.53 | 0.50 | 0.56 |
| pass 3 (draft 3): lead | 0.73 | 0.66 | 0.65 | 0.68 |
| pass 3: gpu | 3.24 | 3.00 | 2.94 | 3.03 |
| pass 3: turn (the server before the next verify) | 33.86 | 7.48 | 6.04 | 117.99 |
| the draft's span (all passes) | 54.85 | 25.16 | 23.81 | 137.83 |
|   the draft's GPU time on its last card | 15.41 | 13.32 | 13.18 | 14.62 |
|   the leads (crossings) | 2.81 | 2.68 | 2.66 | 2.76 |
|   the turns of the passes before the last | 2.77 | 1.39 | 1.36 | 1.47 |
|   the draft's GPU time on the embedding card | 0.12 | 0.08 | 0.08 | 0.10 |
| the host's launch of the main pass (inside verify_span) | 6.34 | 4.76 | 4.64 | 7.44 |

verify_gpu per card, median ms: 13.23, 12.42, 11.94, 12.71, 10.31, 11.98, 10.86, 12.08, 13.69

