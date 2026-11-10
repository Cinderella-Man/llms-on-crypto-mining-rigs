d1off: 5232 evaluations in 6 pieces (cut at gaps over 2000 ms): 9 evaluations and 1 main passes, 2 evaluations and 0 main passes, 90 evaluations and 6 main passes, 1594 evaluations and 94 main passes, 1478 evaluations and 88 main passes, 2059 evaluations and 121 main passes
evaluations between main passes, by card: card 4: 1191, card 8: 1191; the draft's last card: 8

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps |
|---|---:|---:|---:|---|---:|
| long | 94 | 1 | 92 | 1: 1, 4: 91 | 88 |
| greedy | 88 | 2 | 85 | 4: 85 | 82 |
| smoke | 121 | 0 | 120 | 4: 120 | 117 |

**d1off, all requests: 287 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 134.92 | 131.73 | 128.51 | 142.59 |
| verify_span | 109.43 | 107.57 | 105.98 | 114.79 |
|   verify_gpu (the nine cards) | 104.37 | 102.61 | 101.00 | 109.55 |
|   verify_rest (crossings, rows back, acceptance, server) | 5.05 | 5.00 | 4.92 | 5.21 |
| pass 0 (catch-up): lead | 1.50 | 1.48 | 1.46 | 1.57 |
| pass 0: gpu | 3.85 | 3.68 | 3.60 | 4.29 |
| pass 0: turn | 0.32 | 0.28 | 0.27 | 0.30 |
| pass 1 (draft 1): lead | 0.61 | 0.59 | 0.58 | 0.66 |
| pass 1: gpu | 3.22 | 2.67 | 2.61 | 6.28 |
| pass 1: turn | 0.52 | 0.51 | 0.50 | 0.54 |
| pass 2 (draft 2): lead | 0.60 | 0.58 | 0.57 | 0.65 |
| pass 2: gpu | 3.27 | 2.63 | 2.57 | 6.26 |
| pass 2: turn | 0.51 | 0.50 | 0.47 | 0.54 |
| pass 3 (draft 3): lead | 0.62 | 0.59 | 0.57 | 0.66 |
| pass 3: gpu | 3.14 | 2.63 | 2.57 | 6.25 |
| pass 3: turn (the server before the next verify) | 7.33 | 7.53 | 5.21 | 8.50 |
| the draft's span (all passes) | 25.50 | 24.21 | 21.78 | 34.56 |
|   the draft's GPU time on its last card | 13.48 | 11.59 | 11.38 | 22.47 |
|   the leads (crossings) | 3.33 | 3.23 | 3.18 | 3.54 |
|   the turns of the passes before the last | 1.36 | 1.30 | 1.25 | 1.42 |
|   the draft's GPU time on the embedding card | 0.08 | 0.08 | 0.08 | 0.09 |
| the host's launch of the main pass (inside verify_span) | 4.66 | 3.93 | 3.76 | 4.68 |

verify_gpu per card, median ms: 12.71, 11.33, 11.46, 11.58, 9.80, 11.54, 9.74, 11.56, 13.07

**d1off, long: 88 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 141.95 | 140.46 | 138.73 | 150.40 |
| verify_span | 114.58 | 114.43 | 113.32 | 115.44 |
|   verify_gpu (the nine cards) | 109.37 | 109.23 | 108.10 | 110.23 |
|   verify_rest (crossings, rows back, acceptance, server) | 5.21 | 5.19 | 5.13 | 5.25 |
| pass 0 (catch-up): lead | 1.57 | 1.56 | 1.56 | 1.59 |
| pass 0: gpu | 4.28 | 4.27 | 4.21 | 4.32 |
| pass 0: turn | 0.33 | 0.29 | 0.29 | 0.30 |
| pass 1 (draft 1): lead | 0.66 | 0.66 | 0.65 | 0.67 |
| pass 1: gpu | 3.41 | 3.05 | 2.94 | 3.45 |
| pass 1: turn | 0.53 | 0.52 | 0.51 | 0.55 |
| pass 2 (draft 2): lead | 0.65 | 0.65 | 0.64 | 0.66 |
| pass 2: gpu | 3.44 | 3.01 | 2.90 | 4.25 |
| pass 2: turn | 0.53 | 0.52 | 0.50 | 0.57 |
| pass 3 (draft 3): lead | 0.65 | 0.65 | 0.64 | 0.67 |
| pass 3: gpu | 3.32 | 3.00 | 2.90 | 3.05 |
| pass 3: turn (the server before the next verify) | 8.00 | 7.74 | 6.64 | 8.37 |
| the draft's span (all passes) | 27.37 | 26.06 | 24.99 | 35.52 |
|   the draft's GPU time on its last card | 14.45 | 13.32 | 12.97 | 15.01 |
|   the leads (crossings) | 3.54 | 3.53 | 3.51 | 3.57 |
|   the turns of the passes before the last | 1.39 | 1.34 | 1.31 | 1.44 |
|   the draft's GPU time on the embedding card | 0.08 | 0.08 | 0.08 | 0.09 |
| the host's launch of the main pass (inside verify_span) | 5.34 | 4.63 | 4.58 | 4.79 |

verify_gpu per card, median ms: 13.23, 12.40, 11.92, 12.70, 10.30, 11.98, 10.90, 12.08, 13.68

**d1off, greedy: 82 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 133.18 | 131.57 | 129.09 | 142.82 |
| verify_span | 107.07 | 106.88 | 105.80 | 108.09 |
|   verify_gpu (the nine cards) | 102.10 | 101.91 | 100.85 | 103.13 |
|   verify_rest (crossings, rows back, acceptance, server) | 4.98 | 4.97 | 4.91 | 5.03 |
| pass 0 (catch-up): lead | 1.48 | 1.47 | 1.46 | 1.50 |
| pass 0: gpu | 3.66 | 3.63 | 3.59 | 3.72 |
| pass 0: turn | 0.33 | 0.28 | 0.27 | 0.32 |
| pass 1 (draft 1): lead | 0.58 | 0.58 | 0.58 | 0.59 |
| pass 1: gpu | 3.34 | 2.64 | 2.60 | 6.37 |
| pass 1: turn | 0.52 | 0.51 | 0.50 | 0.54 |
| pass 2 (draft 2): lead | 0.58 | 0.57 | 0.57 | 0.59 |
| pass 2: gpu | 3.43 | 2.58 | 2.55 | 6.32 |
| pass 2: turn | 0.51 | 0.48 | 0.47 | 0.54 |
| pass 3 (draft 3): lead | 0.63 | 0.58 | 0.56 | 0.60 |
| pass 3: gpu | 3.23 | 2.59 | 2.54 | 6.31 |
| pass 3: turn (the server before the next verify) | 7.82 | 7.84 | 6.64 | 8.79 |
| the draft's span (all passes) | 26.11 | 24.11 | 23.00 | 35.92 |
|   the draft's GPU time on its last card | 13.66 | 11.48 | 11.32 | 22.70 |
|   the leads (crossings) | 3.27 | 3.22 | 3.18 | 3.25 |
|   the turns of the passes before the last | 1.36 | 1.28 | 1.25 | 1.43 |
|   the draft's GPU time on the embedding card | 0.08 | 0.08 | 0.08 | 0.09 |
| the host's launch of the main pass (inside verify_span) | 4.39 | 3.79 | 3.69 | 3.93 |

verify_gpu per card, median ms: 12.65, 11.24, 11.38, 11.48, 9.71, 11.41, 9.63, 11.44, 12.95

**d1off, smoke: 117 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 130.86 | 129.91 | 126.62 | 133.53 |
| verify_span | 107.20 | 107.02 | 105.95 | 108.44 |
|   verify_gpu (the nine cards) | 102.21 | 102.08 | 100.99 | 103.25 |
|   verify_rest (crossings, rows back, acceptance, server) | 4.99 | 4.96 | 4.91 | 5.03 |
| pass 0 (catch-up): lead | 1.47 | 1.47 | 1.46 | 1.49 |
| pass 0: gpu | 3.67 | 3.65 | 3.60 | 3.73 |
| pass 0: turn | 0.31 | 0.28 | 0.27 | 0.29 |
| pass 1 (draft 1): lead | 0.59 | 0.58 | 0.58 | 0.60 |
| pass 1: gpu | 3.00 | 2.66 | 2.63 | 3.11 |
| pass 1: turn | 0.52 | 0.51 | 0.50 | 0.53 |
| pass 2 (draft 2): lead | 0.58 | 0.57 | 0.56 | 0.59 |
| pass 2: gpu | 3.03 | 2.61 | 2.59 | 3.88 |
| pass 2: turn | 0.50 | 0.48 | 0.47 | 0.54 |
| pass 3 (draft 3): lead | 0.58 | 0.58 | 0.56 | 0.60 |
| pass 3: gpu | 2.93 | 2.62 | 2.58 | 2.67 |
| pass 3: turn (the server before the next verify) | 6.48 | 6.84 | 2.47 | 8.10 |
| the draft's span (all passes) | 23.65 | 23.09 | 18.58 | 25.41 |
|   the draft's GPU time on its last card | 12.64 | 11.54 | 11.46 | 13.25 |
|   the leads (crossings) | 3.21 | 3.21 | 3.17 | 3.25 |
|   the turns of the passes before the last | 1.33 | 1.27 | 1.25 | 1.36 |
|   the draft's GPU time on the embedding card | 0.08 | 0.08 | 0.08 | 0.09 |
| the host's launch of the main pass (inside verify_span) | 4.33 | 3.91 | 3.80 | 4.04 |

verify_gpu per card, median ms: 12.61, 11.24, 11.38, 11.50, 9.71, 11.47, 9.68, 11.50, 13.05

