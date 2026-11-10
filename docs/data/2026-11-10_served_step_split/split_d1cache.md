d1cache: 3934 evaluations in 8 pieces (cut at gaps over 2000 ms): 9 evaluations and 1 main passes, 1 evaluations and 0 main passes, 81 evaluations and 5 main passes, 9 evaluations and 1 main passes, 8 evaluations and 0 main passes, 1193 evaluations and 92 main passes, 1137 evaluations and 88 main passes, 1496 evaluations and 115 main passes
evaluations between main passes, by card: card 4: 2, card 8: 1162; the draft's last card: 8

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps |
|---|---:|---:|---:|---|---:|
| long | 92 | 1 | 90 | 4: 90 | 87 |
| greedy | 88 | 2 | 85 | 4: 85 | 82 |
| smoke | 115 | 0 | 114 | 4: 114 | 111 |

**d1cache, all requests: 280 typical steps, 4 draft passes a step, 4 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 153.19 | 130.89 | 125.76 | 253.88 |
| verify_span | 110.67 | 108.86 | 107.11 | 116.21 |
|   verify_gpu (the nine cards) | 104.51 | 102.79 | 101.08 | 109.85 |
|   verify_rest (crossings, rows back, acceptance, server) | 6.16 | 6.06 | 5.96 | 6.43 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 4.28 | 3.72 | 3.63 | 4.29 |
| pass 0: turn | 11.81 | 0.75 | 0.72 | 2.15 |
| pass 1 (draft 1): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 1: gpu | 3.19 | 2.71 | 2.64 | 3.66 |
| pass 1: turn | 1.05 | 0.96 | 0.93 | 1.13 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 3.77 | 2.67 | 2.60 | 6.30 |
| pass 2: turn | 1.03 | 0.96 | 0.89 | 1.40 |
| pass 3 (draft 3): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 3: gpu | 3.11 | 2.66 | 2.60 | 3.63 |
| pass 3: turn (the server before the next verify) | 14.28 | 6.89 | 2.41 | 9.46 |
| the draft's span (all passes) | 42.52 | 22.45 | 16.73 | 137.73 |
|   the draft's GPU time on its last card | 14.36 | 11.76 | 11.50 | 22.47 |
|   the leads (crossings) | 0.00 | 0.00 | 0.00 | 0.00 |
|   the turns of the passes before the last | 13.88 | 2.77 | 2.56 | 4.69 |
|   the draft's GPU time on the embedding card | 0.00 | 0.00 | 0.00 | 0.00 |
| the host's launch of the main pass (inside verify_span) | 5.29 | 3.97 | 3.78 | 7.76 |

verify_gpu per card, median ms: 12.73, 11.38, 11.49, 11.59, 9.84, 11.54, 9.75, 11.57, 13.06

**d1cache, long: 87 typical steps, 4 draft passes a step, 4 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 204.02 | 248.04 | 137.52 | 266.22 |
| verify_span | 115.86 | 115.75 | 114.57 | 116.91 |
|   verify_gpu (the nine cards) | 109.43 | 109.36 | 108.19 | 110.48 |
|   verify_rest (crossings, rows back, acceptance, server) | 6.43 | 6.37 | 6.28 | 6.49 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 5.54 | 4.26 | 4.20 | 4.30 |
| pass 0: turn | 35.98 | 1.34 | 0.81 | 114.62 |
| pass 1 (draft 1): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 1: gpu | 3.35 | 3.05 | 3.01 | 3.53 |
| pass 1: turn | 1.24 | 1.10 | 1.02 | 1.61 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 4.80 | 3.00 | 2.97 | 3.70 |
| pass 2: turn | 1.19 | 1.06 | 0.99 | 1.52 |
| pass 3 (draft 3): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 3: gpu | 3.28 | 3.00 | 2.97 | 3.03 |
| pass 3: turn (the server before the next verify) | 32.78 | 7.90 | 5.32 | 119.52 |
| the draft's span (all passes) | 88.16 | 131.98 | 22.09 | 150.42 |
|   the draft's GPU time on its last card | 16.96 | 13.31 | 13.22 | 16.38 |
|   the leads (crossings) | 0.00 | 0.00 | 0.00 | 0.00 |
|   the turns of the passes before the last | 38.42 | 3.39 | 2.87 | 117.62 |
|   the draft's GPU time on the embedding card | 0.00 | 0.00 | 0.00 | 0.00 |
| the host's launch of the main pass (inside verify_span) | 6.90 | 6.02 | 4.62 | 8.19 |

verify_gpu per card, median ms: 13.27, 12.46, 11.96, 12.70, 10.32, 12.01, 10.93, 12.07, 13.66

**d1cache, greedy: 82 typical steps, 4 draft passes a step, 4 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 132.27 | 130.74 | 127.90 | 141.85 |
| verify_span | 108.17 | 108.04 | 106.96 | 109.10 |
|   verify_gpu (the nine cards) | 102.14 | 102.01 | 100.93 | 103.09 |
|   verify_rest (crossings, rows back, acceptance, server) | 6.03 | 6.02 | 5.95 | 6.13 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 3.70 | 3.69 | 3.64 | 3.73 |
| pass 0: turn | 1.03 | 0.74 | 0.72 | 1.56 |
| pass 1 (draft 1): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 1: gpu | 3.36 | 2.67 | 2.60 | 6.34 |
| pass 1: turn | 0.97 | 0.95 | 0.93 | 1.02 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 3.71 | 2.63 | 2.57 | 6.85 |
| pass 2: turn | 0.95 | 0.94 | 0.89 | 1.01 |
| pass 3 (draft 3): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 3: gpu | 3.25 | 2.63 | 2.57 | 6.30 |
| pass 3: turn (the server before the next verify) | 7.12 | 7.40 | 5.17 | 8.81 |
| the draft's span (all passes) | 24.10 | 22.55 | 19.94 | 33.98 |
|   the draft's GPU time on its last card | 14.03 | 11.61 | 11.47 | 22.67 |
|   the leads (crossings) | 0.00 | 0.00 | 0.00 | 0.00 |
|   the turns of the passes before the last | 2.95 | 2.71 | 2.56 | 3.49 |
|   the draft's GPU time on the embedding card | 0.00 | 0.00 | 0.00 | 0.00 |
| the host's launch of the main pass (inside verify_span) | 4.58 | 3.87 | 3.76 | 4.01 |

verify_gpu per card, median ms: 12.66, 11.24, 11.38, 11.46, 9.76, 11.43, 9.62, 11.44, 12.97

**d1cache, smoke: 111 typical steps, 4 draft passes a step, 4 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 128.82 | 127.75 | 124.72 | 133.87 |
| verify_span | 108.46 | 108.28 | 106.81 | 109.84 |
|   verify_gpu (the nine cards) | 102.41 | 102.33 | 100.83 | 103.83 |
|   verify_rest (crossings, rows back, acceptance, server) | 6.05 | 6.01 | 5.96 | 6.11 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 3.72 | 3.69 | 3.61 | 3.77 |
| pass 0: turn | 0.83 | 0.73 | 0.72 | 1.23 |
| pass 1 (draft 1): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 1: gpu | 2.94 | 2.68 | 2.64 | 2.74 |
| pass 1: turn | 0.95 | 0.94 | 0.93 | 0.97 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 3.02 | 2.64 | 2.60 | 2.68 |
| pass 2: turn | 0.95 | 0.91 | 0.89 | 0.97 |
| pass 3 (draft 3): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 3: gpu | 2.88 | 2.64 | 2.60 | 2.70 |
| pass 3: turn (the server before the next verify) | 5.08 | 5.41 | 2.39 | 7.96 |
| the draft's span (all passes) | 20.36 | 19.86 | 16.61 | 23.25 |
|   the draft's GPU time on its last card | 12.56 | 11.67 | 11.50 | 11.96 |
|   the leads (crossings) | 0.00 | 0.00 | 0.00 | 0.00 |
|   the turns of the passes before the last | 2.73 | 2.59 | 2.55 | 3.12 |
|   the draft's GPU time on the embedding card | 0.00 | 0.00 | 0.00 | 0.00 |
| the host's launch of the main pass (inside verify_span) | 4.54 | 3.94 | 3.81 | 4.04 |

verify_gpu per card, median ms: 12.64, 11.32, 11.41, 11.52, 9.76, 11.47, 9.66, 11.49, 13.05

