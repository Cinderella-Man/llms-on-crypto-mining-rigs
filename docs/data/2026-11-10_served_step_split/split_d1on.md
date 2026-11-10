d1on: 3908 evaluations in 6 pieces (cut at gaps over 2000 ms): 9 evaluations and 1 main passes, 1 evaluations and 0 main passes, 90 evaluations and 6 main passes, 421 evaluations and 32 main passes, 1137 evaluations and 88 main passes, 2250 evaluations and 173 main passes
evaluations between main passes, by card: card 4: 2, card 8: 1154; the draft's last card: 8

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps |
|---|---:|---:|---:|---|---:|
| long | 32 | 2 | 29 | 4: 29 | 26 |
| greedy | 88 | 2 | 85 | 4: 85 | 82 |
| smoke | 173 | 0 | 172 | 4: 172 | 169 |

**d1on, all requests: 277 typical steps, 4 draft passes a step, 4 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 136.97 | 129.32 | 125.62 | 142.07 |
| verify_span | 109.00 | 108.27 | 107.01 | 114.09 |
|   verify_gpu (the nine cards) | 102.93 | 102.24 | 100.98 | 107.83 |
|   verify_rest (crossings, rows back, acceptance, server) | 6.07 | 6.04 | 5.96 | 6.24 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 3.74 | 3.67 | 3.61 | 4.16 |
| pass 0: turn | 0.90 | 0.73 | 0.72 | 1.31 |
| pass 1 (draft 1): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 1: gpu | 3.18 | 2.69 | 2.64 | 3.98 |
| pass 1: turn | 1.15 | 1.13 | 1.11 | 1.23 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 3.09 | 2.70 | 2.65 | 3.43 |
| pass 2: turn | 0.56 | 0.52 | 0.50 | 0.62 |
| pass 3 (draft 3): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 3: gpu | 3.10 | 2.66 | 2.61 | 3.90 |
| pass 3: turn (the server before the next verify) | 12.26 | 6.83 | 2.20 | 8.27 |
| the draft's span (all passes) | 27.98 | 21.16 | 16.29 | 34.18 |
|   the draft's GPU time on its last card | 13.11 | 11.70 | 11.56 | 14.64 |
|   the leads (crossings) | 0.00 | 0.00 | 0.00 | 0.00 |
|   the turns of the passes before the last | 2.61 | 2.43 | 2.35 | 3.08 |
|   the draft's GPU time on the embedding card | 0.00 | 0.00 | 0.00 | 0.00 |
| the host's launch of the main pass (inside verify_span) | 4.77 | 3.88 | 3.75 | 4.67 |

verify_gpu per card, median ms: 12.67, 11.31, 11.39, 11.51, 9.78, 11.45, 9.67, 11.46, 13.03

**d1on, long: 26 typical steps, 4 draft passes a step, 4 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 204.66 | 246.04 | 138.23 | 260.25 |
| verify_span | 115.76 | 115.29 | 114.09 | 117.00 |
|   verify_gpu (the nine cards) | 109.40 | 108.95 | 107.83 | 110.69 |
|   verify_rest (crossings, rows back, acceptance, server) | 6.35 | 6.34 | 6.25 | 6.43 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 4.30 | 4.27 | 4.20 | 4.30 |
| pass 0: turn | 1.22 | 1.31 | 0.80 | 1.34 |
| pass 1 (draft 1): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 1: gpu | 3.56 | 3.05 | 2.97 | 4.29 |
| pass 1: turn | 1.26 | 1.26 | 1.19 | 1.31 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 3.41 | 3.04 | 2.98 | 3.52 |
| pass 2: turn | 0.68 | 0.64 | 0.59 | 0.70 |
| pass 3 (draft 3): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 3: gpu | 3.43 | 2.99 | 2.94 | 4.25 |
| pass 3: turn (the server before the next verify) | 71.05 | 114.84 | 5.28 | 126.17 |
| the draft's span (all passes) | 88.91 | 131.86 | 21.23 | 142.59 |
|   the draft's GPU time on its last card | 14.70 | 13.36 | 13.21 | 15.45 |
|   the leads (crossings) | 0.00 | 0.00 | 0.00 | 0.00 |
|   the turns of the passes before the last | 3.15 | 3.21 | 2.66 | 3.33 |
|   the draft's GPU time on the embedding card | 0.00 | 0.00 | 0.00 | 0.00 |
| the host's launch of the main pass (inside verify_span) | 8.11 | 5.76 | 4.61 | 8.20 |

verify_gpu per card, median ms: 13.11, 12.39, 12.00, 12.60, 10.31, 12.06, 10.85, 12.09, 13.60

**d1on, greedy: 82 typical steps, 4 draft passes a step, 4 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 132.11 | 130.18 | 127.94 | 142.03 |
| verify_span | 108.13 | 107.98 | 106.82 | 109.18 |
|   verify_gpu (the nine cards) | 102.11 | 101.98 | 100.81 | 103.15 |
|   verify_rest (crossings, rows back, acceptance, server) | 6.02 | 6.02 | 5.94 | 6.09 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 3.68 | 3.65 | 3.60 | 3.76 |
| pass 0: turn | 0.96 | 0.74 | 0.72 | 1.27 |
| pass 1 (draft 1): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 1: gpu | 3.55 | 2.67 | 2.62 | 6.80 |
| pass 1: turn | 1.15 | 1.14 | 1.11 | 1.22 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 3.39 | 2.70 | 2.65 | 6.39 |
| pass 2: turn | 0.55 | 0.52 | 0.50 | 0.60 |
| pass 3 (draft 3): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 3: gpu | 3.49 | 2.66 | 2.61 | 6.36 |
| pass 3: turn (the server before the next verify) | 7.21 | 7.31 | 6.05 | 8.15 |
| the draft's span (all passes) | 23.98 | 21.97 | 20.41 | 34.18 |
|   the draft's GPU time on its last card | 14.11 | 11.69 | 11.54 | 24.11 |
|   the leads (crossings) | 0.00 | 0.00 | 0.00 | 0.00 |
|   the turns of the passes before the last | 2.66 | 2.49 | 2.35 | 3.00 |
|   the draft's GPU time on the embedding card | 0.00 | 0.00 | 0.00 | 0.00 |
| the host's launch of the main pass (inside verify_span) | 4.40 | 3.80 | 3.71 | 3.96 |

verify_gpu per card, median ms: 12.66, 11.25, 11.38, 11.47, 9.74, 11.42, 9.62, 11.43, 12.94

**d1on, smoke: 169 typical steps, 4 draft passes a step, 4 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 128.92 | 128.43 | 124.90 | 131.95 |
| verify_span | 108.38 | 108.17 | 107.01 | 109.54 |
|   verify_gpu (the nine cards) | 102.32 | 102.11 | 100.98 | 103.44 |
|   verify_rest (crossings, rows back, acceptance, server) | 6.05 | 6.04 | 5.96 | 6.13 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 3.68 | 3.67 | 3.62 | 3.73 |
| pass 0: turn | 0.82 | 0.73 | 0.72 | 1.24 |
| pass 1 (draft 1): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 1: gpu | 2.94 | 2.69 | 2.65 | 3.12 |
| pass 1: turn | 1.14 | 1.13 | 1.11 | 1.19 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 2.89 | 2.69 | 2.65 | 2.75 |
| pass 2: turn | 0.54 | 0.51 | 0.50 | 0.59 |
| pass 3 (draft 3): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 3: gpu | 2.87 | 2.65 | 2.60 | 2.69 |
| pass 3: turn (the server before the next verify) | 5.66 | 6.08 | 2.16 | 7.52 |
| the draft's span (all passes) | 20.54 | 20.30 | 16.23 | 22.83 |
|   the draft's GPU time on its last card | 12.37 | 11.69 | 11.57 | 12.89 |
|   the leads (crossings) | 0.00 | 0.00 | 0.00 | 0.00 |
|   the turns of the passes before the last | 2.51 | 2.39 | 2.35 | 2.93 |
|   the draft's GPU time on the embedding card | 0.00 | 0.00 | 0.00 | 0.00 |
| the host's launch of the main pass (inside verify_span) | 4.44 | 3.89 | 3.77 | 4.04 |

verify_gpu per card, median ms: 12.64, 11.30, 11.37, 11.50, 9.76, 11.43, 9.67, 11.44, 13.05

