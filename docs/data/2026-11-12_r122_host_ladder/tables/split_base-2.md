base-2: 3702 evaluations in 4 pieces (cut at gaps over 2000 ms): 9 evaluations and 1 main passes, 3 evaluations and 0 main passes, 1830 evaluations and 89 main passes, 1860 evaluations and 91 main passes
evaluations between main passes, by card: card 4: 684, card 8: 1368; the draft's last card: 8

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps |
|---|---:|---:|---:|---|---:|
| greedy | 89 | 4 | 84 | 8: 84 | 81 |
| long | 91 | 6 | 84 | 8: 83, 16: 1 | 81 |

**base-2, all requests: 162 typical steps, 8 draft passes a step, 12 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 120.51 | 121.58 | 113.74 | 127.10 |
| verify_span | 96.66 | 98.20 | 92.04 | 100.64 |
|   verify_gpu (the nine cards) | 91.94 | 93.43 | 87.45 | 95.89 |
|   verify_rest (crossings, rows back, acceptance, server) | 4.72 | 4.73 | 4.51 | 4.84 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 0.04 | 0.04 | 0.03 | 0.05 |
| pass 0: turn | 0.03 | 0.03 | 0.03 | 0.04 |
| pass 1 (draft 1): lead | 0.46 | 0.46 | 0.45 | 0.47 |
| pass 1: gpu | 3.62 | 3.76 | 3.27 | 3.96 |
| pass 1: turn | 0.40 | 0.33 | 0.32 | 0.35 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 2: turn | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 3 (draft 3): lead | 0.45 | 0.45 | 0.45 | 0.46 |
| pass 3: gpu | 3.20 | 2.78 | 2.51 | 6.29 |
| pass 3: turn | 0.56 | 0.54 | 0.52 | 0.57 |
| pass 4 (draft 4): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 4: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 4: turn | 0.02 | 0.02 | 0.02 | 0.02 |
| pass 5 (draft 5): lead | 0.45 | 0.45 | 0.44 | 0.45 |
| pass 5: gpu | 3.27 | 2.74 | 2.52 | 6.25 |
| pass 5: turn | 0.52 | 0.51 | 0.48 | 0.56 |
| pass 6 (draft 6): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 6: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 6: turn | 0.02 | 0.02 | 0.02 | 0.02 |
| pass 7 (draft 7): lead | 0.45 | 0.45 | 0.44 | 0.45 |
| pass 7: gpu | 3.12 | 2.74 | 2.51 | 6.25 |
| pass 7: turn (the server before the next verify) | 7.13 | 7.13 | 5.53 | 8.28 |
| the draft's span (all passes) | 23.85 | 22.41 | 20.59 | 33.07 |
|   the draft's GPU time on its last card | 13.33 | 12.30 | 10.99 | 22.24 |
|   the leads (crossings) | 1.81 | 1.80 | 1.79 | 1.82 |
|   the turns of the passes before the last | 1.57 | 1.49 | 1.43 | 1.63 |
|   the draft's GPU time on the embedding card | 0.06 | 0.06 | 0.05 | 0.07 |
| the host's launch of the main pass (inside verify_span) | 3.80 | 3.23 | 3.16 | 3.36 |

verify_gpu per card, median ms: 11.26, 11.42, 11.05, 11.53, 9.25, 10.69, 9.54, 10.43, 7.76

**base-2, greedy: 81 typical steps, 8 draft passes a step, 12 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 117.32 | 115.26 | 112.97 | 126.80 |
| verify_span | 93.11 | 92.82 | 91.76 | 94.24 |
|   verify_gpu (the nine cards) | 88.46 | 88.24 | 87.20 | 89.62 |
|   verify_rest (crossings, rows back, acceptance, server) | 4.65 | 4.55 | 4.49 | 4.63 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 0.04 | 0.04 | 0.03 | 0.05 |
| pass 0: turn | 0.03 | 0.03 | 0.03 | 0.04 |
| pass 1 (draft 1): lead | 0.46 | 0.46 | 0.45 | 0.47 |
| pass 1: gpu | 3.33 | 3.30 | 3.26 | 3.36 |
| pass 1: turn | 0.43 | 0.32 | 0.31 | 0.81 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 2: turn | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 3 (draft 3): lead | 0.45 | 0.45 | 0.45 | 0.46 |
| pass 3: gpu | 3.27 | 2.57 | 2.49 | 6.31 |
| pass 3: turn | 0.54 | 0.53 | 0.52 | 0.55 |
| pass 4 (draft 4): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 4: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 4: turn | 0.02 | 0.02 | 0.02 | 0.02 |
| pass 5 (draft 5): lead | 0.45 | 0.44 | 0.44 | 0.45 |
| pass 5: gpu | 3.39 | 2.55 | 2.50 | 6.28 |
| pass 5: turn | 0.51 | 0.50 | 0.48 | 0.56 |
| pass 6 (draft 6): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 6: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 6: turn | 0.02 | 0.02 | 0.02 | 0.02 |
| pass 7 (draft 7): lead | 0.45 | 0.45 | 0.44 | 0.45 |
| pass 7: gpu | 3.18 | 2.54 | 2.50 | 6.25 |
| pass 7: turn (the server before the next verify) | 7.52 | 7.37 | 6.24 | 9.00 |
| the draft's span (all passes) | 24.21 | 22.18 | 20.59 | 33.15 |
|   the draft's GPU time on its last card | 13.31 | 11.08 | 10.97 | 22.31 |
|   the leads (crossings) | 1.80 | 1.80 | 1.79 | 1.82 |
|   the turns of the passes before the last | 1.58 | 1.45 | 1.42 | 2.30 |
|   the draft's GPU time on the embedding card | 0.06 | 0.06 | 0.05 | 0.06 |
| the host's launch of the main pass (inside verify_span) | 3.81 | 3.24 | 3.18 | 3.39 |

verify_gpu per card, median ms: 10.98, 10.64, 10.74, 10.81, 8.86, 10.29, 8.72, 10.01, 7.21

**base-2, long: 81 typical steps, 8 draft passes a step, 12 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 123.70 | 122.91 | 120.05 | 127.75 |
| verify_span | 100.21 | 100.07 | 99.17 | 101.03 |
|   verify_gpu (the nine cards) | 95.42 | 95.33 | 94.37 | 96.22 |
|   verify_rest (crossings, rows back, acceptance, server) | 4.79 | 4.79 | 4.74 | 4.85 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 0.04 | 0.04 | 0.03 | 0.05 |
| pass 0: turn | 0.03 | 0.03 | 0.03 | 0.04 |
| pass 1 (draft 1): lead | 0.46 | 0.46 | 0.45 | 0.47 |
| pass 1: gpu | 3.92 | 3.90 | 3.83 | 3.97 |
| pass 1: turn | 0.37 | 0.33 | 0.32 | 0.34 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 0.03 | 0.03 | 0.03 | 0.04 |
| pass 2: turn | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 3 (draft 3): lead | 0.45 | 0.45 | 0.45 | 0.46 |
| pass 3: gpu | 3.12 | 2.80 | 2.78 | 3.31 |
| pass 3: turn | 0.57 | 0.55 | 0.54 | 0.57 |
| pass 4 (draft 4): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 4: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 4: turn | 0.02 | 0.02 | 0.02 | 0.02 |
| pass 5 (draft 5): lead | 0.45 | 0.45 | 0.44 | 0.46 |
| pass 5: gpu | 3.14 | 2.75 | 2.73 | 4.08 |
| pass 5: turn | 0.53 | 0.52 | 0.51 | 0.56 |
| pass 6 (draft 6): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 6: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 6: turn | 0.02 | 0.02 | 0.02 | 0.02 |
| pass 7 (draft 7): lead | 0.45 | 0.45 | 0.44 | 0.45 |
| pass 7: gpu | 3.05 | 2.76 | 2.73 | 2.84 |
| pass 7: turn (the server before the next verify) | 6.75 | 6.79 | 5.01 | 7.81 |
| the draft's span (all passes) | 23.48 | 22.63 | 20.78 | 25.45 |
|   the draft's GPU time on its last card | 13.36 | 12.37 | 12.25 | 14.24 |
|   the leads (crossings) | 1.81 | 1.81 | 1.79 | 1.83 |
|   the turns of the passes before the last | 1.56 | 1.50 | 1.47 | 1.59 |
|   the draft's GPU time on the embedding card | 0.06 | 0.06 | 0.06 | 0.07 |
| the host's launch of the main pass (inside verify_span) | 3.79 | 3.20 | 3.16 | 3.34 |

verify_gpu per card, median ms: 11.55, 11.73, 11.29, 11.92, 9.43, 10.85, 9.84, 10.72, 7.93

