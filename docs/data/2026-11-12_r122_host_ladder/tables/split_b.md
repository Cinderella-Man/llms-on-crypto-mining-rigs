b: 3534 evaluations in 4 pieces (cut at gaps over 2000 ms): 9 evaluations and 1 main passes, 3 evaluations and 0 main passes, 1830 evaluations and 89 main passes, 1692 evaluations and 83 main passes
evaluations between main passes, by card: card 4: 652, card 8: 1304; the draft's last card: 8

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps |
|---|---:|---:|---:|---|---:|
| greedy | 89 | 4 | 84 | 8: 84 | 81 |
| long | 83 | 6 | 76 | 8: 75, 16: 1 | 73 |

**b, all requests: 154 typical steps, 8 draft passes a step, 12 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 115.56 | 116.70 | 108.63 | 122.81 |
| verify_span | 96.22 | 94.55 | 91.92 | 100.44 |
|   verify_gpu (the nine cards) | 91.57 | 90.00 | 87.43 | 95.70 |
|   verify_rest (crossings, rows back, acceptance, server) | 4.64 | 4.64 | 4.49 | 4.78 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 0.04 | 0.04 | 0.03 | 0.05 |
| pass 0: turn | 0.03 | 0.03 | 0.03 | 0.04 |
| pass 1 (draft 1): lead | 0.46 | 0.46 | 0.45 | 0.47 |
| pass 1: gpu | 3.59 | 3.40 | 3.27 | 3.91 |
| pass 1: turn | 0.39 | 0.31 | 0.30 | 0.33 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 2: turn | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 3 (draft 3): lead | 0.45 | 0.45 | 0.45 | 0.46 |
| pass 3: gpu | 3.15 | 2.77 | 2.50 | 6.26 |
| pass 3: turn | 0.58 | 0.57 | 0.55 | 0.62 |
| pass 4 (draft 4): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 4: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 4: turn | 0.02 | 0.02 | 0.02 | 0.03 |
| pass 5 (draft 5): lead | 0.45 | 0.45 | 0.44 | 0.46 |
| pass 5: gpu | 3.21 | 2.73 | 2.49 | 6.25 |
| pass 5: turn | 0.56 | 0.55 | 0.52 | 0.59 |
| pass 6 (draft 6): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 6: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 6: turn | 0.02 | 0.02 | 0.02 | 0.03 |
| pass 7 (draft 7): lead | 0.45 | 0.45 | 0.44 | 0.46 |
| pass 7: gpu | 3.06 | 2.73 | 2.48 | 6.21 |
| pass 7: turn (the server before the next verify) | 2.74 | 2.32 | 1.97 | 2.61 |
| the draft's span (all passes) | 19.34 | 17.91 | 16.51 | 28.84 |
|   the draft's GPU time on its last card | 13.15 | 12.25 | 10.93 | 22.23 |
|   the leads (crossings) | 1.81 | 1.81 | 1.80 | 1.83 |
|   the turns of the passes before the last | 1.64 | 1.56 | 1.49 | 1.70 |
|   the draft's GPU time on the embedding card | 0.06 | 0.06 | 0.05 | 0.06 |
| the host's launch of the main pass (inside verify_span) | 3.89 | 3.18 | 3.11 | 3.31 |

verify_gpu per card, median ms: 11.14, 10.89, 10.94, 11.05, 9.08, 10.56, 8.96, 10.29, 7.39

**b, greedy: 81 typical steps, 8 draft passes a step, 12 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 112.51 | 109.94 | 108.35 | 122.54 |
| verify_span | 92.96 | 92.76 | 91.63 | 94.06 |
|   verify_gpu (the nine cards) | 88.41 | 88.20 | 87.12 | 89.41 |
|   verify_rest (crossings, rows back, acceptance, server) | 4.54 | 4.52 | 4.47 | 4.60 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 0.04 | 0.04 | 0.03 | 0.04 |
| pass 0: turn | 0.03 | 0.03 | 0.03 | 0.04 |
| pass 1 (draft 1): lead | 0.46 | 0.46 | 0.45 | 0.47 |
| pass 1: gpu | 3.32 | 3.31 | 3.25 | 3.35 |
| pass 1: turn | 0.43 | 0.31 | 0.30 | 1.09 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 2: turn | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 3 (draft 3): lead | 0.45 | 0.45 | 0.45 | 0.46 |
| pass 3: gpu | 3.27 | 2.56 | 2.49 | 6.29 |
| pass 3: turn | 0.58 | 0.57 | 0.54 | 0.62 |
| pass 4 (draft 4): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 4: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 4: turn | 0.02 | 0.02 | 0.02 | 0.03 |
| pass 5 (draft 5): lead | 0.45 | 0.45 | 0.44 | 0.46 |
| pass 5: gpu | 3.38 | 2.53 | 2.47 | 6.27 |
| pass 5: turn | 0.55 | 0.54 | 0.53 | 0.59 |
| pass 6 (draft 6): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 6: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 6: turn | 0.02 | 0.02 | 0.02 | 0.03 |
| pass 7 (draft 7): lead | 0.45 | 0.45 | 0.44 | 0.46 |
| pass 7: gpu | 3.16 | 2.52 | 2.46 | 6.23 |
| pass 7: turn (the server before the next verify) | 2.81 | 2.32 | 2.09 | 2.50 |
| the draft's span (all passes) | 19.56 | 16.73 | 16.38 | 30.24 |
|   the draft's GPU time on its last card | 13.27 | 11.04 | 10.86 | 22.36 |
|   the leads (crossings) | 1.81 | 1.81 | 1.80 | 1.83 |
|   the turns of the passes before the last | 1.67 | 1.53 | 1.49 | 2.36 |
|   the draft's GPU time on the embedding card | 0.06 | 0.06 | 0.05 | 0.06 |
| the host's launch of the main pass (inside verify_span) | 3.72 | 3.14 | 3.10 | 3.31 |

verify_gpu per card, median ms: 10.96, 10.63, 10.76, 10.81, 8.86, 10.28, 8.69, 10.00, 7.23

**b, long: 73 typical steps, 8 draft passes a step, 12 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 118.93 | 117.75 | 116.30 | 124.50 |
| verify_span | 99.83 | 99.56 | 98.24 | 101.02 |
|   verify_gpu (the nine cards) | 95.08 | 94.86 | 93.48 | 96.28 |
|   verify_rest (crossings, rows back, acceptance, server) | 4.75 | 4.75 | 4.71 | 4.79 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 0.04 | 0.04 | 0.03 | 0.05 |
| pass 0: turn | 0.03 | 0.03 | 0.03 | 0.04 |
| pass 1 (draft 1): lead | 0.46 | 0.45 | 0.45 | 0.47 |
| pass 1: gpu | 3.88 | 3.85 | 3.79 | 3.93 |
| pass 1: turn | 0.34 | 0.32 | 0.31 | 0.33 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 2: turn | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 3 (draft 3): lead | 0.45 | 0.45 | 0.45 | 0.46 |
| pass 3: gpu | 3.02 | 2.78 | 2.77 | 2.86 |
| pass 3: turn | 0.59 | 0.59 | 0.55 | 0.63 |
| pass 4 (draft 4): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 4: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 4: turn | 0.02 | 0.02 | 0.02 | 0.03 |
| pass 5 (draft 5): lead | 0.45 | 0.45 | 0.45 | 0.45 |
| pass 5: gpu | 3.03 | 2.74 | 2.73 | 2.82 |
| pass 5: turn | 0.56 | 0.56 | 0.52 | 0.59 |
| pass 6 (draft 6): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 6: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 6: turn | 0.02 | 0.02 | 0.02 | 0.03 |
| pass 7 (draft 7): lead | 0.45 | 0.45 | 0.45 | 0.46 |
| pass 7: gpu | 2.95 | 2.74 | 2.73 | 2.80 |
| pass 7: turn (the server before the next verify) | 2.66 | 2.34 | 1.96 | 2.61 |
| the draft's span (all passes) | 19.10 | 18.01 | 17.51 | 21.65 |
|   the draft's GPU time on its last card | 13.02 | 12.28 | 12.20 | 13.53 |
|   the leads (crossings) | 1.81 | 1.81 | 1.80 | 1.83 |
|   the turns of the passes before the last | 1.60 | 1.58 | 1.49 | 1.63 |
|   the draft's GPU time on the embedding card | 0.06 | 0.06 | 0.05 | 0.06 |
| the host's launch of the main pass (inside verify_span) | 4.08 | 3.19 | 3.16 | 3.29 |

verify_gpu per card, median ms: 11.50, 11.68, 11.26, 11.87, 9.41, 10.82, 9.77, 10.64, 7.91

