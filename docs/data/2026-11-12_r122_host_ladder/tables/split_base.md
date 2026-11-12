base: 3840 evaluations in 4 pieces (cut at gaps over 2000 ms): 9 evaluations and 1 main passes, 3 evaluations and 0 main passes, 1830 evaluations and 89 main passes, 1998 evaluations and 98 main passes
evaluations between main passes, by card: card 4: 712, card 8: 1424; the draft's last card: 8

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps |
|---|---:|---:|---:|---|---:|
| greedy | 89 | 4 | 84 | 8: 84 | 81 |
| long | 98 | 6 | 91 | 8: 90, 16: 1 | 88 |

**base, all requests: 169 typical steps, 8 draft passes a step, 12 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 121.25 | 121.87 | 113.96 | 129.12 |
| verify_span | 96.81 | 98.85 | 92.04 | 100.94 |
|   verify_gpu (the nine cards) | 92.10 | 94.06 | 87.50 | 96.14 |
|   verify_rest (crossings, rows back, acceptance, server) | 4.70 | 4.73 | 4.53 | 4.86 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 0.04 | 0.04 | 0.03 | 0.06 |
| pass 0: turn | 0.03 | 0.03 | 0.03 | 0.04 |
| pass 1 (draft 1): lead | 0.46 | 0.46 | 0.45 | 0.47 |
| pass 1: gpu | 3.62 | 3.75 | 3.27 | 3.93 |
| pass 1: turn | 0.40 | 0.32 | 0.31 | 0.34 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 2: turn | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 3 (draft 3): lead | 0.45 | 0.45 | 0.45 | 0.46 |
| pass 3: gpu | 3.33 | 2.79 | 2.50 | 6.43 |
| pass 3: turn | 0.57 | 0.55 | 0.53 | 0.57 |
| pass 4 (draft 4): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 4: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 4: turn | 0.02 | 0.02 | 0.02 | 0.02 |
| pass 5 (draft 5): lead | 0.45 | 0.45 | 0.44 | 0.46 |
| pass 5: gpu | 3.40 | 2.76 | 2.52 | 6.39 |
| pass 5: turn | 0.54 | 0.52 | 0.49 | 0.58 |
| pass 6 (draft 6): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 6: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 6: turn | 0.02 | 0.02 | 0.02 | 0.02 |
| pass 7 (draft 7): lead | 0.45 | 0.45 | 0.44 | 0.46 |
| pass 7: gpu | 3.24 | 2.75 | 2.52 | 6.27 |
| pass 7: turn (the server before the next verify) | 7.31 | 7.30 | 5.80 | 9.00 |
| the draft's span (all passes) | 24.44 | 22.81 | 20.77 | 34.53 |
|   the draft's GPU time on its last card | 13.72 | 12.33 | 11.01 | 23.18 |
|   the leads (crossings) | 1.81 | 1.81 | 1.80 | 1.83 |
|   the turns of the passes before the last | 1.60 | 1.49 | 1.44 | 2.31 |
|   the draft's GPU time on the embedding card | 0.06 | 0.06 | 0.05 | 0.06 |
| the host's launch of the main pass (inside verify_span) | 3.82 | 3.25 | 3.19 | 3.38 |

verify_gpu per card, median ms: 11.40, 11.49, 11.15, 11.65, 9.26, 10.69, 9.59, 10.41, 7.67

**base, greedy: 81 typical steps, 8 draft passes a step, 12 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 117.59 | 115.36 | 113.42 | 127.25 |
| verify_span | 92.99 | 92.81 | 91.63 | 94.14 |
|   verify_gpu (the nine cards) | 88.42 | 88.24 | 87.08 | 89.61 |
|   verify_rest (crossings, rows back, acceptance, server) | 4.57 | 4.56 | 4.51 | 4.63 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 0.04 | 0.04 | 0.03 | 0.06 |
| pass 0: turn | 0.03 | 0.03 | 0.03 | 0.05 |
| pass 1 (draft 1): lead | 0.46 | 0.46 | 0.45 | 0.47 |
| pass 1: gpu | 3.33 | 3.31 | 3.26 | 3.38 |
| pass 1: turn | 0.39 | 0.31 | 0.30 | 0.38 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 2: turn | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 3 (draft 3): lead | 0.45 | 0.45 | 0.45 | 0.46 |
| pass 3: gpu | 3.27 | 2.56 | 2.49 | 6.28 |
| pass 3: turn | 0.57 | 0.54 | 0.53 | 0.56 |
| pass 4 (draft 4): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 4: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 4: turn | 0.02 | 0.02 | 0.02 | 0.02 |
| pass 5 (draft 5): lead | 0.45 | 0.45 | 0.44 | 0.46 |
| pass 5: gpu | 3.39 | 2.55 | 2.51 | 6.28 |
| pass 5: turn | 0.52 | 0.51 | 0.49 | 0.57 |
| pass 6 (draft 6): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 6: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 6: turn | 0.02 | 0.02 | 0.02 | 0.02 |
| pass 7 (draft 7): lead | 0.45 | 0.45 | 0.44 | 0.45 |
| pass 7: gpu | 3.19 | 2.55 | 2.51 | 6.25 |
| pass 7: turn (the server before the next verify) | 7.90 | 7.70 | 6.39 | 9.50 |
| the draft's span (all passes) | 24.59 | 22.54 | 20.83 | 34.53 |
|   the draft's GPU time on its last card | 13.31 | 11.08 | 10.98 | 22.31 |
|   the leads (crossings) | 1.81 | 1.81 | 1.80 | 1.82 |
|   the turns of the passes before the last | 1.58 | 1.46 | 1.42 | 2.31 |
|   the draft's GPU time on the embedding card | 0.06 | 0.06 | 0.05 | 0.06 |
| the host's launch of the main pass (inside verify_span) | 3.85 | 3.27 | 3.19 | 3.39 |

verify_gpu per card, median ms: 11.03, 10.60, 10.74, 10.81, 8.86, 10.27, 8.68, 10.01, 7.23

**base, long: 88 typical steps, 8 draft passes a step, 12 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 124.61 | 123.07 | 119.75 | 134.08 |
| verify_span | 100.31 | 100.18 | 99.05 | 101.18 |
|   verify_gpu (the nine cards) | 95.49 | 95.35 | 94.25 | 96.37 |
|   verify_rest (crossings, rows back, acceptance, server) | 4.83 | 4.80 | 4.75 | 4.88 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 0.04 | 0.04 | 0.03 | 0.05 |
| pass 0: turn | 0.03 | 0.03 | 0.03 | 0.04 |
| pass 1 (draft 1): lead | 0.46 | 0.46 | 0.45 | 0.47 |
| pass 1: gpu | 3.88 | 3.87 | 3.79 | 3.94 |
| pass 1: turn | 0.40 | 0.32 | 0.31 | 0.34 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 2: turn | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 3 (draft 3): lead | 0.46 | 0.45 | 0.45 | 0.46 |
| pass 3: gpu | 3.38 | 2.83 | 2.78 | 6.43 |
| pass 3: turn | 0.57 | 0.56 | 0.55 | 0.57 |
| pass 4 (draft 4): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 4: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 4: turn | 0.02 | 0.02 | 0.02 | 0.02 |
| pass 5 (draft 5): lead | 0.45 | 0.45 | 0.45 | 0.46 |
| pass 5: gpu | 3.41 | 2.79 | 2.73 | 6.41 |
| pass 5: turn | 0.56 | 0.52 | 0.51 | 0.58 |
| pass 6 (draft 6): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 6: gpu | 0.03 | 0.03 | 0.03 | 0.03 |
| pass 6: turn | 0.02 | 0.02 | 0.02 | 0.02 |
| pass 7 (draft 7): lead | 0.45 | 0.45 | 0.44 | 0.46 |
| pass 7: gpu | 3.29 | 2.79 | 2.74 | 6.39 |
| pass 7: turn (the server before the next verify) | 6.76 | 7.11 | 4.94 | 8.13 |
| the draft's span (all passes) | 24.30 | 22.94 | 20.51 | 34.28 |
|   the draft's GPU time on its last card | 14.09 | 12.40 | 12.26 | 23.25 |
|   the leads (crossings) | 1.82 | 1.81 | 1.80 | 1.84 |
|   the turns of the passes before the last | 1.63 | 1.51 | 1.48 | 1.64 |
|   the draft's GPU time on the embedding card | 0.06 | 0.06 | 0.05 | 0.06 |
| the host's launch of the main pass (inside verify_span) | 3.80 | 3.23 | 3.18 | 3.38 |

verify_gpu per card, median ms: 11.62, 11.76, 11.33, 11.95, 9.45, 10.89, 9.87, 10.69, 7.80

