f0-2: 2986 evaluations in 4 pieces (cut at gaps over 2000 ms): 9 evaluations and 1 main passes, 2 evaluations and 0 main passes, 1485 evaluations and 89 main passes, 1490 evaluations and 90 main passes
evaluations between main passes, by card: card 4: 679, card 8: 679; the draft's last card: 8

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps |
|---|---:|---:|---:|---|---:|
| greedy | 89 | 4 | 84 | 3: 1, 4: 83 | 80 |
| long | 90 | 6 | 83 | 4: 82, 8: 1 | 80 |

**f0-2, all requests: 160 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 113.53 | 114.92 | 106.56 | 119.60 |
| verify_span | 95.57 | 96.49 | 90.85 | 99.83 |
|   verify_gpu (the nine cards) | 91.96 | 92.80 | 87.39 | 96.16 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.61 | 3.66 | 3.43 | 3.71 |
| pass 0 (catch-up): lead | 0.51 | 0.51 | 0.50 | 0.52 |
| pass 0: gpu | 3.61 | 3.75 | 3.28 | 3.91 |
| pass 0: turn | -2.03 | -2.30 | -2.55 | 0.28 |
| pass 1 (draft 1): lead | 2.60 | 2.79 | 0.55 | 3.06 |
| pass 1: gpu | 3.06 | 2.75 | 2.49 | 3.37 |
| pass 1: turn | 0.59 | 0.58 | 0.53 | 0.65 |
| pass 2 (draft 2): lead | 0.51 | 0.51 | 0.49 | 0.52 |
| pass 2: gpu | 3.08 | 2.75 | 2.49 | 3.55 |
| pass 2: turn | 0.56 | 0.55 | 0.52 | 0.58 |
| pass 3 (draft 3): lead | 0.50 | 0.50 | 0.49 | 0.51 |
| pass 3: gpu | 3.02 | 2.75 | 2.49 | 3.25 |
| pass 3: turn (the server before the next verify) | 1.94 | 1.77 | 1.02 | 2.62 |
| the draft's span (all passes) | 17.96 | 16.72 | 15.38 | 26.82 |
|   the draft's GPU time on its last card | 12.78 | 12.16 | 10.77 | 14.02 |
|   the leads (crossings) | 4.12 | 4.33 | 2.08 | 4.57 |
|   the turns of the passes before the last | -0.89 | -1.18 | -1.42 | 1.39 |
|   the draft's GPU time on the embedding card | 0.08 | 0.08 | 0.08 | 0.09 |
| the host's launch of the main pass (inside verify_span) | 3.89 | 3.19 | 3.12 | 3.32 |

verify_gpu per card, median ms: 11.26, 11.40, 11.04, 11.46, 9.12, 10.62, 9.38, 10.37, 7.78

**f0-2, greedy: 80 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 110.07 | 107.96 | 106.36 | 119.39 |
| verify_span | 91.85 | 91.67 | 90.60 | 92.83 |
|   verify_gpu (the nine cards) | 88.35 | 88.16 | 87.14 | 89.36 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.50 | 3.45 | 3.42 | 3.82 |
| pass 0 (catch-up): lead | 0.51 | 0.51 | 0.50 | 0.52 |
| pass 0: gpu | 3.33 | 3.31 | 3.27 | 3.36 |
| pass 0: turn | -1.68 | -2.22 | -2.28 | 0.28 |
| pass 1 (draft 1): lead | 2.29 | 2.71 | 0.50 | 2.77 |
| pass 1: gpu | 3.23 | 2.50 | 2.48 | 6.28 |
| pass 1: turn | 0.60 | 0.58 | 0.53 | 0.65 |
| pass 2 (draft 2): lead | 0.51 | 0.51 | 0.50 | 0.53 |
| pass 2: gpu | 3.26 | 2.51 | 2.48 | 6.26 |
| pass 2: turn | 0.55 | 0.55 | 0.51 | 0.57 |
| pass 3 (draft 3): lead | 0.50 | 0.50 | 0.50 | 0.51 |
| pass 3: gpu | 3.16 | 2.50 | 2.46 | 6.22 |
| pass 3: turn (the server before the next verify) | 1.95 | 1.86 | 1.11 | 2.59 |
| the draft's span (all passes) | 18.22 | 15.94 | 15.14 | 27.18 |
|   the draft's GPU time on its last card | 12.98 | 10.82 | 10.75 | 22.11 |
|   the leads (crossings) | 3.82 | 4.22 | 2.03 | 4.30 |
|   the turns of the passes before the last | -0.53 | -1.08 | -1.15 | 1.59 |
|   the draft's GPU time on the embedding card | 0.09 | 0.08 | 0.08 | 0.09 |
| the host's launch of the main pass (inside verify_span) | 3.75 | 3.16 | 3.10 | 3.32 |

verify_gpu per card, median ms: 10.94, 10.61, 10.76, 10.81, 8.81, 10.27, 8.69, 10.03, 7.25

**f0-2, long: 80 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 116.98 | 115.86 | 114.44 | 119.88 |
| verify_span | 99.28 | 98.83 | 97.42 | 100.35 |
|   verify_gpu (the nine cards) | 95.57 | 95.13 | 93.77 | 96.64 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.72 | 3.68 | 3.66 | 3.71 |
| pass 0 (catch-up): lead | 0.51 | 0.51 | 0.50 | 0.52 |
| pass 0: gpu | 3.90 | 3.88 | 3.80 | 3.94 |
| pass 0: turn | -2.38 | -2.50 | -2.57 | -2.40 |
| pass 1 (draft 1): lead | 2.91 | 2.99 | 2.90 | 3.08 |
| pass 1: gpu | 2.90 | 2.79 | 2.74 | 2.83 |
| pass 1: turn | 0.59 | 0.59 | 0.53 | 0.63 |
| pass 2 (draft 2): lead | 0.51 | 0.50 | 0.49 | 0.51 |
| pass 2: gpu | 2.91 | 2.79 | 2.74 | 2.83 |
| pass 2: turn | 0.56 | 0.56 | 0.52 | 0.59 |
| pass 3 (draft 3): lead | 0.50 | 0.50 | 0.49 | 0.51 |
| pass 3: gpu | 2.88 | 2.79 | 2.75 | 2.82 |
| pass 3: turn (the server before the next verify) | 1.93 | 1.52 | 0.99 | 3.37 |
| the draft's span (all passes) | 17.70 | 17.12 | 16.35 | 18.72 |
|   the draft's GPU time on its last card | 12.58 | 12.23 | 12.13 | 12.35 |
|   the leads (crossings) | 4.43 | 4.51 | 4.42 | 4.61 |
|   the turns of the passes before the last | -1.24 | -1.35 | -1.48 | -1.21 |
|   the draft's GPU time on the embedding card | 0.08 | 0.08 | 0.08 | 0.09 |
| the host's launch of the main pass (inside verify_span) | 4.03 | 3.20 | 3.17 | 3.32 |

verify_gpu per card, median ms: 11.56, 11.73, 11.36, 11.94, 9.42, 10.92, 9.79, 10.67, 7.88

