ph: 1138 evaluations in 1 pieces (cut at gaps over 2000 ms): 1138 evaluations and 88 main passes
evaluations between main passes, by card: card 8: 345; the draft's last card: 8

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps |
|---|---:|---:|---:|---|---:|
| d0p2 | 88 | 0 | 87 | 2: 1, 3: 1, 4: 85 | 82 |

**ph, all requests: 82 typical steps, 4 draft passes a step, 4 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 97.04 | 96.68 | 95.30 | 98.36 |
| verify_span | 84.29 | 84.21 | 82.83 | 85.50 |
|   verify_gpu (the nine cards) | 80.72 | 80.63 | 79.25 | 81.95 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.57 | 3.56 | 3.53 | 3.60 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 1.05 | 1.05 | 1.03 | 1.07 |
| pass 0: turn | 0.98 | 0.97 | 0.88 | 1.06 |
| pass 1 (draft 1): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 1: gpu | 2.63 | 2.59 | 2.56 | 2.62 |
| pass 1: turn | 0.94 | 0.91 | 0.89 | 0.94 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 2.54 | 2.54 | 2.50 | 2.57 |
| pass 2: turn | 0.87 | 0.86 | 0.85 | 0.89 |
| pass 3 (draft 3): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 3: gpu | 2.56 | 2.56 | 2.51 | 2.58 |
| pass 3: turn (the server before the next verify) | 1.19 | 0.96 | 0.95 | 0.99 |
| the draft's span (all passes) | 12.75 | 12.45 | 12.32 | 12.59 |
|   the draft's GPU time on its last card | 8.78 | 8.74 | 8.65 | 8.80 |
|   the leads (crossings) | 0.00 | 0.00 | 0.00 | 0.00 |
|   the turns of the passes before the last | 2.78 | 2.75 | 2.65 | 2.86 |
|   the draft's GPU time on the embedding card | 0.00 | 0.00 | 0.00 | 0.00 |
| the host's launch of the main pass (inside verify_span) | 3.90 | 3.14 | 3.07 | 3.27 |

verify_gpu per card, median ms: 10.56, 9.42, 9.33, 9.31, 7.98, 9.26, 7.62, 9.40, 7.71

**ph, d0p2: 82 typical steps, 4 draft passes a step, 4 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 97.04 | 96.68 | 95.30 | 98.36 |
| verify_span | 84.29 | 84.21 | 82.83 | 85.50 |
|   verify_gpu (the nine cards) | 80.72 | 80.63 | 79.25 | 81.95 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.57 | 3.56 | 3.53 | 3.60 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 1.05 | 1.05 | 1.03 | 1.07 |
| pass 0: turn | 0.98 | 0.97 | 0.88 | 1.06 |
| pass 1 (draft 1): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 1: gpu | 2.63 | 2.59 | 2.56 | 2.62 |
| pass 1: turn | 0.94 | 0.91 | 0.89 | 0.94 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 2.54 | 2.54 | 2.50 | 2.57 |
| pass 2: turn | 0.87 | 0.86 | 0.85 | 0.89 |
| pass 3 (draft 3): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 3: gpu | 2.56 | 2.56 | 2.51 | 2.58 |
| pass 3: turn (the server before the next verify) | 1.19 | 0.96 | 0.95 | 0.99 |
| the draft's span (all passes) | 12.75 | 12.45 | 12.32 | 12.59 |
|   the draft's GPU time on its last card | 8.78 | 8.74 | 8.65 | 8.80 |
|   the leads (crossings) | 0.00 | 0.00 | 0.00 | 0.00 |
|   the turns of the passes before the last | 2.78 | 2.75 | 2.65 | 2.86 |
|   the draft's GPU time on the embedding card | 0.00 | 0.00 | 0.00 | 0.00 |
| the host's launch of the main pass (inside verify_span) | 3.90 | 3.14 | 3.07 | 3.27 |

verify_gpu per card, median ms: 10.56, 9.42, 9.33, 9.31, 7.98, 9.26, 7.62, 9.40, 7.71

