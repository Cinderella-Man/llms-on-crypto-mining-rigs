hw1: 1398 evaluations in 1 pieces (cut at gaps over 2000 ms): 1398 evaluations and 108 main passes
evaluations between main passes, by card: card 8: 425; the draft's last card: 8

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps |
|---|---:|---:|---:|---|---:|
| d0p2 | 108 | 0 | 107 | 1: 1, 4: 106 | 103 |

**hw1, all requests: 103 typical steps, 4 draft passes a step, 4 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 102.83 | 102.60 | 101.22 | 104.07 |
| verify_span | 90.29 | 90.30 | 89.00 | 91.52 |
|   verify_gpu (the nine cards) | 86.69 | 86.72 | 85.41 | 87.95 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.60 | 3.57 | 3.54 | 3.60 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 1.05 | 1.04 | 1.03 | 1.07 |
| pass 0: turn | 0.99 | 0.99 | 0.89 | 1.09 |
| pass 1 (draft 1): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 1: gpu | 2.61 | 2.59 | 2.55 | 2.61 |
| pass 1: turn | 0.91 | 0.91 | 0.88 | 0.94 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 2.53 | 2.52 | 2.49 | 2.55 |
| pass 2: turn | 0.87 | 0.87 | 0.84 | 0.90 |
| pass 3 (draft 3): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 3: gpu | 2.54 | 2.53 | 2.50 | 2.56 |
| pass 3: turn (the server before the next verify) | 1.05 | 0.96 | 0.94 | 0.98 |
| the draft's span (all passes) | 12.54 | 12.40 | 12.26 | 12.55 |
|   the draft's GPU time on its last card | 8.72 | 8.68 | 8.60 | 8.75 |
|   the leads (crossings) | 0.00 | 0.00 | 0.00 | 0.00 |
|   the turns of the passes before the last | 2.77 | 2.76 | 2.67 | 2.86 |
|   the draft's GPU time on the embedding card | 0.00 | 0.00 | 0.00 | 0.00 |
| the host's launch of the main pass (inside verify_span) | 3.59 | 3.16 | 3.06 | 3.28 |

verify_gpu per card, median ms: 10.82, 10.18, 10.39, 10.54, 8.60, 10.08, 8.52, 9.89, 7.71

**hw1, d0p2: 103 typical steps, 4 draft passes a step, 4 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 102.83 | 102.60 | 101.22 | 104.07 |
| verify_span | 90.29 | 90.30 | 89.00 | 91.52 |
|   verify_gpu (the nine cards) | 86.69 | 86.72 | 85.41 | 87.95 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.60 | 3.57 | 3.54 | 3.60 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 1.05 | 1.04 | 1.03 | 1.07 |
| pass 0: turn | 0.99 | 0.99 | 0.89 | 1.09 |
| pass 1 (draft 1): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 1: gpu | 2.61 | 2.59 | 2.55 | 2.61 |
| pass 1: turn | 0.91 | 0.91 | 0.88 | 0.94 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 2.53 | 2.52 | 2.49 | 2.55 |
| pass 2: turn | 0.87 | 0.87 | 0.84 | 0.90 |
| pass 3 (draft 3): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 3: gpu | 2.54 | 2.53 | 2.50 | 2.56 |
| pass 3: turn (the server before the next verify) | 1.05 | 0.96 | 0.94 | 0.98 |
| the draft's span (all passes) | 12.54 | 12.40 | 12.26 | 12.55 |
|   the draft's GPU time on its last card | 8.72 | 8.68 | 8.60 | 8.75 |
|   the leads (crossings) | 0.00 | 0.00 | 0.00 | 0.00 |
|   the turns of the passes before the last | 2.77 | 2.76 | 2.67 | 2.86 |
|   the draft's GPU time on the embedding card | 0.00 | 0.00 | 0.00 | 0.00 |
| the host's launch of the main pass (inside verify_span) | 3.59 | 3.16 | 3.06 | 3.28 |

verify_gpu per card, median ms: 10.82, 10.18, 10.39, 10.54, 8.60, 10.08, 8.52, 9.89, 7.71

