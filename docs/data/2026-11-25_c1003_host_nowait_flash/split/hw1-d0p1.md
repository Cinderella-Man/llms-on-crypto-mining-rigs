hw1: 1421 evaluations in 1 pieces (cut at gaps over 2000 ms): 1421 evaluations and 110 main passes
evaluations between main passes, by card: card 8: 430; the draft's last card: 8

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps |
|---|---:|---:|---:|---|---:|
| d0p1 | 110 | 1 | 108 | 1: 1, 4: 106, 5: 1 | 103 |

**hw1, all requests: 103 typical steps, 4 draft passes a step, 4 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 105.42 | 103.93 | 101.72 | 114.87 |
| verify_span | 90.29 | 90.39 | 89.10 | 91.67 |
|   verify_gpu (the nine cards) | 86.71 | 86.82 | 85.54 | 88.06 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.58 | 3.57 | 3.53 | 3.62 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 1.04 | 1.04 | 1.02 | 1.07 |
| pass 0: turn | 0.98 | 0.98 | 0.89 | 1.09 |
| pass 1 (draft 1): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 1: gpu | 3.15 | 2.59 | 2.55 | 5.96 |
| pass 1: turn | 0.97 | 0.95 | 0.90 | 1.04 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 2.95 | 2.52 | 2.49 | 5.54 |
| pass 2: turn | 0.90 | 0.90 | 0.86 | 0.93 |
| pass 3 (draft 3): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 3: gpu | 2.89 | 2.52 | 2.48 | 3.23 |
| pass 3: turn (the server before the next verify) | 2.25 | 1.89 | 0.97 | 2.86 |
| the draft's span (all passes) | 15.14 | 13.51 | 12.40 | 24.13 |
|   the draft's GPU time on its last card | 10.03 | 8.66 | 8.59 | 15.76 |
|   the leads (crossings) | 0.00 | 0.00 | 0.00 | 0.00 |
|   the turns of the passes before the last | 2.85 | 2.84 | 2.73 | 2.99 |
|   the draft's GPU time on the embedding card | 0.00 | 0.00 | 0.00 | 0.00 |
| the host's launch of the main pass (inside verify_span) | 3.38 | 3.08 | 3.02 | 3.25 |

verify_gpu per card, median ms: 10.87, 10.21, 10.38, 10.51, 8.63, 10.11, 8.57, 9.86, 7.64

**hw1, d0p1: 103 typical steps, 4 draft passes a step, 4 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 105.42 | 103.93 | 101.72 | 114.87 |
| verify_span | 90.29 | 90.39 | 89.10 | 91.67 |
|   verify_gpu (the nine cards) | 86.71 | 86.82 | 85.54 | 88.06 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.58 | 3.57 | 3.53 | 3.62 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 1.04 | 1.04 | 1.02 | 1.07 |
| pass 0: turn | 0.98 | 0.98 | 0.89 | 1.09 |
| pass 1 (draft 1): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 1: gpu | 3.15 | 2.59 | 2.55 | 5.96 |
| pass 1: turn | 0.97 | 0.95 | 0.90 | 1.04 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 2.95 | 2.52 | 2.49 | 5.54 |
| pass 2: turn | 0.90 | 0.90 | 0.86 | 0.93 |
| pass 3 (draft 3): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 3: gpu | 2.89 | 2.52 | 2.48 | 3.23 |
| pass 3: turn (the server before the next verify) | 2.25 | 1.89 | 0.97 | 2.86 |
| the draft's span (all passes) | 15.14 | 13.51 | 12.40 | 24.13 |
|   the draft's GPU time on its last card | 10.03 | 8.66 | 8.59 | 15.76 |
|   the leads (crossings) | 0.00 | 0.00 | 0.00 | 0.00 |
|   the turns of the passes before the last | 2.85 | 2.84 | 2.73 | 2.99 |
|   the draft's GPU time on the embedding card | 0.00 | 0.00 | 0.00 | 0.00 |
| the host's launch of the main pass (inside verify_span) | 3.38 | 3.08 | 3.02 | 3.25 |

verify_gpu per card, median ms: 10.87, 10.21, 10.38, 10.51, 8.63, 10.11, 8.57, 9.86, 7.64

