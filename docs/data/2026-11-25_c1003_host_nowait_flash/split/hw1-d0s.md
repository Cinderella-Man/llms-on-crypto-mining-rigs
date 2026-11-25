hw1: 1255 evaluations in 1 pieces (cut at gaps over 2000 ms): 1255 evaluations and 97 main passes
evaluations between main passes, by card: card 8: 381; the draft's last card: 8

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps |
|---|---:|---:|---:|---|---:|
| d0s | 97 | 0 | 96 | 1: 1, 4: 95 | 91 |

**hw1, all requests: 91 typical steps, 4 draft passes a step, 4 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 105.12 | 103.59 | 102.42 | 113.75 |
| verify_span | 90.16 | 90.08 | 88.97 | 91.22 |
|   verify_gpu (the nine cards) | 86.49 | 86.46 | 85.20 | 87.59 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.67 | 3.59 | 3.56 | 3.65 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 1.04 | 1.04 | 1.03 | 1.06 |
| pass 0: turn | 1.02 | 1.00 | 0.92 | 1.13 |
| pass 1 (draft 1): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 1: gpu | 3.05 | 2.60 | 2.56 | 5.42 |
| pass 1: turn | 0.97 | 0.95 | 0.90 | 1.05 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 2.91 | 2.52 | 2.49 | 3.22 |
| pass 2: turn | 0.91 | 0.90 | 0.86 | 0.95 |
| pass 3 (draft 3): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 3: gpu | 2.90 | 2.53 | 2.49 | 3.20 |
| pass 3: turn (the server before the next verify) | 2.17 | 1.98 | 1.02 | 3.39 |
| the draft's span (all passes) | 14.96 | 13.61 | 12.56 | 23.83 |
|   the draft's GPU time on its last card | 9.90 | 8.68 | 8.59 | 12.15 |
|   the leads (crossings) | 0.00 | 0.00 | 0.00 | 0.00 |
|   the turns of the passes before the last | 2.89 | 2.88 | 2.76 | 3.01 |
|   the draft's GPU time on the embedding card | 0.00 | 0.00 | 0.00 | 0.00 |
| the host's launch of the main pass (inside verify_span) | 3.49 | 3.15 | 3.11 | 3.34 |

verify_gpu per card, median ms: 10.90, 10.17, 10.24, 10.46, 8.55, 10.02, 8.50, 9.89, 7.74

**hw1, d0s: 91 typical steps, 4 draft passes a step, 4 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 105.12 | 103.59 | 102.42 | 113.75 |
| verify_span | 90.16 | 90.08 | 88.97 | 91.22 |
|   verify_gpu (the nine cards) | 86.49 | 86.46 | 85.20 | 87.59 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.67 | 3.59 | 3.56 | 3.65 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 1.04 | 1.04 | 1.03 | 1.06 |
| pass 0: turn | 1.02 | 1.00 | 0.92 | 1.13 |
| pass 1 (draft 1): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 1: gpu | 3.05 | 2.60 | 2.56 | 5.42 |
| pass 1: turn | 0.97 | 0.95 | 0.90 | 1.05 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 2.91 | 2.52 | 2.49 | 3.22 |
| pass 2: turn | 0.91 | 0.90 | 0.86 | 0.95 |
| pass 3 (draft 3): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 3: gpu | 2.90 | 2.53 | 2.49 | 3.20 |
| pass 3: turn (the server before the next verify) | 2.17 | 1.98 | 1.02 | 3.39 |
| the draft's span (all passes) | 14.96 | 13.61 | 12.56 | 23.83 |
|   the draft's GPU time on its last card | 9.90 | 8.68 | 8.59 | 12.15 |
|   the leads (crossings) | 0.00 | 0.00 | 0.00 | 0.00 |
|   the turns of the passes before the last | 2.89 | 2.88 | 2.76 | 3.01 |
|   the draft's GPU time on the embedding card | 0.00 | 0.00 | 0.00 | 0.00 |
| the host's launch of the main pass (inside verify_span) | 3.49 | 3.15 | 3.11 | 3.34 |

verify_gpu per card, median ms: 10.90, 10.17, 10.24, 10.46, 8.55, 10.02, 8.50, 9.89, 7.74

