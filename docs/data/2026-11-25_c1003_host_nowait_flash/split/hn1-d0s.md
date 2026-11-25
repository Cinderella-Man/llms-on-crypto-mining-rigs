hn1: 1255 evaluations in 1 pieces (cut at gaps over 2000 ms): 1255 evaluations and 97 main passes
evaluations between main passes, by card: card 8: 381; the draft's last card: 8

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps |
|---|---:|---:|---:|---|---:|
| d0s | 97 | 0 | 96 | 1: 1, 4: 95 | 91 |

**hn1, all requests: 91 typical steps, 4 draft passes a step, 4 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 105.21 | 103.58 | 102.40 | 113.66 |
| verify_span | 89.99 | 89.98 | 88.87 | 91.12 |
|   verify_gpu (the nine cards) | 86.39 | 86.38 | 85.26 | 87.50 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.60 | 3.59 | 3.57 | 3.63 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 1.04 | 1.04 | 1.03 | 1.06 |
| pass 0: turn | 1.02 | 1.02 | 0.94 | 1.14 |
| pass 1 (draft 1): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 1: gpu | 3.05 | 2.59 | 2.55 | 5.54 |
| pass 1: turn | 0.97 | 0.95 | 0.91 | 1.07 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 2.90 | 2.52 | 2.49 | 3.22 |
| pass 2: turn | 0.91 | 0.90 | 0.86 | 0.96 |
| pass 3 (draft 3): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 3: gpu | 2.89 | 2.52 | 2.49 | 3.19 |
| pass 3: turn (the server before the next verify) | 2.42 | 1.98 | 1.01 | 3.95 |
| the draft's span (all passes) | 15.22 | 13.66 | 12.61 | 23.90 |
|   the draft's GPU time on its last card | 9.88 | 8.67 | 8.59 | 12.29 |
|   the leads (crossings) | 0.00 | 0.00 | 0.00 | 0.00 |
|   the turns of the passes before the last | 2.91 | 2.88 | 2.78 | 3.07 |
|   the draft's GPU time on the embedding card | 0.00 | 0.00 | 0.00 | 0.00 |
| the host's launch of the main pass (inside verify_span) | 3.46 | 3.15 | 3.11 | 3.29 |

verify_gpu per card, median ms: 10.85, 10.17, 10.24, 10.46, 8.54, 9.98, 8.49, 9.91, 7.73

**hn1, d0s: 91 typical steps, 4 draft passes a step, 4 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 105.21 | 103.58 | 102.40 | 113.66 |
| verify_span | 89.99 | 89.98 | 88.87 | 91.12 |
|   verify_gpu (the nine cards) | 86.39 | 86.38 | 85.26 | 87.50 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.60 | 3.59 | 3.57 | 3.63 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 1.04 | 1.04 | 1.03 | 1.06 |
| pass 0: turn | 1.02 | 1.02 | 0.94 | 1.14 |
| pass 1 (draft 1): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 1: gpu | 3.05 | 2.59 | 2.55 | 5.54 |
| pass 1: turn | 0.97 | 0.95 | 0.91 | 1.07 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 2.90 | 2.52 | 2.49 | 3.22 |
| pass 2: turn | 0.91 | 0.90 | 0.86 | 0.96 |
| pass 3 (draft 3): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 3: gpu | 2.89 | 2.52 | 2.49 | 3.19 |
| pass 3: turn (the server before the next verify) | 2.42 | 1.98 | 1.01 | 3.95 |
| the draft's span (all passes) | 15.22 | 13.66 | 12.61 | 23.90 |
|   the draft's GPU time on its last card | 9.88 | 8.67 | 8.59 | 12.29 |
|   the leads (crossings) | 0.00 | 0.00 | 0.00 | 0.00 |
|   the turns of the passes before the last | 2.91 | 2.88 | 2.78 | 3.07 |
|   the draft's GPU time on the embedding card | 0.00 | 0.00 | 0.00 | 0.00 |
| the host's launch of the main pass (inside verify_span) | 3.46 | 3.15 | 3.11 | 3.29 |

verify_gpu per card, median ms: 10.85, 10.17, 10.24, 10.46, 8.54, 9.98, 8.49, 9.91, 7.73

