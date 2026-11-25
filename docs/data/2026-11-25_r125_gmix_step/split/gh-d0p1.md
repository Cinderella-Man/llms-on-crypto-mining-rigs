gh: 1372 evaluations in 1 pieces (cut at gaps over 2000 ms): 1372 evaluations and 106 main passes
evaluations between main passes, by card: card 8: 417; the draft's last card: 8

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps |
|---|---:|---:|---:|---|---:|
| d0p1 | 106 | 1 | 104 | 4: 103, 5: 1 | 100 |

**gh, all requests: 100 typical steps, 4 draft passes a step, 4 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 117.75 | 116.02 | 113.98 | 125.72 |
| verify_span | 102.26 | 102.23 | 100.71 | 103.95 |
|   verify_gpu (the nine cards) | 98.67 | 98.63 | 97.15 | 100.32 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.59 | 3.57 | 3.54 | 3.63 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 1.05 | 1.05 | 1.04 | 1.07 |
| pass 0: turn | 0.99 | 0.98 | 0.89 | 1.10 |
| pass 1 (draft 1): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 1: gpu | 3.29 | 2.59 | 2.55 | 6.39 |
| pass 1: turn | 0.96 | 0.96 | 0.91 | 1.02 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 3.14 | 2.53 | 2.49 | 6.37 |
| pass 2: turn | 0.91 | 0.90 | 0.87 | 0.94 |
| pass 3 (draft 3): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 3: gpu | 3.09 | 2.52 | 2.49 | 6.35 |
| pass 3: turn (the server before the next verify) | 2.06 | 1.78 | 1.00 | 3.41 |
| the draft's span (all passes) | 15.50 | 13.36 | 12.59 | 24.61 |
|   the draft's GPU time on its last card | 10.58 | 8.69 | 8.60 | 20.22 |
|   the leads (crossings) | 0.00 | 0.00 | 0.00 | 0.00 |
|   the turns of the passes before the last | 2.85 | 2.84 | 2.73 | 2.97 |
|   the draft's GPU time on the embedding card | 0.00 | 0.00 | 0.00 | 0.00 |
| the host's launch of the main pass (inside verify_span) | 3.44 | 3.08 | 3.03 | 3.19 |

verify_gpu per card, median ms: 14.89, 14.14, 12.06, 10.14, 11.55, 11.33, 7.66, 9.43, 7.65

**gh, d0p1: 100 typical steps, 4 draft passes a step, 4 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 117.75 | 116.02 | 113.98 | 125.72 |
| verify_span | 102.26 | 102.23 | 100.71 | 103.95 |
|   verify_gpu (the nine cards) | 98.67 | 98.63 | 97.15 | 100.32 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.59 | 3.57 | 3.54 | 3.63 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 1.05 | 1.05 | 1.04 | 1.07 |
| pass 0: turn | 0.99 | 0.98 | 0.89 | 1.10 |
| pass 1 (draft 1): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 1: gpu | 3.29 | 2.59 | 2.55 | 6.39 |
| pass 1: turn | 0.96 | 0.96 | 0.91 | 1.02 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 3.14 | 2.53 | 2.49 | 6.37 |
| pass 2: turn | 0.91 | 0.90 | 0.87 | 0.94 |
| pass 3 (draft 3): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 3: gpu | 3.09 | 2.52 | 2.49 | 6.35 |
| pass 3: turn (the server before the next verify) | 2.06 | 1.78 | 1.00 | 3.41 |
| the draft's span (all passes) | 15.50 | 13.36 | 12.59 | 24.61 |
|   the draft's GPU time on its last card | 10.58 | 8.69 | 8.60 | 20.22 |
|   the leads (crossings) | 0.00 | 0.00 | 0.00 | 0.00 |
|   the turns of the passes before the last | 2.85 | 2.84 | 2.73 | 2.97 |
|   the draft's GPU time on the embedding card | 0.00 | 0.00 | 0.00 | 0.00 |
| the host's launch of the main pass (inside verify_span) | 3.44 | 3.08 | 3.03 | 3.19 |

verify_gpu per card, median ms: 14.89, 14.14, 12.06, 10.14, 11.55, 11.33, 7.66, 9.43, 7.65

