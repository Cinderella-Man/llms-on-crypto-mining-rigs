gh: 1349 evaluations in 1 pieces (cut at gaps over 2000 ms): 1349 evaluations and 104 main passes
evaluations between main passes, by card: card 8: 412; the draft's last card: 8

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps |
|---|---:|---:|---:|---|---:|
| d0p2 | 104 | 0 | 103 | 4: 103 | 100 |

**gh, all requests: 100 typical steps, 4 draft passes a step, 4 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 115.09 | 114.83 | 113.26 | 116.68 |
| verify_span | 102.44 | 102.37 | 100.81 | 104.09 |
|   verify_gpu (the nine cards) | 98.86 | 98.78 | 97.26 | 100.49 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.58 | 3.57 | 3.55 | 3.60 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 1.05 | 1.05 | 1.03 | 1.07 |
| pass 0: turn | 0.97 | 0.98 | 0.87 | 1.09 |
| pass 1 (draft 1): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 1: gpu | 2.62 | 2.60 | 2.57 | 2.63 |
| pass 1: turn | 0.92 | 0.92 | 0.88 | 0.94 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 2.55 | 2.55 | 2.51 | 2.58 |
| pass 2: turn | 0.87 | 0.86 | 0.85 | 0.89 |
| pass 3 (draft 3): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 3: gpu | 2.56 | 2.55 | 2.52 | 2.59 |
| pass 3: turn (the server before the next verify) | 1.10 | 0.96 | 0.94 | 0.99 |
| the draft's span (all passes) | 12.65 | 12.48 | 12.32 | 12.61 |
|   the draft's GPU time on its last card | 8.79 | 8.76 | 8.67 | 8.83 |
|   the leads (crossings) | 0.00 | 0.00 | 0.00 | 0.00 |
|   the turns of the passes before the last | 2.76 | 2.76 | 2.64 | 2.88 |
|   the draft's GPU time on the embedding card | 0.00 | 0.00 | 0.00 | 0.00 |
| the host's launch of the main pass (inside verify_span) | 3.59 | 3.15 | 3.05 | 3.27 |

verify_gpu per card, median ms: 15.01, 14.13, 12.09, 10.14, 11.52, 11.30, 7.65, 9.40, 7.70

**gh, d0p2: 100 typical steps, 4 draft passes a step, 4 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 115.09 | 114.83 | 113.26 | 116.68 |
| verify_span | 102.44 | 102.37 | 100.81 | 104.09 |
|   verify_gpu (the nine cards) | 98.86 | 98.78 | 97.26 | 100.49 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.58 | 3.57 | 3.55 | 3.60 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 1.05 | 1.05 | 1.03 | 1.07 |
| pass 0: turn | 0.97 | 0.98 | 0.87 | 1.09 |
| pass 1 (draft 1): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 1: gpu | 2.62 | 2.60 | 2.57 | 2.63 |
| pass 1: turn | 0.92 | 0.92 | 0.88 | 0.94 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 2.55 | 2.55 | 2.51 | 2.58 |
| pass 2: turn | 0.87 | 0.86 | 0.85 | 0.89 |
| pass 3 (draft 3): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 3: gpu | 2.56 | 2.55 | 2.52 | 2.59 |
| pass 3: turn (the server before the next verify) | 1.10 | 0.96 | 0.94 | 0.99 |
| the draft's span (all passes) | 12.65 | 12.48 | 12.32 | 12.61 |
|   the draft's GPU time on its last card | 8.79 | 8.76 | 8.67 | 8.83 |
|   the leads (crossings) | 0.00 | 0.00 | 0.00 | 0.00 |
|   the turns of the passes before the last | 2.76 | 2.76 | 2.64 | 2.88 |
|   the draft's GPU time on the embedding card | 0.00 | 0.00 | 0.00 | 0.00 |
| the host's launch of the main pass (inside verify_span) | 3.59 | 3.15 | 3.05 | 3.27 |

verify_gpu per card, median ms: 15.01, 14.13, 12.09, 10.14, 11.52, 11.30, 7.65, 9.40, 7.70

