w5h1: 1130 evaluations in 1 pieces (cut at gaps over 2000 ms): 1130 evaluations and 81 main passes
evaluations between main passes, by card: card 8: 400; the draft's last card: 8

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps |
|---|---:|---:|---:|---|---:|
| d0s | 81 | 0 | 80 | 5: 80 | 76 |

**w5h1, all requests: 76 typical steps, 5 draft passes a step, 5 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 142.77 | 140.04 | 137.72 | 154.00 |
| verify_span | 122.43 | 122.27 | 121.05 | 123.98 |
|   verify_gpu (the nine cards) | 118.08 | 117.93 | 116.69 | 119.63 |
|   verify_rest (crossings, rows back, acceptance, server) | 4.35 | 4.33 | 4.30 | 4.47 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 1.19 | 1.19 | 1.17 | 1.21 |
| pass 0: turn | 1.02 | 1.03 | 0.85 | 1.16 |
| pass 1 (draft 1): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 1: gpu | 3.54 | 2.52 | 2.47 | 6.20 |
| pass 1: turn | 0.96 | 0.95 | 0.90 | 1.02 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 3.26 | 2.44 | 2.41 | 6.13 |
| pass 2: turn | 0.91 | 0.91 | 0.87 | 0.95 |
| pass 3 (draft 3): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 3: gpu | 3.20 | 2.44 | 2.39 | 6.15 |
| pass 3: turn | 0.91 | 0.90 | 0.87 | 0.94 |
| pass 4 (draft 4): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 4: gpu | 3.20 | 2.44 | 2.39 | 6.14 |
| pass 4: turn (the server before the next verify) | 2.14 | 1.82 | 1.21 | 2.63 |
| the draft's span (all passes) | 20.34 | 16.81 | 15.95 | 31.64 |
|   the draft's GPU time on its last card | 14.40 | 11.04 | 10.87 | 25.85 |
|   the leads (crossings) | 0.00 | 0.00 | 0.00 | 0.00 |
|   the turns of the passes before the last | 3.80 | 3.78 | 3.63 | 3.96 |
|   the draft's GPU time on the embedding card | 0.00 | 0.00 | 0.00 | 0.00 |
| the host's launch of the main pass (inside verify_span) | 3.71 | 3.25 | 3.20 | 3.47 |

verify_gpu per card, median ms: 14.83, 12.87, 13.33, 13.38, 10.98, 13.41, 11.05, 13.37, 14.79

**w5h1, d0s: 76 typical steps, 5 draft passes a step, 5 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 142.77 | 140.04 | 137.72 | 154.00 |
| verify_span | 122.43 | 122.27 | 121.05 | 123.98 |
|   verify_gpu (the nine cards) | 118.08 | 117.93 | 116.69 | 119.63 |
|   verify_rest (crossings, rows back, acceptance, server) | 4.35 | 4.33 | 4.30 | 4.47 |
| pass 0 (catch-up): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 0: gpu | 1.19 | 1.19 | 1.17 | 1.21 |
| pass 0: turn | 1.02 | 1.03 | 0.85 | 1.16 |
| pass 1 (draft 1): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 1: gpu | 3.54 | 2.52 | 2.47 | 6.20 |
| pass 1: turn | 0.96 | 0.95 | 0.90 | 1.02 |
| pass 2 (draft 2): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 2: gpu | 3.26 | 2.44 | 2.41 | 6.13 |
| pass 2: turn | 0.91 | 0.91 | 0.87 | 0.95 |
| pass 3 (draft 3): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 3: gpu | 3.20 | 2.44 | 2.39 | 6.15 |
| pass 3: turn | 0.91 | 0.90 | 0.87 | 0.94 |
| pass 4 (draft 4): lead | 0.00 | 0.00 | 0.00 | 0.00 |
| pass 4: gpu | 3.20 | 2.44 | 2.39 | 6.14 |
| pass 4: turn (the server before the next verify) | 2.14 | 1.82 | 1.21 | 2.63 |
| the draft's span (all passes) | 20.34 | 16.81 | 15.95 | 31.64 |
|   the draft's GPU time on its last card | 14.40 | 11.04 | 10.87 | 25.85 |
|   the leads (crossings) | 0.00 | 0.00 | 0.00 | 0.00 |
|   the turns of the passes before the last | 3.80 | 3.78 | 3.63 | 3.96 |
|   the draft's GPU time on the embedding card | 0.00 | 0.00 | 0.00 | 0.00 |
| the host's launch of the main pass (inside verify_span) | 3.71 | 3.25 | 3.20 | 3.47 |

verify_gpu per card, median ms: 14.83, 12.87, 13.33, 13.38, 10.98, 13.41, 11.05, 13.37, 14.79

