fka250: 2890 evaluations in 4 pieces (cut at gaps over 2000 ms): 9 evaluations and 1 main passes, 2 evaluations and 0 main passes, 1485 evaluations and 89 main passes, 1394 evaluations and 84 main passes
evaluations between main passes, by card: card 4: 655, card 8: 655; the draft's last card: 8

| request | main passes | prompt passes | generation steps | by draft passes a step | typical steps |
|---|---:|---:|---:|---|---:|
| greedy | 89 | 4 | 84 | 3: 1, 4: 83 | 80 |
| long | 84 | 6 | 77 | 4: 76, 8: 1 | 74 |

**fka250, all requests: 154 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 114.12 | 115.32 | 106.48 | 122.88 |
| verify_span | 95.25 | 93.63 | 90.77 | 99.63 |
|   verify_gpu (the nine cards) | 91.63 | 90.15 | 87.28 | 95.91 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.62 | 3.66 | 3.42 | 3.82 |
| pass 0 (catch-up): lead | 0.54 | 0.51 | 0.51 | 0.52 |
| pass 0: gpu | 3.61 | 3.43 | 3.28 | 3.91 |
| pass 0: turn | -1.81 | -2.26 | -2.51 | 0.32 |
| pass 1 (draft 1): lead | 2.42 | 2.75 | 0.51 | 3.00 |
| pass 1: gpu | 3.33 | 2.74 | 2.49 | 6.32 |
| pass 1: turn | 0.60 | 0.60 | 0.54 | 0.64 |
| pass 2 (draft 2): lead | 0.51 | 0.51 | 0.50 | 0.53 |
| pass 2: gpu | 3.36 | 2.76 | 2.49 | 6.29 |
| pass 2: turn | 0.56 | 0.55 | 0.52 | 0.59 |
| pass 3 (draft 3): lead | 0.51 | 0.51 | 0.50 | 0.52 |
| pass 3: gpu | 3.27 | 2.75 | 2.49 | 6.28 |
| pass 3: turn (the server before the next verify) | 1.96 | 1.78 | 1.11 | 2.63 |
| the draft's span (all passes) | 18.87 | 16.92 | 15.40 | 28.15 |
|   the draft's GPU time on its last card | 13.57 | 12.13 | 10.78 | 22.27 |
|   the leads (crossings) | 3.99 | 4.28 | 2.04 | 4.55 |
|   the turns of the passes before the last | -0.65 | -1.13 | -1.33 | 1.54 |
|   the draft's GPU time on the embedding card | 0.09 | 0.09 | 0.08 | 0.09 |
| the host's launch of the main pass (inside verify_span) | 3.81 | 3.20 | 3.12 | 3.34 |

verify_gpu per card, median ms: 11.14, 10.83, 11.04, 11.07, 9.10, 10.65, 8.95, 10.33, 7.35

**fka250, greedy: 80 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 110.01 | 107.94 | 106.26 | 119.35 |
| verify_span | 91.77 | 91.61 | 90.46 | 92.79 |
|   verify_gpu (the nine cards) | 88.27 | 88.12 | 87.02 | 89.34 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.50 | 3.44 | 3.42 | 3.82 |
| pass 0 (catch-up): lead | 0.51 | 0.51 | 0.50 | 0.52 |
| pass 0: gpu | 3.34 | 3.33 | 3.27 | 3.39 |
| pass 0: turn | -1.68 | -2.21 | -2.28 | 0.31 |
| pass 1 (draft 1): lead | 2.30 | 2.71 | 0.51 | 2.78 |
| pass 1: gpu | 3.24 | 2.51 | 2.49 | 6.29 |
| pass 1: turn | 0.60 | 0.59 | 0.51 | 0.65 |
| pass 2 (draft 2): lead | 0.51 | 0.51 | 0.50 | 0.53 |
| pass 2: gpu | 3.26 | 2.51 | 2.49 | 6.26 |
| pass 2: turn | 0.54 | 0.54 | 0.52 | 0.56 |
| pass 3 (draft 3): lead | 0.50 | 0.50 | 0.50 | 0.51 |
| pass 3: gpu | 3.17 | 2.50 | 2.48 | 6.25 |
| pass 3: turn (the server before the next verify) | 1.94 | 1.81 | 1.11 | 2.63 |
| the draft's span (all passes) | 18.24 | 15.95 | 15.14 | 27.18 |
|   the draft's GPU time on its last card | 13.01 | 10.85 | 10.76 | 22.17 |
|   the leads (crossings) | 3.83 | 4.23 | 2.04 | 4.31 |
|   the turns of the passes before the last | -0.54 | -1.07 | -1.16 | 1.56 |
|   the draft's GPU time on the embedding card | 0.09 | 0.08 | 0.08 | 0.09 |
| the host's launch of the main pass (inside verify_span) | 3.75 | 3.17 | 3.11 | 3.29 |

verify_gpu per card, median ms: 10.93, 10.60, 10.77, 10.83, 8.78, 10.29, 8.64, 9.99, 7.23

**fka250, long: 74 typical steps, 4 draft passes a step, 8 draft evaluations**

| part | mean, ms | median | p10 | p90 |
|---|---:|---:|---:|---:|
| step | 118.56 | 116.61 | 114.98 | 127.84 |
| verify_span | 99.02 | 98.83 | 97.86 | 100.14 |
|   verify_gpu (the nine cards) | 95.27 | 95.06 | 94.19 | 96.24 |
|   verify_rest (crossings, rows back, acceptance, server) | 3.75 | 3.70 | 3.66 | 3.74 |
| pass 0 (catch-up): lead | 0.58 | 0.51 | 0.51 | 0.53 |
| pass 0: gpu | 3.89 | 3.88 | 3.80 | 3.96 |
| pass 0: turn | -1.96 | -2.45 | -2.53 | 0.32 |
| pass 1 (draft 1): lead | 2.56 | 2.94 | 0.51 | 3.02 |
| pass 1: gpu | 3.44 | 2.79 | 2.74 | 6.47 |
| pass 1: turn | 0.60 | 0.60 | 0.55 | 0.63 |
| pass 2 (draft 2): lead | 0.51 | 0.51 | 0.50 | 0.53 |
| pass 2: gpu | 3.46 | 2.79 | 2.74 | 6.45 |
| pass 2: turn | 0.58 | 0.58 | 0.53 | 0.60 |
| pass 3 (draft 3): lead | 0.51 | 0.51 | 0.50 | 0.52 |
| pass 3: gpu | 3.39 | 2.80 | 2.74 | 6.48 |
| pass 3: turn (the server before the next verify) | 1.99 | 1.75 | 1.11 | 2.34 |
| the draft's span (all passes) | 19.55 | 17.33 | 16.51 | 28.96 |
|   the draft's GPU time on its last card | 14.18 | 12.22 | 12.10 | 23.34 |
|   the leads (crossings) | 4.16 | 4.49 | 2.05 | 4.59 |
|   the turns of the passes before the last | -0.77 | -1.28 | -1.37 | 1.53 |
|   the draft's GPU time on the embedding card | 0.09 | 0.09 | 0.08 | 0.09 |
| the host's launch of the main pass (inside verify_span) | 3.88 | 3.21 | 3.18 | 3.35 |

verify_gpu per card, median ms: 11.53, 11.76, 11.30, 11.91, 9.38, 10.88, 9.78, 10.65, 7.87

