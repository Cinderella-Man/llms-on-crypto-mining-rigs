trace 671 lines in 12 pieces (cut at gaps over 2000 ms); requests in the CSV that read a prompt: 12

| request | prompt lines | generation lines | drafted (trace) | accepted (trace) | generation span, ms | prompt's last line end to the first generation line, ms | server gen_ms | server drafted | server accepted | main graph built/reused | draft graph built/reused | prompt lines: main graph built/reused | main passes' GPU time, ms (all cards) | draft's GPU time, ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| warmup0@0 | 1 | 15 | 56 | 16 | 2992.03 | 0.13 | 2992.039 | 56 | 16 | 2/0 | 21/50 | 2/0 | 2094.49 | 245.32 |
| decode1@0 | 1 | 89 | 356 | 166 | 12905.36 | 0.14 | 12905.341 | 356 | 166 | 0/0 | 102/343 | 2/0 | 10724.87 | 1272.54 |
| decode2@0 | 1 | 86 | 344 | 169 | 12003.98 | 0.08 | 12003.919 | 344 | 169 | 0/0 | 90/340 | 1/0 | 10240.64 | 963.16 |
| samples@0 | 1 | 80 | 318 | 175 | 11791.96 | 0.14 | 11791.976 | 318 | 175 | 1/0 | 94/306 | 1/0 | 9659.08 | 1157.60 |
| prime@4096 | 1 | 0 | 0 | 0 | 0.00 | nan | 0.001 | 0 | 0 | 0/0 | 0/0 | 9/0 | 0.00 | 0.00 |
| decode1@4096 | 1 | 67 | 266 | 188 | 10813.57 | 0.16 | 10813.295 | 266 | 188 | 1/0 | 71/262 | 2/0 | 9140.73 | 894.84 |
| decode2@4096 | 1 | 67 | 264 | 188 | 10566.33 | 0.08 | 10565.994 | 264 | 188 | 1/0 | 67/264 | 1/0 | 9061.95 | 816.80 |
| samples@4096 | 1 | 71 | 281 | 184 | 11233.94 | 0.13 | 11233.641 | 281 | 184 | 1/0 | 76/279 | 1/0 | 9514.37 | 948.36 |
| prime@8192 | 2 | 0 | 0 | 0 | 0.00 | nan | 0.001 | 0 | 0 | 0/0 | 0/0 | 17/0 | 0.00 | 0.00 |
| decode1@8192 | 1 | 59 | 233 | 196 | 10340.68 | 0.15 | 10340.144 | 233 | 196 | 1/0 | 63/229 | 2/0 | 8864.75 | 773.67 |
| decode2@8192 | 1 | 58 | 229 | 197 | 9840.50 | 0.10 | 9839.916 | 229 | 197 | 1/0 | 58/229 | 1/0 | 8473.91 | 723.03 |
| samples@8192 | 1 | 66 | 262 | 189 | 11640.75 | 0.09 | 11640.146 | 262 | 189 | 1/0 | 71/259 | 1/0 | 9660.83 | 929.27 |

### Depth 0: pass 1 against pass 2

- generation lines: 89 and 86; drafted and accepted per line equal in 39 of 86 paired lines
- the span, pass 1 − pass 2, ms: all 86 paired steps 484.46; steps 0-2 -8.99 (-2.56, -5.13, -1.30); steps 3 on 493.45 (median 1.11 a step); the server's gen_ms 12905.341 − 12003.919 = 901.42
- the span, median a step: pass 1 140.45, pass 2 139.22 (steps 3 on)

| stage | pass 1, ms (sum) | pass 2, ms (sum) | difference | of which steps 0-2 | median difference a step, steps 3 on |
|---|---:|---:|---:|---:|---:|
| loop | 1.07 | 1.04 | 0.03 | 0.00 | 0.000 |
| draft | 1507.78 | 1210.20 | 297.58 | -3.49 | 0.230 |
| checkpoints | 0.22 | 0.20 | 0.03 | 0.00 | 0.000 |
| batch | 0.76 | 0.64 | 0.12 | 0.00 | 0.001 |
| verify | 10820.51 | 10637.72 | 182.79 | -5.54 | 0.552 |
| catch-up | 88.13 | 86.43 | 1.70 | 0.04 | -0.001 |
| acceptance | 43.37 | 43.45 | -0.08 | -0.01 | 0.001 |
| between | 26.59 | 24.31 | 2.28 | 0.01 | 0.002 |
| verify: waits for the cards | 10316.00 | 10174.77 | 141.23 | 0.65 | 0.239 |
| main passes' GPU time (all cards) | 10369.97 | 10240.64 | 129.32 | -8.65 | 0.163 |
| the draft's GPU time | 1239.25 | 963.16 | 276.09 | -5.86 | -0.011 |

The ten mark pairs with the largest difference (summed over the paired steps):

| mark pair | pass 1, ms | pass 2, ms | difference |
|---|---:|---:|---:|
| `v.sync#0 -> v.sync.end` | 10316.00 | 10174.77 | 141.23 |
| `d.sync#1 -> d.sync.end` | 58.73 | 2.17 | 56.56 |
| `d.sync#15 -> d.sync.end` | 56.14 | 1.56 | 54.57 |
| `d.sync#8 -> d.sync.end` | 55.95 | 1.58 | 54.36 |
| `d.sync#22 -> d.sync.end` | 51.53 | 1.54 | 49.99 |
| `v.ub.in#0 -> v.ub.comp` | 477.69 | 437.43 | 40.25 |
| `mtp.dp.ret#0 -> d.sync` | 18.74 | 5.08 | 13.65 |
| `mtp.dp.ret#1 -> d.sync` | 18.34 | 4.72 | 13.62 |
| `mtp.dp.ret#2 -> d.sync` | 17.16 | 4.52 | 12.64 |
| `s.vp.end#0 -> d.sync` | 15.75 | 5.10 | 10.65 |

Step by step, pass 1 − pass 2, ms: -2.6, -5.1, -1.3, 0.7, 1.5, 1.2, 2.0, 0.7, 16.2, 15.4, 4.2, 0.6, 0.3, 0.4, 0.2, 0.9, 0.1, 0.2, 0.6, 0.4, 0.3, 0.0, 0.9, 1.4, 15.7, 15.4, 6.3, 2.6, -3.8, -0.2, 16.1, 17.6, 4.7, -0.5, 1.3, 0.8, 16.1, 15.1, 3.9, 0.5, 1.1, 3.3, -1.5, 15.9, 13.4, 5.3, 1.1, -1.9, 11.3, 17.5, 16.5, 1.8, -0.9, -0.4, 0.3, 16.0, 17.0, 18.5, 5.0, 2.6, 0.0, -0.1, -0.6, -0.7, 10.6, 0.1, -16.6, -6.9, 0.6, 148.0, 8.2, 0.1, 0.9, 1.6, 20.4, 16.1, 5.1, -1.0, -2.2, 0.0, 2.9, 3.0, -1.6, 0.2, 0.5, 2.7

### Depth 4096: pass 1 against pass 2

- generation lines: 67 and 67; drafted and accepted per line equal in 65 of 67 paired lines
- the span, pass 1 − pass 2, ms: all 67 paired steps 247.24; steps 0-2 133.13 (132.33, 3.99, -3.18); steps 3 on 114.11 (median 0.83 a step); the server's gen_ms 10813.295 − 10565.994 = 247.30
- the span, median a step: pass 1 156.33, pass 2 155.22 (steps 3 on)

| stage | pass 1, ms (sum) | pass 2, ms (sum) | difference | of which steps 0-2 | median difference a step, steps 3 on |
|---|---:|---:|---:|---:|---:|
| loop | 1.56 | 1.59 | -0.03 | 0.00 | -0.001 |
| draft | 1135.81 | 1030.38 | 105.43 | 16.98 | 0.171 |
| checkpoints | 0.16 | 0.15 | 0.00 | 0.00 | 0.000 |
| batch | 0.60 | 0.51 | 0.09 | 0.00 | 0.002 |
| verify | 9525.00 | 9381.83 | 143.17 | 115.80 | 0.557 |
| catch-up | 93.65 | 94.50 | -0.85 | 0.31 | -0.004 |
| acceptance | 36.70 | 36.49 | 0.21 | 0.00 | 0.001 |
| between | 20.09 | 20.87 | -0.78 | 0.03 | -0.010 |
| verify: waits for the cards | 9082.44 | 8849.98 | 232.46 | 86.22 | -0.070 |
| main passes' GPU time (all cards) | 9140.73 | 9061.95 | 78.78 | 102.19 | -0.048 |
| the draft's GPU time | 894.84 | 816.80 | 78.04 | 5.41 | -0.032 |

The ten mark pairs with the largest difference (summed over the paired steps):

| mark pair | pass 1, ms | pass 2, ms | difference |
|---|---:|---:|---:|
| `v.sync#0 -> v.sync.end` | 9082.44 | 8849.98 | 232.46 |
| `v.ub.in#0 -> v.ub.comp` | 404.91 | 496.55 | -91.64 |
| `d.sync#1 -> d.sync.end` | 16.91 | 1.71 | 15.21 |
| `d.sync#8 -> d.sync.end` | 14.07 | 1.20 | 12.87 |
| `d.sync#15 -> d.sync.end` | 14.05 | 1.24 | 12.81 |
| `d.sync#22 -> d.sync.end` | 12.09 | 1.18 | 10.90 |
| `p.ub.apply#0 -> p.ub.build` | 25.90 | 15.46 | 10.44 |
| `mtp.dp.ret#0 -> d.sync` | 13.91 | 3.89 | 10.02 |
| `mtp.dp.ret#1 -> d.sync` | 13.19 | 3.66 | 9.53 |
| `mtp.dp.ret#2 -> d.sync` | 11.74 | 3.40 | 8.33 |

Step by step, pass 1 − pass 2, ms: 132.3, 4.0, -3.2, 0.2, 1.2, 1.4, 1.2, 21.0, 15.3, 4.6, 0.3, 0.2, 0.7, 1.8, 0.3, 10.2, 0.9, 1.0, 0.8, 1.1, 1.0, 0.5, 0.9, 0.2, 1.1, -1.3, 1.2, 0.6, 1.1, 1.0, 0.6, 1.0, 6.0, 0.9, 0.4, 1.2, 0.9, 0.7, 0.0, 0.2, 0.2, -0.1, -0.2, 3.5, 0.1, -0.5, 0.7, 0.2, -0.3, -0.5, -0.1, 0.9, -0.1, 0.1, 0.7, -0.1, 0.1, 1.0, 0.8, 1.7, 16.0, 14.4, 4.1, -0.3, 1.0, 1.4, -11.5

### Depth 8192: pass 1 against pass 2

- generation lines: 59 and 58; drafted and accepted per line equal in 43 of 58 paired lines
- the span, pass 1 − pass 2, ms: all 58 paired steps 284.40; steps 0-2 137.76 (130.29, 8.32, -0.85); steps 3 on 146.64 (median 0.50 a step); the server's gen_ms 10340.144 − 9839.916 = 500.23
- the span, median a step: pass 1 170.92, pass 2 170.55 (steps 3 on)

| stage | pass 1, ms (sum) | pass 2, ms (sum) | difference | of which steps 0-2 | median difference a step, steps 3 on |
|---|---:|---:|---:|---:|---:|
| loop | 1.95 | 1.97 | -0.02 | 0.01 | -0.000 |
| draft | 993.97 | 934.24 | 59.73 | 4.74 | 0.119 |
| checkpoints | 0.13 | 0.13 | 0.00 | 0.00 | 0.000 |
| batch | 0.54 | 0.52 | 0.02 | 0.00 | 0.001 |
| verify | 8972.34 | 8747.29 | 225.05 | 132.07 | 0.353 |
| catch-up | 107.66 | 106.93 | 0.74 | 0.93 | -0.007 |
| acceptance | 30.27 | 30.99 | -0.73 | 0.03 | 0.002 |
| between | 18.05 | 18.44 | -0.39 | -0.01 | -0.011 |
| verify: waits for the cards | 8637.93 | 8436.73 | 201.20 | 105.28 | 0.111 |
| main passes' GPU time (all cards) | 8676.56 | 8473.91 | 202.65 | 122.16 | 0.178 |
| the draft's GPU time | 769.09 | 723.03 | 46.06 | 0.79 | 0.018 |

The ten mark pairs with the largest difference (summed over the paired steps):

| mark pair | pass 1, ms | pass 2, ms | difference |
|---|---:|---:|---:|
| `v.sync#0 -> v.sync.end` | 8637.93 | 8436.73 | 201.20 |
| `v.ub.in#0 -> v.ub.comp` | 315.80 | 280.90 | 34.90 |
| `v.ub.apply#0 -> v.ub.build` | 0.00 | 11.05 | -11.05 |
| `d.sync#1 -> d.sync.end` | 10.49 | 1.49 | 9.00 |
| `d.sync#15 -> d.sync.end` | 7.56 | 1.04 | 6.52 |
| `d.sync#8 -> d.sync.end` | 7.47 | 1.04 | 6.43 |
| `mtp.dp.ret#0 -> d.sync` | 8.61 | 3.46 | 5.15 |
| `mtp.dp.ret#1 -> d.sync` | 8.31 | 3.18 | 5.13 |
| `s.vp.end#0 -> d.sync` | 8.51 | 3.53 | 4.98 |
| `d.sync#22 -> d.sync.end` | 5.93 | 1.02 | 4.91 |

Step by step, pass 1 − pass 2, ms: 130.3, 8.3, -0.9, 0.3, 17.7, 20.3, 2.5, 0.5, 2.9, -1.4, 2.8, -1.0, 0.7, 2.3, 0.2, -1.6, 1.1, -0.0, 2.8, -1.1, 0.5, -0.4, 0.6, -0.7, 1.4, 0.9, 0.3, 1.6, -0.5, -0.1, 0.6, 4.9, 2.7, 1.6, 2.2, -1.1, 1.4, 0.5, 1.2, -1.5, 1.7, 1.2, -0.6, -0.0, -0.5, 0.1, -0.1, 0.6, -1.7, 1.6, 0.3, 2.1, -1.7, 1.8, -1.1, 0.2, -0.9, 78.5

