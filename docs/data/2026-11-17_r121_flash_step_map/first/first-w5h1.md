trace 671 lines in 12 pieces (cut at gaps over 2000 ms); requests in the CSV that read a prompt: 12

| request | prompt lines | generation lines | drafted (trace) | accepted (trace) | generation span, ms | prompt's last line end to the first generation line, ms | server gen_ms | server drafted | server accepted | main graph built/reused | draft graph built/reused | prompt lines: main graph built/reused | main passes' GPU time, ms (all cards) | draft's GPU time, ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| warmup0@0 | 1 | 15 | 56 | 16 | 2893.46 | 0.10 | 2893.44 | 56 | 16 | 2/0 | 21/50 | 2/0 | 2119.44 | 246.13 |
| decode1@0 | 1 | 89 | 356 | 166 | 12893.26 | 0.13 | 12893.232 | 356 | 166 | 0/0 | 102/343 | 2/0 | 10716.47 | 1268.16 |
| decode2@0 | 1 | 86 | 344 | 169 | 11993.88 | 0.08 | 11993.806 | 344 | 169 | 0/0 | 90/340 | 1/0 | 10229.89 | 959.76 |
| samples@0 | 1 | 80 | 318 | 175 | 11655.01 | 0.13 | 11655.025 | 318 | 175 | 1/0 | 94/306 | 1/0 | 9648.39 | 1156.28 |
| prime@4096 | 1 | 0 | 0 | 0 | 0.00 | nan | 0.001 | 0 | 0 | 0/0 | 0/0 | 9/0 | 0.00 | 0.00 |
| decode1@4096 | 1 | 67 | 266 | 188 | 10798.44 | 0.15 | 10798.139 | 266 | 188 | 1/0 | 71/262 | 2/0 | 9140.90 | 895.45 |
| decode2@4096 | 1 | 67 | 264 | 188 | 10574.64 | 0.09 | 10574.328 | 264 | 188 | 1/0 | 67/264 | 1/0 | 9064.21 | 815.29 |
| samples@4096 | 1 | 71 | 281 | 184 | 11227.98 | 0.13 | 11227.67 | 281 | 184 | 1/0 | 76/279 | 1/0 | 9514.87 | 950.17 |
| prime@8192 | 2 | 0 | 0 | 0 | 0.00 | nan | 0.001 | 0 | 0 | 0/0 | 0/0 | 17/0 | 0.00 | 0.00 |
| decode1@8192 | 1 | 59 | 233 | 196 | 10346.25 | 0.14 | 10345.72 | 233 | 196 | 1/0 | 63/229 | 2/0 | 8858.29 | 774.43 |
| decode2@8192 | 1 | 58 | 229 | 197 | 9974.18 | 0.09 | 9973.57 | 229 | 197 | 1/0 | 58/229 | 1/0 | 8469.80 | 722.69 |
| samples@8192 | 1 | 66 | 262 | 189 | 11591.89 | 0.09 | 11591.291 | 262 | 189 | 1/0 | 71/259 | 1/0 | 9666.92 | 929.67 |

### Depth 0: pass 1 against pass 2

- generation lines: 89 and 86; drafted and accepted per line equal in 39 of 86 paired lines
- the span, pass 1 − pass 2, ms: all 86 paired steps 482.49; steps 0-2 -5.47 (-3.35, -1.40, -0.72); steps 3 on 487.96 (median 1.53 a step); the server's gen_ms 12893.232 − 11993.806 = 899.43
- the span, median a step: pass 1 140.17, pass 2 139.00 (steps 3 on)

| stage | pass 1, ms (sum) | pass 2, ms (sum) | difference | of which steps 0-2 | median difference a step, steps 3 on |
|---|---:|---:|---:|---:|---:|
| loop | 1.04 | 1.04 | -0.00 | 0.00 | 0.000 |
| draft | 1503.50 | 1206.68 | 296.82 | -4.41 | 0.258 |
| checkpoints | 0.26 | 0.21 | 0.05 | -0.00 | 0.000 |
| batch | 0.77 | 0.62 | 0.15 | 0.00 | 0.002 |
| verify | 10814.10 | 10630.05 | 184.05 | -0.99 | 0.550 |
| catch-up | 87.97 | 86.54 | 1.43 | -0.04 | 0.005 |
| acceptance | 44.15 | 44.46 | -0.31 | -0.07 | -0.005 |
| between | 24.59 | 24.29 | 0.30 | 0.05 | -0.003 |
| verify: waits for the cards | 10308.11 | 10165.44 | 142.67 | 1.20 | 0.256 |
| main passes' GPU time (all cards) | 10361.97 | 10229.89 | 132.08 | -8.15 | 0.136 |
| the draft's GPU time | 1235.00 | 959.76 | 275.24 | -5.86 | -0.028 |

The ten mark pairs with the largest difference (summed over the paired steps):

| mark pair | pass 1, ms | pass 2, ms | difference |
|---|---:|---:|---:|
| `v.sync#0 -> v.sync.end` | 10308.11 | 10165.44 | 142.67 |
| `d.sync#1 -> d.sync.end` | 58.54 | 2.22 | 56.32 |
| `d.sync#15 -> d.sync.end` | 56.01 | 1.68 | 54.32 |
| `d.sync#8 -> d.sync.end` | 55.88 | 1.69 | 54.19 |
| `d.sync#22 -> d.sync.end` | 51.24 | 1.66 | 49.58 |
| `v.ub.in#0 -> v.ub.comp` | 479.63 | 439.41 | 40.22 |
| `mtp.dp.ret#0 -> d.sync` | 18.93 | 5.47 | 13.47 |
| `mtp.dp.ret#1 -> d.sync` | 17.78 | 5.07 | 12.70 |
| `mtp.dp.ret#2 -> d.sync` | 17.35 | 4.79 | 12.57 |
| `d.sync#0 -> d.sync.end` | 228.07 | 216.65 | 11.43 |

Step by step, pass 1 − pass 2, ms: -3.3, -1.4, -0.7, 1.5, 0.7, 1.5, 1.8, 0.2, 16.2, 15.0, 4.1, 0.4, 2.7, 0.2, 0.3, 1.6, 0.2, 0.2, 0.3, -0.2, 0.7, -3.0, 2.7, 3.0, 15.8, 14.9, 6.0, 2.1, -3.3, 0.2, 15.3, 17.2, 4.9, -0.6, 1.0, 0.6, 16.0, 16.8, 4.2, 0.5, 0.9, 3.5, -0.8, 16.0, 13.4, 5.8, 1.2, -0.5, 2.8, 18.5, 16.5, 2.1, 0.0, -1.0, 0.3, 16.3, 17.3, 18.2, 5.5, 2.8, 0.6, 0.4, -3.1, -0.0, 2.6, 0.2, -17.3, -6.7, 0.8, 149.2, 8.3, 0.3, 4.5, 1.5, 20.4, 16.1, 5.7, -1.3, -2.1, -0.1, 2.5, 2.9, -1.6, 0.1, 0.7, 2.6

### Depth 4096: pass 1 against pass 2

- generation lines: 67 and 67; drafted and accepted per line equal in 65 of 67 paired lines
- the span, pass 1 − pass 2, ms: all 67 paired steps 223.80; steps 0-2 125.84 (132.26, -3.49, -2.93); steps 3 on 97.96 (median 0.58 a step); the server's gen_ms 10798.139 − 10574.328 = 223.81
- the span, median a step: pass 1 156.16, pass 2 155.43 (steps 3 on)

| stage | pass 1, ms (sum) | pass 2, ms (sum) | difference | of which steps 0-2 | median difference a step, steps 3 on |
|---|---:|---:|---:|---:|---:|
| loop | 1.58 | 1.66 | -0.09 | 0.01 | -0.001 |
| draft | 1135.33 | 1029.01 | 106.33 | 20.46 | 0.158 |
| checkpoints | 0.16 | 0.16 | 0.00 | -0.00 | 0.000 |
| batch | 0.59 | 0.52 | 0.07 | 0.02 | 0.001 |
| verify | 9509.73 | 9390.51 | 119.21 | 105.30 | 0.365 |
| catch-up | 94.02 | 95.36 | -1.35 | 0.01 | -0.008 |
| acceptance | 36.94 | 36.87 | 0.06 | 0.03 | -0.003 |
| between | 20.10 | 20.54 | -0.44 | 0.02 | -0.008 |
| verify: waits for the cards | 9083.53 | 8849.26 | 234.27 | 88.14 | -0.018 |
| main passes' GPU time (all cards) | 9140.90 | 9064.21 | 76.69 | 101.06 | -0.025 |
| the draft's GPU time | 895.45 | 815.29 | 80.16 | 5.68 | 0.000 |

The ten mark pairs with the largest difference (summed over the paired steps):

| mark pair | pass 1, ms | pass 2, ms | difference |
|---|---:|---:|---:|
| `v.sync#0 -> v.sync.end` | 9083.53 | 8849.26 | 234.27 |
| `v.ub.in#0 -> v.ub.comp` | 389.16 | 505.26 | -116.10 |
| `d.sync#1 -> d.sync.end` | 16.95 | 2.23 | 14.72 |
| `p.ub.apply#0 -> p.ub.build` | 25.64 | 12.27 | 13.37 |
| `d.sync#8 -> d.sync.end` | 14.23 | 1.33 | 12.90 |
| `d.sync#15 -> d.sync.end` | 14.03 | 1.45 | 12.58 |
| `d.sync#22 -> d.sync.end` | 12.04 | 1.30 | 10.75 |
| `mtp.dp.ret#0 -> d.sync` | 13.08 | 4.42 | 8.66 |
| `mtp.dp.ret#1 -> d.sync` | 12.17 | 4.55 | 7.62 |
| `s.vp.end#0 -> d.sync` | 11.24 | 4.51 | 6.73 |

Step by step, pass 1 − pass 2, ms: 132.3, -3.5, -2.9, 0.8, 1.6, 1.3, 0.5, 21.5, 15.3, 4.5, 0.3, 0.1, 0.8, 1.5, 0.3, 1.4, 0.3, 0.6, 0.8, 1.3, 0.8, 0.4, 1.0, 0.4, 0.7, 0.7, 5.6, 1.0, 0.9, -3.1, 0.7, 0.6, 1.6, -0.1, 1.4, 0.2, -0.1, -0.0, -0.2, -0.2, 0.7, 0.2, -0.2, 2.8, 0.1, -0.1, 0.2, 0.1, 0.0, -0.3, -0.2, 0.0, -0.0, -0.3, 0.6, -0.0, 0.1, 1.2, 1.0, 1.1, 13.6, 14.8, 4.5, -0.2, 1.3, 8.6, -16.8

### Depth 8192: pass 1 against pass 2

- generation lines: 59 and 58; drafted and accepted per line equal in 43 of 58 paired lines
- the span, pass 1 − pass 2, ms: all 58 paired steps 142.85; steps 0-2 23.69 (20.70, 3.98, -0.99); steps 3 on 119.16 (median 0.12 a step); the server's gen_ms 10345.72 − 9973.57 = 372.15
- the span, median a step: pass 1 170.81, pass 2 170.74 (steps 3 on)

| stage | pass 1, ms (sum) | pass 2, ms (sum) | difference | of which steps 0-2 | median difference a step, steps 3 on |
|---|---:|---:|---:|---:|---:|
| loop | 1.95 | 2.04 | -0.09 | -0.00 | -0.001 |
| draft | 1001.60 | 938.06 | 63.54 | 13.24 | 0.024 |
| checkpoints | 0.14 | 0.14 | -0.01 | -0.00 | 0.000 |
| batch | 0.50 | 0.49 | 0.01 | 0.00 | 0.000 |
| verify | 8957.64 | 8877.77 | 79.87 | 9.43 | 0.020 |
| catch-up | 106.14 | 105.79 | 0.35 | 1.00 | -0.006 |
| acceptance | 31.10 | 31.57 | -0.47 | 0.08 | -0.002 |
| between | 17.95 | 18.31 | -0.35 | -0.05 | -0.011 |
| verify: waits for the cards | 8638.53 | 8414.27 | 224.26 | 137.71 | -0.080 |
| main passes' GPU time (all cards) | 8669.96 | 8469.80 | 200.17 | 128.64 | -0.119 |
| the draft's GPU time | 769.88 | 722.69 | 47.19 | 1.46 | 0.001 |

The ten mark pairs with the largest difference (summed over the paired steps):

| mark pair | pass 1, ms | pass 2, ms | difference |
|---|---:|---:|---:|
| `v.sync#0 -> v.sync.end` | 8638.53 | 8414.27 | 224.26 |
| `v.ub.in#0 -> v.ub.comp` | 300.69 | 433.33 | -132.64 |
| `p.ub.apply#0 -> p.ub.build` | 23.88 | 11.52 | 12.36 |
| `v.ub.apply#0 -> v.ub.build` | 0.00 | 11.11 | -11.11 |
| `d.sync#1 -> d.sync.end` | 10.54 | 1.75 | 8.79 |
| `d.sync#15 -> d.sync.end` | 7.67 | 1.23 | 6.44 |
| `d.sync#8 -> d.sync.end` | 7.49 | 1.22 | 6.27 |
| `d.sync#22 -> d.sync.end` | 5.72 | 1.12 | 4.59 |
| `d.ub.in#0 -> d.ub.comp` | 37.05 | 32.49 | 4.56 |
| `s.vp.end#0 -> d.sync` | 8.90 | 6.16 | 2.75 |

Step by step, pass 1 − pass 2, ms: 20.7, 4.0, -1.0, 0.5, 15.0, 17.9, 2.6, -0.1, 2.3, -2.2, 2.1, -2.1, -0.5, 1.6, -0.2, -1.5, 0.8, -1.0, 2.8, -1.6, -0.1, -0.6, 0.1, 2.4, 0.7, -0.1, -0.1, 1.5, -0.4, -0.3, 0.6, 0.7, 2.2, 1.4, 1.4, -0.9, 1.1, -1.2, -0.1, -1.5, 1.2, 1.3, -0.7, -0.3, -0.8, 0.2, -0.1, 0.1, -2.2, 1.4, -0.2, 1.9, -2.2, 1.6, -1.6, 0.5, -0.8, 76.9

