trace 792 lines in 12 pieces (cut at gaps over 2000 ms); requests in the CSV that read a prompt: 12

| request | prompt lines | generation lines | drafted (trace) | accepted (trace) | generation span, ms | prompt's last line end to the first generation line, ms | server gen_ms | server drafted | server accepted | main graph built/reused | draft graph built/reused | prompt lines: main graph built/reused | main passes' GPU time, ms (all cards) | draft's GPU time, ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| warmup0@0 | 1 | 17 | 45 | 14 | 2837.10 | 0.10 | 2837.071 | 45 | 14 | 3/0 | 20/42 | 2/0 | 2006.30 | 177.90 |
| decode1@0 | 1 | 108 | 321 | 147 | 11266.35 | 0.09 | 11266.295 | 321 | 147 | 1/0 | 116/313 | 2/0 | 9306.39 | 1032.64 |
| decode2@0 | 1 | 107 | 318 | 148 | 10726.41 | 0.08 | 10726.36 | 318 | 148 | 1/0 | 109/316 | 1/0 | 8984.94 | 898.72 |
| samples@0 | 1 | 101 | 302 | 154 | 10613.64 | 0.13 | 10613.611 | 302 | 154 | 1/0 | 110/294 | 1/0 | 8786.43 | 1003.52 |
| prime@4096 | 1 | 0 | 0 | 0 | 0.00 | nan | 0.001 | 0 | 0 | 0/0 | 0/0 | 9/0 | 0.00 | 0.00 |
| decode1@4096 | 1 | 78 | 231 | 177 | 9053.57 | 2.55 | 9055.7 | 231 | 177 | 1/0 | 80/229 | 2/0 | 7575.09 | 760.29 |
| decode2@4096 | 1 | 78 | 231 | 177 | 8749.69 | 0.12 | 8749.389 | 231 | 177 | 1/0 | 79/230 | 1/0 | 7372.26 | 724.63 |
| samples@4096 | 1 | 76 | 228 | 179 | 8630.88 | 0.15 | 8630.6 | 228 | 179 | 0/0 | 79/225 | 1/0 | 7176.96 | 768.15 |
| prime@8192 | 2 | 0 | 0 | 0 | 0.00 | nan | 0.001 | 0 | 0 | 0/0 | 0/0 | 17/0 | 0.00 | 0.00 |
| decode1@8192 | 1 | 71 | 212 | 184 | 8600.18 | 0.15 | 8599.624 | 212 | 184 | 1/0 | 75/208 | 2/0 | 7169.00 | 721.10 |
| decode2@8192 | 1 | 70 | 209 | 185 | 8145.28 | 0.09 | 8144.659 | 209 | 185 | 1/0 | 71/208 | 1/0 | 6824.30 | 670.75 |
| samples@8192 | 1 | 73 | 216 | 182 | 8709.76 | 0.09 | 8706.65 | 216 | 182 | 1/0 | 75/214 | 1/0 | 7093.27 | 714.45 |

### Depth 0: pass 1 against pass 2

- generation lines: 108 and 107; drafted and accepted per line equal in 92 of 107 paired lines
- the span, pass 1 − pass 2, ms: all 107 paired steps 344.65; steps 0-2 -15.62 (-4.93, -6.20, -4.49); steps 3 on 360.27 (median 0.82 a step); the server's gen_ms 11266.295 − 10726.36 = 539.93
- the span, median a step: pass 1 100.90, pass 2 99.87 (steps 3 on)

| stage | pass 1, ms (sum) | pass 2, ms (sum) | difference | of which steps 0-2 | median difference a step, steps 3 on |
|---|---:|---:|---:|---:|---:|
| loop | 1.28 | 1.29 | -0.02 | -0.00 | 0.000 |
| draft | 1283.32 | 1129.67 | 153.65 | -4.88 | 0.211 |
| checkpoints | 0.24 | 0.23 | 0.00 | 0.00 | 0.000 |
| batch | 0.90 | 0.74 | 0.16 | 0.00 | 0.002 |
| verify | 9604.75 | 9415.33 | 189.43 | -10.60 | 0.514 |
| catch-up | 106.35 | 105.67 | 0.68 | -0.12 | 0.005 |
| acceptance | 47.72 | 47.36 | 0.36 | -0.02 | 0.001 |
| between | 26.51 | 26.12 | 0.39 | -0.01 | 0.001 |
| verify: waits for the cards | 9015.90 | 8846.74 | 169.16 | 1.44 | 0.143 |
| main passes' GPU time (all cards) | 9128.99 | 8984.94 | 144.05 | -8.29 | 0.120 |
| the draft's GPU time | 1031.75 | 898.72 | 133.03 | -6.34 | -0.000 |

The ten mark pairs with the largest difference (summed over the paired steps):

| mark pair | pass 1, ms | pass 2, ms | difference |
|---|---:|---:|---:|
| `v.sync#0 -> v.sync.end` | 9015.90 | 8846.74 | 169.16 |
| `d.sync#1 -> d.sync.end` | 35.99 | 2.84 | 33.15 |
| `d.sync#8 -> d.sync.end` | 32.78 | 2.19 | 30.58 |
| `v.ub.in#0 -> v.ub.comp` | 561.69 | 531.16 | 30.53 |
| `d.sync#15 -> d.sync.end` | 30.06 | 1.95 | 28.12 |
| `mtp.dp.ret#0 -> d.sync` | 19.78 | 7.60 | 12.17 |
| `mtp.dp.ret#1 -> d.sync` | 18.62 | 7.27 | 11.35 |
| `s.vp.end#0 -> d.sync` | 18.48 | 7.14 | 11.34 |
| `v.ub.apply#0 -> v.ub.build` | 0.00 | 11.25 | -11.25 |
| `mtp.dp.row#2 -> mtp.drf.end` | 15.52 | 6.73 | 8.79 |

Step by step, pass 1 − pass 2, ms: -4.9, -6.2, -4.5, 0.1, 0.8, 0.5, 0.5, 0.6, 3.6, 1.2, 0.8, 1.3, 0.8, 0.9, 1.4, -0.9, -0.6, -0.3, -1.0, 0.3, -2.9, -0.6, 11.8, 11.5, 2.6, 11.6, 8.9, 2.6, -0.0, 0.0, -1.1, -2.5, 0.0, 0.0, 1.3, 0.8, 3.0, 0.7, -0.0, 1.3, 0.9, 0.9, 0.2, 2.2, 0.4, 0.1, 1.2, 1.1, 2.1, 0.8, 0.8, 0.3, 0.2, 7.1, 2.3, 0.7, 0.1, 0.2, 1.0, 0.9, 1.9, 1.0, 0.9, 0.2, 0.5, 1.1, 0.3, 0.0, 0.1, 1.4, 0.0, 0.8, 0.8, 0.5, 1.2, 12.9, 12.6, 4.9, 1.2, -1.7, 10.2, 14.5, 4.1, -0.9, 2.8, 0.4, -21.9, 139.1, 5.8, 0.7, 0.6, 1.6, 2.1, -0.1, 2.5, 0.7, -0.9, 15.5, 14.3, 6.5, 0.0, -0.9, 9.9, 0.3, -0.4, 1.4, 29.1

### Depth 4096: pass 1 against pass 2

- generation lines: 78 and 78; drafted and accepted per line equal in 78 of 78 paired lines
- the span, pass 1 − pass 2, ms: all 78 paired steps 303.88; steps 0-2 94.75 (106.64, -6.85, -5.04); steps 3 on 209.13 (median 0.72 a step); the server's gen_ms 9055.7 − 8749.389 = 306.31
- the span, median a step: pass 1 112.64, pass 2 111.94 (steps 3 on)

| stage | pass 1, ms (sum) | pass 2, ms (sum) | difference | of which steps 0-2 | median difference a step, steps 3 on |
|---|---:|---:|---:|---:|---:|
| loop | 1.80 | 1.90 | -0.10 | 0.00 | -0.001 |
| draft | 960.65 | 912.01 | 48.65 | 6.78 | 0.165 |
| checkpoints | 0.16 | 0.16 | 0.01 | 0.00 | 0.000 |
| batch | 0.75 | 0.56 | 0.19 | 0.02 | 0.001 |
| verify | 7926.24 | 7669.77 | 256.47 | 88.01 | 0.432 |
| catch-up | 107.33 | 108.74 | -1.41 | -0.10 | -0.013 |
| acceptance | 36.01 | 35.97 | 0.03 | 0.08 | -0.004 |
| between | 20.62 | 20.57 | 0.05 | -0.04 | 0.001 |
| verify: waits for the cards | 7483.24 | 7292.40 | 190.84 | 74.91 | -0.024 |
| main passes' GPU time (all cards) | 7575.09 | 7372.26 | 202.83 | 83.69 | -0.009 |
| the draft's GPU time | 760.29 | 724.63 | 35.66 | 5.55 | 0.007 |

The ten mark pairs with the largest difference (summed over the paired steps):

| mark pair | pass 1, ms | pass 2, ms | difference |
|---|---:|---:|---:|
| `v.sync#0 -> v.sync.end` | 7483.24 | 7292.40 | 190.84 |
| `v.ub.in#0 -> v.ub.comp` | 406.73 | 346.53 | 60.20 |
| `mtp.dp.ret#0 -> d.sync` | 14.70 | 4.81 | 9.89 |
| `mtp.dp.ret#1 -> d.sync` | 13.89 | 4.29 | 9.60 |
| `d.sync#7 -> d.sync.end` | 210.86 | 220.18 | -9.32 |
| `s.vp.end#0 -> d.sync` | 13.75 | 4.47 | 9.28 |
| `d.sync#1 -> d.sync.end` | 10.62 | 1.95 | 8.67 |
| `mtp.dp.row#2 -> mtp.drf.end` | 11.37 | 3.74 | 7.63 |
| `d.sync#14 -> d.sync.end` | 139.38 | 146.22 | -6.85 |
| `d.sync#8 -> d.sync.end` | 8.10 | 1.44 | 6.66 |

Step by step, pass 1 − pass 2, ms: 106.6, -6.9, -5.0, -1.9, -0.2, 0.8, 1.1, 16.1, 11.7, 5.0, 0.4, 0.2, 1.0, 1.8, 0.8, 1.0, 1.1, 1.0, 1.5, 0.7, 7.6, 1.0, 0.8, 1.0, 0.9, 1.2, 0.8, -0.2, 0.9, 1.1, 1.0, 0.9, 0.3, 0.8, 1.0, 1.0, 0.0, 0.6, 0.2, 0.7, 0.9, 1.0, 0.8, 0.4, 0.2, 0.3, 0.4, 0.5, 0.8, 2.7, -0.1, 3.7, 0.9, 0.0, 2.5, 1.1, 0.1, 0.0, -0.0, -0.2, 0.0, -0.0, 0.3, -0.1, -0.3, 0.7, 0.0, -0.4, 0.0, -0.1, -0.2, -0.0, -0.1, -0.2, 0.7, 0.5, -0.2, 129.0

### Depth 8192: pass 1 against pass 2

- generation lines: 71 and 70; drafted and accepted per line equal in 53 of 70 paired lines
- the span, pass 1 − pass 2, ms: all 70 paired steps 208.40; steps 0-2 140.03 (137.05, 3.26, -0.27); steps 3 on 68.36 (median 0.43 a step); the server's gen_ms 8599.624 − 8144.659 = 454.97
- the span, median a step: pass 1 116.48, pass 2 116.11 (steps 3 on)

| stage | pass 1, ms (sum) | pass 2, ms (sum) | difference | of which steps 0-2 | median difference a step, steps 3 on |
|---|---:|---:|---:|---:|---:|
| loop | 2.39 | 2.40 | -0.02 | -0.01 | -0.000 |
| draft | 925.51 | 870.10 | 55.41 | 15.32 | 0.040 |
| checkpoints | 0.15 | 0.15 | 0.00 | 0.00 | 0.000 |
| batch | 0.61 | 0.58 | 0.03 | 0.01 | 0.000 |
| verify | 7247.74 | 7094.81 | 152.93 | 124.96 | 0.361 |
| catch-up | 126.06 | 126.47 | -0.41 | -0.20 | 0.003 |
| acceptance | 31.62 | 32.27 | -0.64 | -0.03 | -0.006 |
| between | 19.60 | 18.51 | 1.09 | -0.01 | 0.012 |
| verify: waits for the cards | 6893.26 | 6750.55 | 142.71 | 113.24 | -0.127 |
| main passes' GPU time (all cards) | 6954.43 | 6824.30 | 130.13 | 122.08 | -0.113 |
| the draft's GPU time | 713.59 | 670.75 | 42.85 | 11.59 | 0.006 |

The ten mark pairs with the largest difference (summed over the paired steps):

| mark pair | pass 1, ms | pass 2, ms | difference |
|---|---:|---:|---:|
| `v.sync#0 -> v.sync.end` | 6893.26 | 6750.55 | 142.71 |
| `v.ub.in#0 -> v.ub.comp` | 336.29 | 314.57 | 21.72 |
| `d.ub.in#1 -> d.ub.comp` | 48.39 | 32.83 | 15.56 |
| `v.ub.apply#0 -> v.ub.build` | 0.00 | 10.90 | -10.90 |
| `mtp.dp.ret#0 -> d.sync` | 13.41 | 4.23 | 9.17 |
| `d.sync#1 -> d.sync.end` | 10.47 | 1.80 | 8.67 |
| `d.sync#8 -> d.sync.end` | 7.63 | 1.29 | 6.34 |
| `d.sync#15 -> d.sync.end` | 6.09 | 1.25 | 4.83 |
| `d.sync#0 -> d.sync.end` | 197.25 | 202.01 | -4.75 |
| `d.sync#7 -> d.sync.end` | 199.92 | 204.47 | -4.54 |

Step by step, pass 1 − pass 2, ms: 137.0, 3.3, -0.3, 12.8, 15.9, 5.2, 0.3, 4.3, -1.5, -2.0, 2.1, -1.1, 1.1, 0.1, 0.4, -0.2, -0.6, 2.0, 1.4, -1.0, 2.3, -1.0, 0.9, -1.0, 0.4, 1.1, -0.9, 0.6, -0.4, 0.5, 1.6, 14.0, -0.5, 0.6, 0.1, 2.1, 2.1, 1.0, 2.2, 0.6, 2.2, -0.4, 0.8, -0.1, 1.2, -0.3, -2.2, 1.5, 0.0, 1.3, -1.1, 1.2, -1.7, 0.2, 0.2, 0.4, -0.6, 0.5, -1.3, 0.5, 0.3, 0.6, -1.4, 1.5, -1.0, 0.0, -0.1, 1.3, -1.2, 0.7

