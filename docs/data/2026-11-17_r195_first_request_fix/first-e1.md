trace 457 lines in 10 pieces (cut at gaps over 2000 ms); requests in the CSV that read a prompt: 10

| request | prompt lines | generation lines | drafted (trace) | accepted (trace) | generation span, ms | prompt's last line end to the first generation line, ms | server gen_ms | server drafted | server accepted | main graph built/reused | draft graph built/reused | prompt lines: main graph built/reused | main passes' GPU time, ms (all cards) | draft's GPU time, ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| warmup0@0 | 1 | 17 | 45 | 14 | 2844.72 | 0.09 | 2844.688 | 45 | 14 | 3/0 | 20/42 | 2/0 | 2015.99 | 181.74 |
| prime@4096 | 1 | 0 | 0 | 0 | 0.00 | nan | 0.001 | 0 | 0 | 0/0 | 0/0 | 9/0 | 0.00 | 0.00 |
| decode1@4096 | 1 | 77 | 229 | 178 | 8966.28 | 0.13 | 8965.945 | 229 | 178 | 1/0 | 80/226 | 2/0 | 7488.87 | 759.20 |
| decode2@4096 | 1 | 77 | 229 | 178 | 8636.02 | 0.16 | 8635.736 | 229 | 178 | 1/0 | 78/228 | 1/0 | 7252.85 | 723.84 |
| prime@6144 | 2 | 0 | 0 | 0 | 0.00 | nan | 0.001 | 0 | 0 | 0/0 | 0/0 | 13/0 | 0.00 | 0.00 |
| decode1@6144 | 1 | 69 | 204 | 186 | 8461.87 | 0.18 | 8461.486 | 204 | 186 | 1/0 | 71/202 | 2/0 | 7090.59 | 685.07 |
| decode2@6144 | 1 | 68 | 201 | 187 | 8019.35 | 0.09 | 8018.888 | 201 | 187 | 1/0 | 69/200 | 1/0 | 6749.77 | 643.89 |
| prime@8192 | 2 | 0 | 0 | 0 | 0.00 | nan | 0.001 | 0 | 0 | 0/0 | 0/0 | 17/0 | 0.00 | 0.00 |
| decode1@8192 | 1 | 69 | 207 | 186 | 8255.84 | 0.10 | 8255.253 | 207 | 186 | 0/0 | 72/204 | 2/0 | 6872.10 | 697.21 |
| decode2@8192 | 1 | 68 | 204 | 187 | 7948.80 | 0.12 | 7948.177 | 204 | 187 | 0/0 | 69/203 | 1/0 | 6560.11 | 653.52 |

### Depth 4096: pass 1 against pass 2

- generation lines: 77 and 77; drafted and accepted per line equal in 77 of 77 paired lines
- the span, pass 1 − pass 2, ms: all 77 paired steps 330.26; steps 0-2 138.81 (130.75, 6.84, 1.22); steps 3 on 191.45 (median 0.29 a step); the server's gen_ms 8965.945 − 8635.736 = 330.21
- the span, median a step: pass 1 112.74, pass 2 112.22 (steps 3 on)

| stage | pass 1, ms (sum) | pass 2, ms (sum) | difference | of which steps 0-2 | median difference a step, steps 3 on |
|---|---:|---:|---:|---:|---:|
| loop | 1.89 | 2.10 | -0.21 | -0.00 | -0.003 |
| draft | 964.44 | 917.39 | 47.05 | 7.31 | 0.106 |
| checkpoints | 0.33 | 0.30 | 0.03 | 0.01 | 0.000 |
| batch | 0.68 | 0.62 | 0.06 | 0.00 | 0.000 |
| verify | 7833.25 | 7549.06 | 284.19 | 131.81 | 0.145 |
| catch-up | 108.05 | 109.16 | -1.11 | -0.40 | -0.013 |
| acceptance | 38.29 | 38.24 | 0.05 | 0.10 | 0.000 |
| between | 19.34 | 19.14 | 0.20 | -0.01 | 0.002 |
| verify: waits for the cards | 7383.82 | 7164.14 | 219.68 | 116.70 | -0.109 |
| main passes' GPU time (all cards) | 7488.87 | 7252.85 | 236.02 | 126.01 | -0.092 |
| the draft's GPU time | 759.20 | 723.84 | 35.36 | 6.03 | -0.014 |

The ten mark pairs with the largest difference (summed over the paired steps):

| mark pair | pass 1, ms | pass 2, ms | difference |
|---|---:|---:|---:|
| `v.sync#0 -> v.sync.end` | 7383.82 | 7164.14 | 219.68 |
| `v.ub.in#0 -> v.ub.comp` | 417.01 | 353.23 | 63.78 |
| `d.sync#1 -> d.sync.end` | 11.08 | 2.05 | 9.03 |
| `mtp.dp.ret#1 -> d.sync` | 11.85 | 4.38 | 7.48 |
| `d.sync#7 -> d.sync.end` | 207.35 | 214.40 | -7.05 |
| `mtp.dp.ret#0 -> d.sync` | 11.40 | 4.65 | 6.76 |
| `d.sync#8 -> d.sync.end` | 8.30 | 1.54 | 6.76 |
| `s.vp.end#0 -> d.sync` | 11.07 | 4.44 | 6.63 |
| `mtp.dp.row#2 -> mtp.drf.end` | 9.31 | 3.68 | 5.64 |
| `d.sync#15 -> d.sync.end` | 7.00 | 1.37 | 5.63 |

Step by step, pass 1 − pass 2, ms: 130.8, 6.8, 1.2, 1.1, 0.9, 0.4, 1.1, 1.9, 16.2, 13.2, 4.3, 0.8, 4.7, 1.1, 0.6, 5.5, 0.8, -0.0, -0.0, 0.2, -0.1, -0.4, 1.0, 0.8, 1.7, 1.0, 0.6, -0.1, 0.7, 0.9, 0.8, 0.8, 0.4, 0.7, 0.1, 0.9, 0.8, 0.2, 0.3, 0.1, 1.2, -0.2, -0.3, 0.2, 0.2, -0.2, -0.0, 0.3, 0.3, 2.8, -0.4, 2.6, 0.4, 0.1, -0.1, 3.0, -0.0, 0.1, 0.2, -0.2, -0.2, -0.1, -0.3, -0.5, 2.0, -0.1, -0.3, -0.1, 0.1, -0.2, 0.6, -0.1, -0.3, 0.2, -0.0, 0.1, 117.4

### Depth 6144: pass 1 against pass 2

- generation lines: 69 and 68; drafted and accepted per line equal in 55 of 68 paired lines
- the span, pass 1 − pass 2, ms: all 68 paired steps 238.82; steps 0-2 143.41 (120.81, 5.58, 17.03); steps 3 on 95.41 (median 0.58 a step); the server's gen_ms 8461.486 − 8018.888 = 442.60
- the span, median a step: pass 1 118.70, pass 2 118.11 (steps 3 on)

| stage | pass 1, ms (sum) | pass 2, ms (sum) | difference | of which steps 0-2 | median difference a step, steps 3 on |
|---|---:|---:|---:|---:|---:|
| loop | 2.02 | 2.12 | -0.10 | 0.01 | -0.002 |
| draft | 887.15 | 834.32 | 52.83 | 19.34 | 0.120 |
| checkpoints | 0.36 | 0.27 | 0.09 | 0.00 | 0.000 |
| batch | 0.61 | 0.55 | 0.07 | 0.01 | 0.000 |
| verify | 7202.04 | 7015.35 | 186.69 | 123.73 | 0.461 |
| catch-up | 113.86 | 114.36 | -0.50 | 0.26 | 0.000 |
| acceptance | 33.30 | 33.38 | -0.08 | 0.09 | 0.007 |
| between | 18.83 | 19.00 | -0.18 | -0.03 | 0.005 |
| verify: waits for the cards | 6825.69 | 6661.48 | 164.22 | 114.16 | -0.010 |
| main passes' GPU time (all cards) | 6903.78 | 6749.77 | 154.02 | 122.52 | -0.039 |
| the draft's GPU time | 683.53 | 643.89 | 39.64 | 17.47 | -0.011 |

The ten mark pairs with the largest difference (summed over the paired steps):

| mark pair | pass 1, ms | pass 2, ms | difference |
|---|---:|---:|---:|
| `v.sync#0 -> v.sync.end` | 6825.69 | 6661.48 | 164.22 |
| `v.ub.in#0 -> v.ub.comp` | 357.14 | 323.68 | 33.47 |
| `v.ub.apply#0 -> v.ub.build` | 0.00 | 11.02 | -11.02 |
| `d.sync#1 -> d.sync.end` | 10.67 | 1.77 | 8.91 |
| `mtp.dp.ret#0 -> d.sync` | 11.28 | 4.07 | 7.21 |
| `mtp.dp.ret#1 -> d.sync` | 10.71 | 3.92 | 6.79 |
| `d.sync#8 -> d.sync.end` | 8.09 | 1.42 | 6.67 |
| `s.vp.end#0 -> d.sync` | 10.73 | 4.15 | 6.58 |
| `mtp.dp.row#2 -> mtp.drf.end` | 9.03 | 3.50 | 5.53 |
| `d.sync#15 -> d.sync.end` | 6.52 | 1.26 | 5.26 |

Step by step, pass 1 − pass 2, ms: 120.8, 5.6, 17.0, 11.0, 6.1, 1.7, 0.3, -1.2, 1.1, 0.1, 1.1, 0.7, 0.8, -1.2, 2.9, 3.3, 0.3, 5.9, -2.3, -0.5, 4.0, -0.5, 1.2, 3.0, -1.7, 2.4, 0.1, 0.9, 0.4, 3.1, -1.7, 1.0, -0.1, 0.6, 0.5, 0.9, 0.5, 1.0, -1.1, 0.0, -0.3, 1.1, -1.1, 0.8, -1.4, 0.1, 0.6, 1.3, 0.4, 0.7, 0.5, 1.4, -0.1, 0.7, -0.9, 0.6, 4.1, -0.1, -1.0, 3.5, 0.6, -0.8, -0.2, 0.9, -1.5, 1.7, 0.3, 39.1

### Depth 8192: pass 1 against pass 2

- generation lines: 69 and 68; drafted and accepted per line equal in 62 of 68 paired lines
- the span, pass 1 − pass 2, ms: all 68 paired steps 189.68; steps 0-2 132.54 (125.34, 6.51, 0.69); steps 3 on 57.15 (median 0.33 a step); the server's gen_ms 8255.253 − 7948.177 = 307.08
- the span, median a step: pass 1 116.92, pass 2 116.72 (steps 3 on)

| stage | pass 1, ms (sum) | pass 2, ms (sum) | difference | of which steps 0-2 | median difference a step, steps 3 on |
|---|---:|---:|---:|---:|---:|
| loop | 2.41 | 2.45 | -0.04 | -0.00 | -0.001 |
| draft | 902.48 | 862.93 | 39.55 | 4.80 | 0.050 |
| checkpoints | 0.31 | 0.27 | 0.04 | 0.02 | 0.000 |
| batch | 0.60 | 0.55 | 0.05 | 0.01 | 0.000 |
| verify | 7055.40 | 6905.60 | 149.80 | 127.53 | 0.207 |
| catch-up | 126.08 | 125.68 | 0.41 | 0.00 | 0.010 |
| acceptance | 32.75 | 33.07 | -0.32 | 0.15 | 0.005 |
| between | 18.46 | 18.26 | 0.20 | 0.01 | 0.002 |
| verify: waits for the cards | 6696.29 | 6582.48 | 113.82 | 117.93 | -0.142 |
| main passes' GPU time (all cards) | 6774.20 | 6560.11 | 214.09 | 128.20 | 0.027 |
| the draft's GPU time | 687.56 | 653.52 | 34.05 | 3.94 | -0.016 |

The ten mark pairs with the largest difference (summed over the paired steps):

| mark pair | pass 1, ms | pass 2, ms | difference |
|---|---:|---:|---:|
| `v.sync#0 -> v.sync.end` | 6696.29 | 6582.48 | 113.82 |
| `v.ub.in#0 -> v.ub.comp` | 340.06 | 304.88 | 35.17 |
| `d.sync#1 -> d.sync.end` | 10.69 | 1.80 | 8.89 |
| `d.sync#8 -> d.sync.end` | 7.97 | 1.41 | 6.56 |
| `d.sync#15 -> d.sync.end` | 6.24 | 1.28 | 4.96 |
| `mtp.dp.ret#1 -> d.sync` | 8.87 | 4.01 | 4.85 |
| `d.sync#7 -> d.sync.end` | 191.14 | 195.59 | -4.45 |
| `mtp.dp.ret#0 -> d.sync` | 8.62 | 4.18 | 4.44 |
| `s.vp.end#0 -> d.sync` | 8.57 | 4.25 | 4.31 |
| `mtp.dp.row#2 -> mtp.drf.end` | 7.11 | 3.48 | 3.63 |

Step by step, pass 1 − pass 2, ms: 125.3, 6.5, 0.7, 0.7, 0.9, 14.2, 14.8, 2.7, -0.4, 3.1, -0.6, -0.8, 1.1, -0.9, 2.3, -0.3, -0.1, 0.9, -1.9, 0.7, 2.3, 1.4, 5.9, 0.8, -1.3, 0.1, 0.9, 0.1, 0.6, 0.6, 0.2, -0.2, 1.2, -0.2, 0.3, 0.6, -0.8, 1.8, 1.8, 0.3, -1.3, 3.7, 0.2, 1.3, 0.4, -0.0, -0.6, 1.3, 2.1, -0.5, 1.1, -1.9, 0.2, -0.2, 0.3, -0.6, 0.5, -1.5, 0.9, -0.1, 0.7, -1.1, 1.8, -1.3, -0.2, 1.0, -1.0, -0.6

