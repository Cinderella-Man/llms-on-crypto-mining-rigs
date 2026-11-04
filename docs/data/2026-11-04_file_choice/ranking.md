KLD against Swift-1.5 Q8_0 (noise floor, the reference at another micro-batch: agent 0.029, wiki 0.017, code 0.004). Fit: F16 cache, largest `-ub` with 150 MiB left on every card and the fullest card's free MiB, whole layers unless marked `-ot`. Speed: est. ms of GPU time per decoded token at short context, no draft, kernels timed on a P106 / with the not-yet-timed kernels at their expected speed.

| File | Agent KLD (same top) | Wiki, code KLD | On the cards | Table | 64k | 128k | 262k | 64k + DFlash2 | est. ms/token (tok/s) | vs Q2_0 |
|---|---|---|---|---|---|---|---|---|---|---|
| ISTA IQ3_S | 0.117 ± 0.002 (93.4 %) | 0.161, 0.032 | 51,945 MiB | shard 2 | no (-674) | no (-983) | no (-1498) | no (-711) | 100.3 (10.0) / 98.6 (10.1) | +17.9 / +25.7 |
| Swift IQ3_S | 0.121 ± 0.003 (93.7 %) | 0.153, 0.032 | 52,066 MiB | shard 1, reshard | no (-689) | no (-998) | no (-1513) | no (-726) | 102.5 (9.8) / 100.8 (9.9) | +20.0 / +27.8 |
| Swift IQ3_XXS | 0.176 ± 0.004 (92.3 %) | 0.248, 0.054 | 44,710 MiB | shard 3 | ub 64: 155 | -ot ub 128: 159 (10 moved) | no (-598) | -ot ub 256: 202 (9 moved) | 98.9 (10.1) / 91.9 (10.9) | +16.4 / +19.0 |
| Swift IQ3_XXS-hcq8 | not scored | 0.245, — | 44,018 MiB | shard 3 | ub 128: 187 | -ot ub 128: 233 (10 moved) | no (-526) | ub 64: 201 | 114.4 (8.7) / 90.6 (11.0) | +32.0 / +17.6 |
| Swift IQ2_XS | 0.225 ± 0.005 (91.5 %) | 0.351, 0.075 | 37,196 MiB | shard 1, reshard | ub 512: 569 | ub 256: 453 | -ot ub 128: 230 (16 moved) | ub 512: 505 | 87.2 (11.5) / 82.4 (12.1) | +4.8 / +9.5 |
| ISTA Coder IQ1_M | 0.274 ± 0.005 (90.7 %) | 0.846, 0.131 | 27,904 MiB | shard 2 | ub 512: 1719 | ub 512: 1124 | ub 256: 581 | ub 512: 1634 | 99.9 (10.0) / 98.2 (10.2) | +17.4 / +25.2 |
| Swift Q2_0-hcq8 | 0.283 ± 0.005 (90.7 %) | 0.428, 0.112 | 35,038 MiB | shard 3 | ub 512: 822 | ub 512: 200 | ub 64: 298 | ub 512: 659 | 98.0 (10.2) / 71.6 (14.0) | +15.5 / -1.4 |
| Swift Q2_0 | 0.285 ± 0.006 (90.7 %) | 0.428, 0.112 | 35,730 MiB | shard 3 | ub 512: 737 | ub 256: 664 | ub 64: 221 | ub 512: 573 | 82.4 (12.1) / 73.0 (13.7) | — |
