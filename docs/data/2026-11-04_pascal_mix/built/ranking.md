KLD against Swift-1.5 Q8_0 (noise floor, the reference at another micro-batch: agent 0.029, wiki 0.017, code 0.004). Fit: F16 cache, largest `-ub` with 150 MiB left on every card and the fullest card's free MiB, whole layers unless marked `-ot`. Speed: est. ms of GPU time per decoded token at short context, no draft, kernels timed on a P106 / with the not-yet-timed kernels at their expected speed.

| File | Agent KLD (same top) | Wiki, code KLD | On the cards | Table | 64k | 128k | 262k | 64k + DFlash2 | est. ms/token (tok/s) | vs Q2_0 |
|---|---|---|---|---|---|---|---|---|---|---|
| pmix IQ3-FAST | 0.174 ± 0.004 (92.3 %) | 0.249, 0.055 | 44,628 MiB | shard 3 | ub 64: 169 | -ot ub 128: 167 (10 moved) | no (-590) | -ot ub 256: 214 (9 moved) | 93.4 (10.7) / 88.9 (11.3) | +10.9 / +15.9 |
| Swift IQ3_XXS-C5 | 0.179 ± 0.004 (92.2 %) | 0.248, 0.054 | 44,710 MiB | shard 3 | ub 64: 155 | -ot ub 128: 159 (10 moved) | no (-598) | -ot ub 256: 202 (9 moved) | 98.9 (10.1) / 91.9 (10.9) | +16.4 / +19.0 |
| pmix Q2-SPINE6 | 0.256 ± 0.005 (90.9 %) | 0.421, 0.104 | 36,522 MiB | shard 3 | ub 512: 644 | ub 256: 593 | ub 64: 164 | ub 512: 456 | 82.0 (12.2) / 81.9 (12.2) | -0.5 / +8.9 |
| pmix Q2-FAST | 0.279 ± 0.005 (90.7 %) | 0.426, 0.108 | 35,735 MiB | shard 3 | ub 512: 737 | ub 256: 672 | ub 64: 229 | ub 512: 569 | 72.4 (13.8) / 72.4 (13.8) | -10.0 / -0.6 |
| Swift Q2_0-C5 | 0.287 ± 0.006 (90.7 %) | 0.428, 0.112 | 35,730 MiB | shard 3 | ub 512: 737 | ub 256: 664 | ub 64: 221 | ub 512: 573 | 82.4 (12.1) / 73.0 (13.7) | — |
