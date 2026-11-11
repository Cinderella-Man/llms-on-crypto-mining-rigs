## The server's own timings per reply

Steps by the server's counts: generated tokens minus accepted draft tokens (each verify pass gives one sampled token and the drafts it accepts). ms a step = the reply's generation time over that count.

| load | request | prompt tokens | generated | drafted | accepted | share accepted | steps (server) | tokens a step | generation ms | ms a step | tok/s | prompt ms | steps in the profile |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| prof | greedy | 79 | 256 | 255 | 170 | 0.667 | 86 | 2.977 | 11321.6 | 131.65 | 22.52 | 7355.8 | 84 |
| prof | long | 2760 | 256 | 278 | 161 | 0.579 | 95 | 2.695 | 19432.1 | 204.55 | 13.12 | 29809.6 | 92 |
| trace | greedy | 79 | 256 | 255 | 170 | 0.667 | 86 | 2.977 | 11521.9 | 133.98 | 22.13 | 8164.6 | 84 |
| trace | long | 2760 | 256 | 280 | 161 | 0.575 | 95 | 2.695 | 14950.9 | 157.38 | 17.06 | 30249.5 | 93 |

## The step per load, from the profile (typical steps)

**step, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| prof | 128.92 (130.75; 81) | 139.17 (168.68; 89) | 134.05 |
| trace | 130.25 (131.93; 80) | 138.49 (145.21; 90) | 134.37 |

**verify_span, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| prof | 105.96 (106.11; 81) | 113.68 (113.82; 89) | 109.82 |
| trace | 105.91 (106.01; 80) | 113.60 (113.71; 90) | 109.75 |

**verify_gpu, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| prof | 101.69 (101.86; 81) | 109.18 (109.30; 89) | 105.43 |
| trace | 101.65 (101.76; 80) | 109.09 (109.18; 90) | 105.37 |

**verify_rest, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| prof | 4.25 (4.25; 81) | 4.51 (4.52; 89) | 4.38 |
| trace | 4.26 (4.26; 80) | 4.53 (4.53; 90) | 4.39 |

**the draft's span, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| prof | 22.47 (24.63; 81) | 25.16 (54.85; 89) | 23.82 |
| trace | 24.16 (25.91; 80) | 24.81 (31.51; 90) | 24.48 |

**the draft's GPU time, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| prof | 11.40 (13.61; 81) | 13.32 (15.41; 89) | 12.36 |
| trace | 11.37 (13.62; 80) | 13.36 (14.72; 90) | 12.37 |

**the leads, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| prof | 2.34 (2.39; 81) | 2.68 (2.81; 89) | 2.51 |
| trace | 2.34 (2.35; 80) | 2.69 (2.69; 90) | 2.52 |

**the turns before the last, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| prof | 1.31 (1.40; 81) | 1.39 (2.77; 89) | 1.35 |
| trace | 1.29 (1.37; 80) | 1.35 (1.41; 90) | 1.32 |

**the last turn, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| prof | 7.25 (7.24; 81) | 7.48 (33.86; 89) | 7.37 |
| trace | 8.43 (8.57; 80) | 7.37 (12.69; 90) | 7.90 |

## The greedy request, step by step against prof

| load | same text as the reference | drafted | accepted | steps paired | step: mean difference, ms | standard error | median difference | the draft's span | verify_span | the leads | the turns before the last | the last turn |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| prof | yes | 255 | 170 | — |  |  |  |  |  |  |  |  |
| trace | yes | 255 | 170 | 80 | +1.26 | 0.15 | +1.05 | +1.24 | +0.02 | -0.04 | -0.02 | +1.30 |

