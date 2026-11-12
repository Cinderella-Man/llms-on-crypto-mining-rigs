## The server's own timings per reply

Steps by the server's counts: generated tokens minus accepted draft tokens (each verify pass gives one sampled token and the drafts it accepts). ms a step = the reply's generation time over that count.

| load | request | prompt tokens | generated | drafted | accepted | share accepted | steps (server) | tokens a step | generation ms | ms a step | tok/s | prompt ms | steps in the profile |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | greedy | 79 | 256 | 254 | 170 | 0.669 | 86 | 2.977 | 13427.7 | 156.14 | 18.99 | 8571.4 | 84 |
| base | long | 2760 | 256 | 273 | 163 | 0.597 | 93 | 2.753 | 13511.2 | 145.28 | 18.87 | 26388.4 | 91 |
| base-2 | greedy | 79 | 256 | 254 | 170 | 0.669 | 86 | 2.977 | 13416.3 | 156.00 | 19.01 | 8503.1 | 84 |
| base-2 | long | 2760 | 256 | 254 | 170 | 0.669 | 86 | 2.977 | 11569.8 | 134.53 | 22.04 | 26504.9 | 84 |
| f | greedy | 79 | 256 | 254 | 170 | 0.669 | 86 | 2.977 | 10856.6 | 126.24 | 23.49 | 7522.2 | 84 |
| f | long | 2760 | 256 | 246 | 173 | 0.703 | 83 | 3.084 | 10624.5 | 128.01 | 24.00 | 26436.8 | 81 |
| f-2 | greedy | 79 | 256 | 254 | 170 | 0.669 | 86 | 2.977 | 10824.7 | 125.87 | 23.56 | 7545.3 | 84 |
| f-2 | long | 2760 | 256 | 273 | 163 | 0.597 | 93 | 2.753 | 12863.7 | 138.32 | 19.82 | 26304.4 | 91 |

## The step per load, from the profile (typical steps)

**step, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| base | 115.36 (117.59; 81) | 123.07 (124.61; 88) | 119.22 |
| base-2 | 115.26 (117.32; 81) | 122.91 (123.70; 81) | 119.09 |
| f | 108.09 (110.30; 80) | 115.68 (116.81; 78) | 111.89 |
| f-2 | 107.92 (110.20; 80) | 116.06 (117.43; 88) | 111.99 |

**verify_span, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| base | 92.81 (92.99; 81) | 100.18 (100.31; 88) | 96.50 |
| base-2 | 92.82 (93.11; 81) | 100.07 (100.21; 81) | 96.44 |
| f | 91.87 (92.04; 80) | 98.85 (99.10; 78) | 95.36 |
| f-2 | 91.74 (91.94; 80) | 98.82 (99.05; 88) | 95.28 |

**verify_gpu, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| base | 88.24 (88.42; 81) | 95.35 (95.48; 88) | 91.80 |
| base-2 | 88.24 (88.46; 81) | 95.33 (95.42; 81) | 91.78 |
| f | 88.27 (88.47; 80) | 95.15 (95.38; 78) | 91.71 |
| f-2 | 88.22 (88.42; 80) | 95.05 (95.31; 88) | 91.64 |

**verify_rest, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| base | 4.56 (4.57; 81) | 4.80 (4.83; 88) | 4.68 |
| base-2 | 4.55 (4.65; 81) | 4.79 (4.79; 81) | 4.67 |
| f | 3.44 (3.57; 80) | 3.70 (3.72; 78) | 3.57 |
| f-2 | 3.45 (3.52; 80) | 3.70 (3.74; 88) | 3.57 |

**the draft's span, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| base | 22.54 (24.59; 81) | 22.94 (24.30; 88) | 22.74 |
| base-2 | 22.18 (24.21; 81) | 22.63 (23.48; 81) | 22.40 |
| f | 16.00 (18.27; 80) | 16.86 (17.70; 78) | 16.43 |
| f-2 | 16.00 (18.27; 80) | 17.24 (18.38; 88) | 16.62 |

**the draft's GPU time, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| base | 11.08 (13.31; 81) | 12.40 (14.09; 88) | 11.74 |
| base-2 | 11.08 (13.31; 81) | 12.37 (13.36; 81) | 11.72 |
| f | 10.83 (12.99; 80) | 12.18 (12.55; 78) | 11.50 |
| f-2 | 10.83 (12.99; 80) | 12.22 (13.31; 88) | 11.53 |

**the leads, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| base | 1.81 (1.81; 81) | 1.81 (1.82; 88) | 1.81 |
| base-2 | 1.80 (1.80; 81) | 1.81 (1.81; 81) | 1.80 |
| f | 4.21 (3.84; 80) | 4.46 (4.40; 78) | 4.33 |
| f-2 | 4.21 (3.82; 80) | 4.48 (4.24; 88) | 4.35 |

**the turns before the last, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| base | 1.46 (1.58; 81) | 1.50 (1.63; 88) | 1.48 |
| base-2 | 1.45 (1.58; 81) | 1.50 (1.56; 81) | 1.48 |
| f | -1.05 (-0.52; 80) | -1.32 (-1.21; 78) | -1.19 |
| f-2 | -1.07 (-0.53; 80) | -1.31 (-1.01; 88) | -1.19 |

**the last turn, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| base | 7.70 (7.90; 81) | 7.11 (6.76; 88) | 7.41 |
| base-2 | 7.37 (7.52; 81) | 6.79 (6.75; 81) | 7.08 |
| f | 1.84 (1.95; 80) | 1.39 (1.96; 78) | 1.61 |
| f-2 | 1.83 (1.98; 80) | 1.80 (1.84; 88) | 1.81 |

## Each arm against the reference loads

Reference: base and base-2. Difference = the load's value minus the mean of the two; spread = their difference.

| load | value | step, ms | minus the reference | reference spread | class |
|---|---|---:|---:|---:|---|
| f | mean of the medians | 111.89 | -7.27 | 0.13 (119.22, 119.09) | gain |
| f | greedy, median | 108.09 | -7.22 | 0.10 (115.36, 115.26) | gain |
| f | long, median | 115.68 | -7.31 | 0.16 (123.07, 122.91) | gain |
| f-2 | mean of the medians | 111.99 | -7.16 | 0.13 (119.22, 119.09) | gain |
| f-2 | greedy, median | 107.92 | -7.39 | 0.10 (115.36, 115.26) | gain |
| f-2 | long, median | 116.06 | -6.94 | 0.16 (123.07, 122.91) | gain |

## The greedy request, step by step against base

| load | same text as the reference | drafted | accepted | steps paired | step: mean difference, ms | standard error | median difference | the draft's span | verify_span | the leads | the turns before the last | the last turn |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | yes | 254 | 170 | — |  |  |  |  |  |  |  |  |
| base-2 | yes | 254 | 170 | 81 | -0.27 | 0.10 | -0.13 | -0.38 | +0.12 | -0.00 | +0.00 | -0.38 |
| f | yes | 254 | 170 | not paired (84 and 84 steps) |  |  |  |  |  |  |  |  |
| f-2 | yes | 254 | 170 | not paired (84 and 84 steps) |  |  |  |  |  |  |  |  |

