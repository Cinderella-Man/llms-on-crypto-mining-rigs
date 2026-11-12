## The server's own timings per reply

Steps by the server's counts: generated tokens minus accepted draft tokens (each verify pass gives one sampled token and the drafts it accepts). ms a step = the reply's generation time over that count.

| load | request | prompt tokens | generated | drafted | accepted | share accepted | steps (server) | tokens a step | generation ms | ms a step | tok/s | prompt ms | steps in the profile |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | greedy | 79 | 256 | 254 | 170 | 0.669 | 86 | 2.977 | 13427.7 | 156.14 | 18.99 | 8571.4 | 84 |
| base | long | 2760 | 256 | 273 | 163 | 0.597 | 93 | 2.753 | 13511.2 | 145.28 | 18.87 | 26388.4 | 91 |
| base-2 | greedy | 79 | 256 | 254 | 170 | 0.669 | 86 | 2.977 | 13416.3 | 156.00 | 19.01 | 8503.1 | 84 |
| base-2 | long | 2760 | 256 | 254 | 170 | 0.669 | 86 | 2.977 | 11569.8 | 134.53 | 22.04 | 26504.9 | 84 |
| a | greedy | 79 | 256 | 254 | 170 | 0.669 | 86 | 2.977 | 13054.3 | 151.79 | 19.53 | 8530.0 | 84 |
| a | long | 2760 | 256 | 276 | 163 | 0.591 | 93 | 2.753 | 12133.3 | 130.47 | 21.02 | 26478.0 | 91 |
| a-2 | greedy | 79 | 256 | 254 | 170 | 0.669 | 86 | 2.977 | 12993.0 | 151.08 | 19.63 | 8556.1 | 84 |
| a-2 | long | 2760 | 256 | 260 | 168 | 0.646 | 88 | 2.909 | 11566.3 | 131.44 | 22.05 | 26494.5 | 86 |

## The step per load, from the profile (typical steps)

**step, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| base | 115.36 (117.59; 81) | 123.07 (124.61; 88) | 119.22 |
| base-2 | 115.26 (117.32; 81) | 122.91 (123.70; 81) | 119.09 |
| a | 110.57 (113.09; 81) | 118.95 (120.68; 88) | 114.76 |
| a-2 | 110.43 (112.79; 81) | 118.89 (120.84; 83) | 114.66 |

**verify_span, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| base | 92.81 (92.99; 81) | 100.18 (100.31; 88) | 96.50 |
| base-2 | 92.82 (93.11; 81) | 100.07 (100.21; 81) | 96.44 |
| a | 92.87 (93.05; 81) | 100.04 (100.13; 88) | 96.46 |
| a-2 | 92.74 (92.92; 81) | 99.88 (99.99; 83) | 96.31 |

**verify_gpu, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| base | 88.24 (88.42; 81) | 95.35 (95.48; 88) | 91.80 |
| base-2 | 88.24 (88.46; 81) | 95.33 (95.42; 81) | 91.78 |
| a | 88.32 (88.47; 81) | 95.29 (95.36; 88) | 91.80 |
| a-2 | 88.17 (88.36; 81) | 95.10 (95.24; 83) | 91.63 |

**verify_rest, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| base | 4.56 (4.57; 81) | 4.80 (4.83; 88) | 4.68 |
| base-2 | 4.55 (4.65; 81) | 4.79 (4.79; 81) | 4.67 |
| a | 4.54 (4.58; 81) | 4.76 (4.77; 88) | 4.65 |
| a-2 | 4.51 (4.55; 81) | 4.74 (4.75; 83) | 4.62 |

**the draft's span, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| base | 22.54 (24.59; 81) | 22.94 (24.30; 88) | 22.74 |
| base-2 | 22.18 (24.21; 81) | 22.63 (23.48; 81) | 22.40 |
| a | 17.18 (20.04; 81) | 18.80 (20.54; 88) | 17.99 |
| a-2 | 17.27 (19.87; 81) | 18.73 (20.85; 83) | 18.00 |

**the draft's GPU time, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| base | 11.08 (13.31; 81) | 12.40 (14.09; 88) | 11.74 |
| base-2 | 11.08 (13.31; 81) | 12.37 (13.36; 81) | 11.72 |
| a | 11.06 (13.27; 81) | 12.42 (13.86; 88) | 11.74 |
| a-2 | 11.07 (13.28; 81) | 12.33 (14.17; 83) | 11.70 |

**the leads, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| base | 1.81 (1.81; 81) | 1.81 (1.82; 88) | 1.81 |
| base-2 | 1.80 (1.80; 81) | 1.81 (1.81; 81) | 1.80 |
| a | 1.81 (1.81; 81) | 1.81 (1.81; 88) | 1.81 |
| a-2 | 1.81 (1.82; 81) | 1.81 (1.82; 83) | 1.81 |

**the turns before the last, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| base | 1.46 (1.58; 81) | 1.50 (1.63; 88) | 1.48 |
| base-2 | 1.45 (1.58; 81) | 1.50 (1.56; 81) | 1.48 |
| a | 1.55 (1.70; 81) | 1.59 (1.70; 88) | 1.57 |
| a-2 | 1.61 (1.72; 81) | 1.62 (1.70; 83) | 1.62 |

**the last turn, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| base | 7.70 (7.90; 81) | 7.11 (6.76; 88) | 7.41 |
| base-2 | 7.37 (7.52; 81) | 6.79 (6.75; 81) | 7.08 |
| a | 2.71 (3.26; 81) | 2.96 (3.16; 88) | 2.84 |
| a-2 | 2.72 (3.05; 81) | 2.91 (3.15; 83) | 2.82 |

## Each arm against the reference loads

Reference: base and base-2. Difference = the load's value minus the mean of the two; spread = their difference.

| load | value | step, ms | minus the reference | reference spread | class |
|---|---|---:|---:|---:|---|
| a | mean of the medians | 114.76 | -4.39 | 0.13 (119.22, 119.09) | gain |
| a | greedy, median | 110.57 | -4.74 | 0.10 (115.36, 115.26) | gain |
| a | long, median | 118.95 | -4.04 | 0.16 (123.07, 122.91) | gain |
| a-2 | mean of the medians | 114.66 | -4.49 | 0.13 (119.22, 119.09) | gain |
| a-2 | greedy, median | 110.43 | -4.89 | 0.10 (115.36, 115.26) | gain |
| a-2 | long, median | 118.89 | -4.10 | 0.16 (123.07, 122.91) | gain |

## The greedy request, step by step against base

| load | same text as the reference | drafted | accepted | steps paired | step: mean difference, ms | standard error | median difference | the draft's span | verify_span | the leads | the turns before the last | the last turn |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | yes | 254 | 170 | — |  |  |  |  |  |  |  |  |
| base-2 | yes | 254 | 170 | 81 | -0.27 | 0.10 | -0.13 | -0.38 | +0.12 | -0.00 | +0.00 | -0.38 |
| a | yes | 254 | 170 | 81 | -4.50 | 0.20 | -4.79 | -4.55 | +0.05 | +0.00 | +0.12 | -4.64 |
| a-2 | yes | 254 | 170 | 81 | -4.80 | 0.17 | -4.90 | -4.72 | -0.08 | +0.01 | +0.14 | -4.85 |

