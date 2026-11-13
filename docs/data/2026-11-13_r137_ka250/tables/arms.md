## The server's own timings per reply

Steps by the server's counts: generated tokens minus accepted draft tokens (each verify pass gives one sampled token and the drafts it accepts). ms a step = the reply's generation time over that count.

| load | request | prompt tokens | generated | drafted | accepted | share accepted | steps (server) | tokens a step | generation ms | ms a step | tok/s | prompt ms | steps in the profile |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| f1 | greedy | 79 | 256 | 254 | 170 | 0.669 | 86 | 2.977 | 10044.5 | 116.80 | 25.39 | 4287.7 | 83 |
| f1 | long | 2760 | 256 | 245 | 173 | 0.706 | 83 | 3.084 | 9859.0 | 118.78 | 25.86 | 24793.3 | 80 |
| f1-2 | greedy | 79 | 256 | 254 | 170 | 0.669 | 86 | 2.977 | 10095.4 | 117.39 | 25.26 | 4022.4 | 83 |
| f1-2 | long | 2760 | 256 | 263 | 167 | 0.635 | 89 | 2.876 | 10650.8 | 119.67 | 23.94 | 24751.8 | 86 |
| f1ka250 | greedy | 79 | 256 | 254 | 170 | 0.669 | 86 | 2.977 | 10111.1 | 117.57 | 25.22 | 3980.7 | 83 |
| f1ka250 | long | 2760 | 256 | 239 | 175 | 0.732 | 81 | 3.160 | 9925.8 | 122.54 | 25.69 | 24763.4 | 78 |
| f1ka250-2 | greedy | 79 | 256 | 254 | 170 | 0.669 | 86 | 2.977 | 10104.1 | 117.49 | 25.24 | 4030.6 | 83 |
| f1ka250-2 | long | 2760 | 256 | 261 | 167 | 0.640 | 89 | 2.876 | 10654.1 | 119.71 | 23.93 | 24782.8 | 86 |
| f0ka250 | greedy | 79 | 256 | 254 | 170 | 0.669 | 86 | 2.977 | 10064.2 | 117.03 | 25.34 | 4468.3 | 84 |
| f0ka250 | long | 2760 | 61 | 54 | 42 | 0.778 | 19 | 3.211 | 2407.5 | 126.71 | 24.92 | 24429.8 | 17 |
| f0ka250-2 | greedy | 79 | 256 | 254 | 170 | 0.669 | 86 | 2.977 | 10064.6 | 117.03 | 25.34 | 4430.4 | 84 |
| f0ka250-2 | long | 2760 | 256 | 253 | 169 | 0.668 | 87 | 2.943 | 10715.1 | 123.16 | 23.80 | 24486.4 | 85 |

## The step per load, from the profile (typical steps)

**step, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| f1 | 107.66 (110.15; 78) | 115.30 (115.99; 76) | 111.48 |
| f1-2 | 107.38 (109.78; 78) | 115.73 (117.37; 82) | 111.55 |
| f1ka250 | 107.61 (109.97; 78) | 116.46 (118.29; 73) | 112.04 |
| f1ka250-2 | 107.31 (109.84; 78) | 115.64 (117.37; 82) | 111.48 |
| f0ka250 | 107.48 (109.65; 80) | 114.75 (117.20; 14) | 111.12 |
| f0ka250-2 | 107.41 (109.68; 80) | 115.95 (118.59; 82) | 111.68 |

**verify_span, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| f1 | 91.32 (91.36; 78) | 98.31 (98.32; 76) | 94.81 |
| f1-2 | 91.37 (91.37; 78) | 98.61 (98.52; 82) | 94.99 |
| f1ka250 | 91.37 (91.38; 78) | 98.97 (99.08; 73) | 95.17 |
| f1ka250-2 | 91.31 (91.36; 78) | 98.49 (98.47; 82) | 94.90 |
| f0ka250 | 91.40 (91.52; 80) | 97.56 (97.64; 14) | 94.48 |
| f0ka250-2 | 91.36 (91.50; 80) | 98.67 (98.77; 82) | 95.01 |

**verify_gpu, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| f1 | 87.82 (87.87; 78) | 94.56 (94.61; 76) | 91.19 |
| f1-2 | 87.83 (87.87; 78) | 94.91 (94.79; 82) | 91.37 |
| f1ka250 | 87.86 (87.89; 78) | 95.28 (95.34; 73) | 91.57 |
| f1ka250-2 | 87.82 (87.86; 78) | 94.77 (94.73; 82) | 91.29 |
| f0ka250 | 87.77 (88.01; 80) | 93.86 (93.89; 14) | 90.81 |
| f0ka250-2 | 87.84 (88.00; 80) | 94.89 (95.01; 82) | 91.36 |

**verify_rest, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| f1 | 3.45 (3.49; 78) | 3.70 (3.71; 76) | 3.57 |
| f1-2 | 3.45 (3.49; 78) | 3.70 (3.73; 82) | 3.58 |
| f1ka250 | 3.45 (3.49; 78) | 3.71 (3.74; 73) | 3.58 |
| f1ka250-2 | 3.46 (3.50; 78) | 3.71 (3.74; 82) | 3.58 |
| f0ka250 | 3.45 (3.51; 80) | 3.69 (3.75; 14) | 3.57 |
| f0ka250-2 | 3.44 (3.51; 80) | 3.70 (3.76; 82) | 3.57 |

**the draft's span, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| f1 | 15.83 (18.79; 78) | 17.01 (17.68; 76) | 16.42 |
| f1-2 | 15.78 (18.41; 78) | 17.11 (18.85; 82) | 16.44 |
| f1ka250 | 15.92 (18.59; 78) | 17.04 (19.21; 73) | 16.48 |
| f1ka250-2 | 15.83 (18.48; 78) | 17.13 (18.90; 82) | 16.48 |
| f0ka250 | 15.82 (18.13; 80) | 17.17 (19.56; 14) | 16.50 |
| f0ka250-2 | 15.83 (18.17; 80) | 17.26 (19.82; 82) | 16.54 |

**the draft's GPU time, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| f1 | 10.78 (13.32; 78) | 12.15 (12.63; 76) | 11.46 |
| f1-2 | 10.78 (13.33; 78) | 12.18 (13.82; 82) | 11.48 |
| f1ka250 | 10.79 (13.31; 78) | 12.15 (14.05; 73) | 11.47 |
| f1ka250-2 | 10.78 (13.35; 78) | 12.17 (13.93; 82) | 11.48 |
| f0ka250 | 10.80 (12.94; 80) | 12.15 (13.93; 14) | 11.47 |
| f0ka250-2 | 10.77 (12.93; 80) | 12.17 (14.34; 82) | 11.47 |

**the leads, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| f1 | 4.22 (3.79; 78) | 4.48 (4.42; 76) | 4.35 |
| f1-2 | 4.22 (3.81; 78) | 4.49 (4.20; 82) | 4.36 |
| f1ka250 | 4.21 (3.81; 78) | 4.44 (4.24; 73) | 4.32 |
| f1ka250-2 | 4.22 (3.81; 78) | 4.48 (4.17; 82) | 4.35 |
| f0ka250 | 4.21 (3.83; 80) | 4.35 (4.01; 14) | 4.28 |
| f0ka250-2 | 4.20 (3.81; 80) | 4.45 (4.04; 82) | 4.32 |

**the turns before the last, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| f1 | -1.02 (-0.47; 78) | -1.28 (-1.19; 76) | -1.15 |
| f1-2 | -1.07 (-0.51; 78) | -1.29 (-0.93; 82) | -1.18 |
| f1ka250 | -0.99 (-0.47; 78) | -1.28 (-0.84; 73) | -1.14 |
| f1ka250-2 | -1.06 (-0.51; 78) | -1.27 (-0.87; 82) | -1.16 |
| f0ka250 | -1.04 (-0.52; 80) | -1.17 (-0.68; 14) | -1.11 |
| f0ka250-2 | -1.03 (-0.52; 80) | -1.28 (-0.72; 82) | -1.16 |

**the last turn, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| f1 | 1.80 (2.15; 78) | 1.67 (1.81; 76) | 1.74 |
| f1-2 | 1.74 (1.78; 78) | 1.68 (1.76; 82) | 1.71 |
| f1ka250 | 1.73 (1.94; 78) | 1.66 (1.76; 73) | 1.69 |
| f1ka250-2 | 1.77 (1.83; 78) | 1.72 (1.67; 82) | 1.74 |
| f0ka250 | 1.76 (1.87; 80) | 1.65 (2.30; 14) | 1.71 |
| f0ka250-2 | 1.76 (1.95; 80) | 1.76 (2.16; 82) | 1.76 |

## Each arm against the reference loads

Reference: f1 and f1-2. Difference = the load's value minus the mean of the two; spread = their difference.

| load | value | step, ms | minus the reference | reference spread | class |
|---|---|---:|---:|---:|---|
| f1ka250 | mean of the medians | 112.04 | +0.52 | 0.07 (111.48, 111.55) | loss |
| f1ka250 | greedy, median | 107.61 | +0.09 | 0.29 (107.66, 107.38) | negligible |
| f1ka250 | long, median | 116.46 | +0.95 | 0.43 (115.30, 115.73) | loss |
| f1ka250-2 | mean of the medians | 111.48 | -0.04 | 0.07 (111.48, 111.55) | negligible |
| f1ka250-2 | greedy, median | 107.31 | -0.21 | 0.29 (107.66, 107.38) | negligible |
| f1ka250-2 | long, median | 115.64 | +0.13 | 0.43 (115.30, 115.73) | negligible |
| f0ka250 | mean of the medians | 111.12 | -0.40 | 0.07 (111.48, 111.55) | gain |
| f0ka250 | greedy, median | 107.48 | -0.04 | 0.29 (107.66, 107.38) | negligible |
| f0ka250 | long, median | 114.75 | -0.76 | 0.43 (115.30, 115.73) | gain |
| f0ka250-2 | mean of the medians | 111.68 | +0.16 | 0.07 (111.48, 111.55) | loss |
| f0ka250-2 | greedy, median | 107.41 | -0.11 | 0.29 (107.66, 107.38) | negligible |
| f0ka250-2 | long, median | 115.95 | +0.43 | 0.43 (115.30, 115.73) | loss |

## The greedy request, step by step against f1

| load | same text as the reference | drafted | accepted | steps paired | step: mean difference, ms | standard error | median difference | the draft's span | verify_span | the leads | the turns before the last | the last turn |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| f1 | yes | 254 | 170 | — |  |  |  |  |  |  |  |  |
| f1-2 | yes | 254 | 170 | 78 | -0.37 | 0.19 | -0.05 | -0.38 | +0.01 | +0.02 | -0.04 | -0.37 |
| f1ka250 | yes | 254 | 170 | 78 | -0.18 | 0.22 | -0.02 | -0.20 | +0.02 | +0.02 | +0.01 | -0.21 |
| f1ka250-2 | yes | 254 | 170 | 78 | -0.31 | 0.19 | -0.04 | -0.31 | +0.00 | +0.02 | -0.04 | -0.32 |
| f0ka250 | yes | 254 | 170 | not paired (83 and 84 steps) |  |  |  |  |  |  |  |  |
| f0ka250-2 | yes | 254 | 170 | not paired (83 and 84 steps) |  |  |  |  |  |  |  |  |

