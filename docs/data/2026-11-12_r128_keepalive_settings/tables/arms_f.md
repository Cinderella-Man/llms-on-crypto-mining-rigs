## The server's own timings per reply

Steps by the server's counts: generated tokens minus accepted draft tokens (each verify pass gives one sampled token and the drafts it accepts). ms a step = the reply's generation time over that count.

| load | request | prompt tokens | generated | drafted | accepted | share accepted | steps (server) | tokens a step | generation ms | ms a step | tok/s | prompt ms | steps in the profile |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| f | greedy | 79 | 256 | 254 | 170 | 0.669 | 86 | 2.977 | 10856.6 | 126.24 | 23.49 | 7522.2 | 84 |
| f | long | 2760 | 256 | 246 | 173 | 0.703 | 83 | 3.084 | 10624.5 | 128.01 | 24.00 | 26436.8 | 81 |
| f-2 | greedy | 79 | 256 | 254 | 170 | 0.669 | 86 | 2.977 | 10824.7 | 125.87 | 23.56 | 7545.3 | 84 |
| f-2 | long | 2760 | 256 | 273 | 163 | 0.597 | 93 | 2.753 | 12863.7 | 138.32 | 19.82 | 26304.4 | 91 |
| fka250 | greedy | 79 | 256 | 254 | 170 | 0.669 | 86 | 2.977 | 10094.1 | 117.37 | 25.26 | 4519.7 | 84 |
| fka250 | long | 2760 | 256 | 234 | 177 | 0.756 | 79 | 3.241 | 9535.1 | 120.70 | 26.74 | 24722.0 | 77 |
| feager | greedy | 79 | 256 | 254 | 170 | 0.669 | 86 | 2.977 | 19088.3 | 221.96 | 13.36 | 29483.2 | 83 |
| feager | long | 2760 | 256 | 256 | 169 | 0.660 | 87 | 2.943 | 10259.1 | 117.92 | 24.86 | 21836.7 | 85 |

## The step per load, from the profile (typical steps)

**step, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| f | 108.09 (110.30; 80) | 115.68 (116.81; 78) | 111.89 |
| f-2 | 107.92 (110.20; 80) | 116.06 (117.43; 88) | 111.99 |
| fka250 | 107.94 (110.01; 80) | 116.61 (118.56; 74) | 112.28 |
| feager | 214.93 (205.45; 79) | 116.48 (118.19; 82) | 165.70 |

**verify_span, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| f | 91.87 (92.04; 80) | 98.85 (99.10; 78) | 95.36 |
| f-2 | 91.74 (91.94; 80) | 98.82 (99.05; 88) | 95.28 |
| fka250 | 91.61 (91.77; 80) | 98.83 (99.02; 74) | 95.22 |
| feager | 91.76 (95.44; 79) | 98.82 (98.94; 82) | 95.29 |

**verify_gpu, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| f | 88.27 (88.47; 80) | 95.15 (95.38; 78) | 91.71 |
| f-2 | 88.22 (88.42; 80) | 95.05 (95.31; 88) | 91.64 |
| fka250 | 88.12 (88.27; 80) | 95.06 (95.27; 74) | 91.59 |
| feager | 88.16 (88.39; 79) | 95.11 (95.18; 82) | 91.63 |

**verify_rest, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| f | 3.44 (3.57; 80) | 3.70 (3.72; 78) | 3.57 |
| f-2 | 3.45 (3.52; 80) | 3.70 (3.74; 88) | 3.57 |
| fka250 | 3.44 (3.50; 80) | 3.70 (3.75; 74) | 3.57 |
| feager | 3.47 (7.05; 79) | 3.72 (3.76; 82) | 3.60 |

**the draft's span, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| f | 16.00 (18.27; 80) | 16.86 (17.70; 78) | 16.43 |
| f-2 | 16.00 (18.27; 80) | 17.24 (18.38; 88) | 16.62 |
| fka250 | 15.95 (18.24; 80) | 17.33 (19.55; 74) | 16.64 |
| feager | 121.01 (110.01; 79) | 17.47 (19.25; 82) | 69.24 |

**the draft's GPU time, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| f | 10.83 (12.99; 80) | 12.18 (12.55; 78) | 11.50 |
| f-2 | 10.83 (12.99; 80) | 12.22 (13.31; 88) | 11.53 |
| fka250 | 10.85 (13.01; 80) | 12.22 (14.18; 74) | 11.53 |
| feager | 10.89 (13.30; 79) | 12.29 (13.79; 82) | 11.59 |

**the leads, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| f | 4.21 (3.84; 80) | 4.46 (4.40; 78) | 4.33 |
| f-2 | 4.21 (3.82; 80) | 4.48 (4.24; 88) | 4.35 |
| fka250 | 4.23 (3.83; 80) | 4.49 (4.16; 74) | 4.36 |
| feager | 3.79 (3.32; 79) | 4.42 (3.63; 82) | 4.10 |

**the turns before the last, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| f | -1.05 (-0.52; 80) | -1.32 (-1.21; 78) | -1.19 |
| f-2 | -1.07 (-0.53; 80) | -1.31 (-1.01; 88) | -1.19 |
| fka250 | -1.07 (-0.54; 80) | -1.28 (-0.77; 74) | -1.18 |
| feager | -0.69 (42.76; 79) | -1.23 (-0.18; 82) | -0.96 |

**the last turn, ms: median (mean) per request, and the mean of the three medians**

| load | greedy | long | mean of the medians |
|---|---:|---:|---:|
| f | 1.84 (1.95; 80) | 1.39 (1.96; 78) | 1.61 |
| f-2 | 1.83 (1.98; 80) | 1.80 (1.84; 88) | 1.81 |
| fka250 | 1.81 (1.94; 80) | 1.75 (1.99; 74) | 1.78 |
| feager | 2.90 (50.63; 79) | 1.83 (2.01; 82) | 2.37 |

## Each arm against the reference loads

Reference: f and f-2. Difference = the load's value minus the mean of the two; spread = their difference.

| load | value | step, ms | minus the reference | reference spread | class |
|---|---|---:|---:|---:|---|
| fka250 | mean of the medians | 112.28 | +0.34 | 0.10 (111.89, 111.99) | loss |
| fka250 | greedy, median | 107.94 | -0.07 | 0.17 (108.09, 107.92) | negligible |
| fka250 | long, median | 116.61 | +0.74 | 0.37 (115.68, 116.06) | loss |
| feager | mean of the medians | 165.70 | +53.76 | 0.10 (111.89, 111.99) | loss |
| feager | greedy, median | 214.93 | +106.92 | 0.17 (108.09, 107.92) | loss |
| feager | long, median | 116.48 | +0.61 | 0.37 (115.68, 116.06) | loss |

## The greedy request, step by step against f

| load | same text as the reference | drafted | accepted | steps paired | step: mean difference, ms | standard error | median difference | the draft's span | verify_span | the leads | the turns before the last | the last turn |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| f | yes | 254 | 170 | — |  |  |  |  |  |  |  |  |
| f-2 | yes | 254 | 170 | 80 | -0.10 | 0.15 | -0.01 | -0.00 | -0.10 | -0.03 | -0.01 | +0.03 |
| fka250 | yes | 254 | 170 | 80 | -0.29 | 0.13 | -0.19 | -0.03 | -0.26 | -0.01 | -0.02 | -0.01 |
| feager | yes | 254 | 170 | not paired (84 and 83 steps) |  |  |  |  |  |  |  |  |

