## The server's own timings per reply

Steps by the server's counts: generated tokens minus accepted draft tokens (each verify pass gives one sampled token and the drafts it accepts). ms a step = the reply's generation time over that count.

| load | request | prompt tokens | generated | drafted | accepted | share accepted | steps (server) | tokens a step | generation ms | ms a step | tok/s | prompt ms | steps in the profile |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| warm | long | 2760 | 256 | 233 | 177 | 0.760 | 79 | 3.241 | 14306.3 | 181.09 | 17.82 | 30112.6 | — |
| warm | greedy | 79 | 256 | 255 | 170 | 0.667 | 86 | 2.977 | 11262.9 | 130.96 | 22.64 | 4948.9 | — |
| warm | smoke | 4 | 374 | 363 | 252 | 0.694 | 122 | 3.066 | 15597.0 | 127.84 | 23.91 | 1823.7 | — |
| d1off | long | 2760 | 256 | 275 | 162 | 0.589 | 94 | 2.723 | 16982.0 | 180.66 | 15.02 | 34591.9 | 92 |
| d1off | greedy | 79 | 256 | 255 | 170 | 0.667 | 86 | 2.977 | 11334.4 | 131.80 | 22.50 | 4893.6 | 85 |
| d1off | smoke | 4 | 343 | 363 | 221 | 0.609 | 122 | 2.811 | 15870.8 | 130.09 | 21.55 | 1826.6 | 120 |
| d1cache | long | 2760 | 256 | 273 | 164 | 0.601 | 92 | 2.783 | 19830.2 | 215.55 | 12.86 | 37358.9 | 90 |
| d1cache | greedy | 79 | 256 | 255 | 170 | 0.667 | 86 | 2.977 | 11261.7 | 130.95 | 22.64 | 5830.0 | 85 |
| d1cache | smoke | 4 | 377 | 345 | 262 | 0.759 | 115 | 3.278 | 14860.4 | 129.22 | 25.30 | 1818.2 | 114 |
| d1carry | long | 2760 | 53 | 51 | 36 | 0.706 | 17 | 3.118 | 3867.2 | 227.48 | 13.45 | 32846.7 | 16 |
| d1carry | greedy | 79 | 256 | 258 | 169 | 0.655 | 87 | 2.943 | 11426.1 | 131.33 | 22.32 | 5034.1 | 86 |
| d1carry | smoke | 4 | 309 | 321 | 201 | 0.626 | 108 | 2.861 | 13961.8 | 129.28 | 22.06 | 1836.0 | 106 |
| d1on | long | 2760 | 109 | 93 | 78 | 0.839 | 31 | 3.516 | 7730.9 | 249.38 | 13.97 | 33263.9 | 29 |
| d1on | greedy | 79 | 256 | 255 | 170 | 0.667 | 86 | 2.977 | 11247.6 | 130.79 | 22.67 | 4957.4 | 85 |
| d1on | smoke | 4 | 481 | 519 | 307 | 0.592 | 174 | 2.764 | 22317.5 | 128.26 | 21.51 | 1813.2 | 172 |
| d1off-2 | long | 2760 | 60 | 57 | 40 | 0.702 | 20 | 3.000 | 4120.0 | 206.00 | 14.32 | 33817.5 | 18 |
| d1off-2 | greedy | 79 | 256 | 255 | 170 | 0.667 | 86 | 2.977 | 11279.0 | 131.15 | 22.61 | 5083.9 | 85 |
| d1off-2 | smoke | 4 | 163 | 147 | 113 | 0.769 | 50 | 3.260 | 6339.8 | 126.80 | 25.55 | 1753.7 | 48 |

## The step per load, from the profile (typical steps)

**step, ms: median (mean) per request, and the mean of the three medians**

| load | long | greedy | smoke | mean of the medians |
|---|---:|---:|---:|---:|
| d1off | 140.46 (141.95; 88) | 131.57 (133.18; 82) | 129.91 (130.86; 117) | 133.98 |
| d1cache | 248.04 (204.02; 87) | 130.74 (132.27; 82) | 127.75 (128.82; 111) | 168.84 |
| d1carry | 140.60 (143.19; 13) | 130.61 (132.61; 83) | 129.06 (130.17; 103) | 133.43 |
| d1on | 246.04 (204.66; 26) | 130.18 (132.11; 82) | 128.43 (128.92; 169) | 168.22 |
| d1off-2 | 139.26 (141.04; 15) | 130.51 (132.50; 82) | 127.75 (128.78; 45) | 132.51 |

**verify_span, ms: median (mean) per request, and the mean of the three medians**

| load | long | greedy | smoke | mean of the medians |
|---|---:|---:|---:|---:|
| d1off | 114.43 (114.58; 88) | 106.88 (107.07; 82) | 107.02 (107.20; 117) | 109.44 |
| d1cache | 115.75 (115.86; 87) | 108.03 (108.17; 82) | 108.28 (108.46; 111) | 110.69 |
| d1carry | 113.39 (113.51; 13) | 106.83 (106.97; 83) | 106.81 (107.14; 103) | 109.01 |
| d1on | 115.29 (115.76; 26) | 107.98 (108.13; 82) | 108.17 (108.38; 169) | 110.48 |
| d1off-2 | 113.41 (114.10; 15) | 106.87 (107.03; 82) | 106.55 (106.53; 45) | 108.94 |

**verify_gpu, ms: median (mean) per request, and the mean of the three medians**

| load | long | greedy | smoke | mean of the medians |
|---|---:|---:|---:|---:|
| d1off | 109.23 (109.37; 88) | 101.91 (102.10; 82) | 102.08 (102.21; 117) | 104.41 |
| d1cache | 109.36 (109.43; 87) | 102.01 (102.14; 82) | 102.33 (102.41; 111) | 104.57 |
| d1carry | 108.26 (108.33; 13) | 101.88 (102.02; 83) | 101.84 (102.17; 103) | 103.99 |
| d1on | 108.95 (109.40; 26) | 101.98 (102.11; 82) | 102.11 (102.32; 169) | 104.35 |
| d1off-2 | 108.13 (108.90; 15) | 101.91 (102.02; 82) | 101.53 (101.56; 45) | 103.86 |

**verify_rest, ms: median (mean) per request, and the mean of the three medians**

| load | long | greedy | smoke | mean of the medians |
|---|---:|---:|---:|---:|
| d1off | 5.19 (5.21; 88) | 4.97 (4.98; 82) | 4.96 (4.99; 117) | 5.04 |
| d1cache | 6.37 (6.43; 87) | 6.02 (6.03; 82) | 6.01 (6.05; 111) | 6.14 |
| d1carry | 5.18 (5.18; 13) | 4.94 (4.95; 83) | 4.96 (4.97; 103) | 5.03 |
| d1on | 6.34 (6.35; 26) | 6.02 (6.02; 82) | 6.04 (6.05; 169) | 6.13 |
| d1off-2 | 5.21 (5.20; 15) | 4.94 (5.01; 82) | 4.97 (4.97; 45) | 5.04 |

**the draft's span, ms: median (mean) per request, and the mean of the three medians**

| load | long | greedy | smoke | mean of the medians |
|---|---:|---:|---:|---:|
| d1off | 26.06 (27.37; 88) | 24.11 (26.11; 82) | 23.09 (23.65; 117) | 24.42 |
| d1cache | 131.98 (88.16; 87) | 22.55 (24.10; 82) | 19.86 (20.36; 111) | 58.13 |
| d1carry | 26.48 (29.69; 13) | 23.47 (25.65; 83) | 22.37 (23.03; 103) | 24.11 |
| d1on | 131.86 (88.91; 26) | 21.97 (23.98; 82) | 20.30 (20.54; 169) | 58.04 |
| d1off-2 | 25.71 (26.94; 15) | 23.53 (25.47; 82) | 22.11 (22.25; 45) | 23.78 |

**the draft's GPU time, ms: median (mean) per request, and the mean of the three medians**

| load | long | greedy | smoke | mean of the medians |
|---|---:|---:|---:|---:|
| d1off | 13.32 (14.45; 88) | 11.48 (13.66; 82) | 11.54 (12.64; 117) | 12.12 |
| d1cache | 13.31 (16.96; 87) | 11.61 (14.03; 82) | 11.67 (12.56; 111) | 12.20 |
| d1carry | 13.15 (15.34; 13) | 11.65 (14.04; 83) | 11.61 (12.84; 103) | 12.14 |
| d1on | 13.36 (14.70; 26) | 11.69 (14.11; 82) | 11.69 (12.37; 169) | 12.25 |
| d1off-2 | 12.93 (13.07; 15) | 11.40 (13.59; 82) | 11.42 (12.00; 45) | 11.91 |

**the leads, ms: median (mean) per request, and the mean of the three medians**

| load | long | greedy | smoke | mean of the medians |
|---|---:|---:|---:|---:|
| d1off | 3.53 (3.54; 88) | 3.22 (3.27; 82) | 3.21 (3.21; 117) | 3.32 |
| d1cache | 0.00 (0.00; 87) | 0.00 (0.00; 82) | 0.00 (0.00; 111) | 0.00 |
| d1carry | 3.12 (3.14; 13) | 2.82 (2.82; 83) | 2.83 (2.83; 103) | 2.92 |
| d1on | 0.00 (0.00; 26) | 0.00 (0.00; 82) | 0.00 (0.00; 169) | 0.00 |
| d1off-2 | 3.51 (3.51; 15) | 3.21 (3.21; 82) | 3.19 (3.20; 45) | 3.30 |

**the turns before the last, ms: median (mean) per request, and the mean of the three medians**

| load | long | greedy | smoke | mean of the medians |
|---|---:|---:|---:|---:|
| d1off | 1.34 (1.39; 88) | 1.28 (1.36; 82) | 1.27 (1.33; 117) | 1.30 |
| d1cache | 3.39 (38.42; 87) | 2.71 (2.95; 82) | 2.59 (2.73; 111) | 2.89 |
| d1carry | 1.55 (3.55; 13) | 1.51 (1.60; 83) | 1.49 (1.53; 103) | 1.52 |
| d1on | 3.21 (3.15; 26) | 2.49 (2.66; 82) | 2.39 (2.51; 169) | 2.70 |
| d1off-2 | 1.35 (1.44; 15) | 1.29 (1.39; 82) | 1.28 (1.30; 45) | 1.30 |

**the last turn, ms: median (mean) per request, and the mean of the three medians**

| load | long | greedy | smoke | mean of the medians |
|---|---:|---:|---:|---:|
| d1off | 7.74 (8.00; 88) | 7.84 (7.82; 82) | 6.84 (6.48; 117) | 7.47 |
| d1cache | 7.90 (32.78; 87) | 7.40 (7.12; 82) | 5.41 (5.08; 111) | 6.90 |
| d1carry | 7.53 (7.66; 13) | 7.20 (7.18; 83) | 6.32 (5.82; 103) | 7.02 |
| d1on | 114.84 (71.05; 26) | 7.31 (7.21; 82) | 6.08 (5.66; 169) | 42.75 |
| d1off-2 | 7.91 (8.92; 15) | 7.48 (7.27; 82) | 6.31 (5.74; 45) | 7.23 |

## Each arm against the reference loads

Reference: d1off and d1off-2. Difference = the load's value minus the mean of the two; spread = their difference.

| load | value | step, ms | minus the reference | reference spread | class |
|---|---|---:|---:|---:|---|
| d1cache | mean of the medians | 168.84 | +35.60 | 1.47 (133.98, 132.51) | loss |
| d1cache | long, median | 248.04 | +108.18 | 1.20 (140.46, 139.26) | loss |
| d1cache | greedy, median | 130.74 | -0.30 | 1.06 (131.57, 130.51) | negligible |
| d1cache | smoke, median | 127.75 | -1.08 | 2.16 (129.91, 127.75) | negligible |
| d1carry | mean of the medians | 133.43 | +0.18 | 1.47 (133.98, 132.51) | negligible |
| d1carry | long, median | 140.60 | +0.74 | 1.20 (140.46, 139.26) | negligible |
| d1carry | greedy, median | 130.61 | -0.43 | 1.06 (131.57, 130.51) | negligible |
| d1carry | smoke, median | 129.06 | +0.23 | 2.16 (129.91, 127.75) | negligible |
| d1on | mean of the medians | 168.22 | +34.97 | 1.47 (133.98, 132.51) | loss |
| d1on | long, median | 246.04 | +106.18 | 1.20 (140.46, 139.26) | loss |
| d1on | greedy, median | 130.18 | -0.86 | 1.06 (131.57, 130.51) | negligible |
| d1on | smoke, median | 128.43 | -0.40 | 2.16 (129.91, 127.75) | negligible |

## The greedy request, step by step against d1off

| load | same text as the reference | drafted | accepted | steps paired | step: mean difference, ms | standard error | median difference | the draft's span | verify_span | the leads | the turns before the last | the last turn |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| warm | yes | 255 | 170 | — |  |  |  |  |  |  |  |  |
| d1off | yes | 255 | 170 | — |  |  |  |  |  |  |  |  |
| d1cache | yes | 255 | 170 | 82 | -0.91 | 0.17 | -0.81 | -2.01 | +1.09 | -3.27 | +1.59 | -0.70 |
| d1carry | NO | 258 | 169 | — |  |  |  |  |  |  |  |  |
| d1on | yes | 255 | 170 | 82 | -1.07 | 0.13 | -1.05 | -2.13 | +1.06 | -3.27 | +1.31 | -0.62 |
| d1off-2 | yes | 255 | 170 | 82 | -0.68 | 0.13 | -0.69 | -0.64 | -0.04 | -0.06 | +0.03 | -0.55 |

