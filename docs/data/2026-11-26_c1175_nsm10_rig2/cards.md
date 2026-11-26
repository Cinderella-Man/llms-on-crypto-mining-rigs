# Cards: cards.csv (the lane's card log)

5,270 samples from 2026-11-26 11:04:01.321 to 2026-11-26 11:47:55.822 (2,634.5 s); 9 cards; a sample every 0.500 s (median), longest gap 0.59 s, 0 gaps over 2.0 s (0.0 s unlogged).
Busy: utilization above 0 % (a kernel ran in NVML's last sample period). Low: an SM clock below 1,800 MHz.

| card | samples | P0 s | P8 s | busy s | SM MHz busy, median | lowest | busy below 1,800 MHz | memory MHz busy, median | peak W | peak °C | most MiB | reasons, all samples | reasons, busy samples |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0 | 5,270 | 1,524.5 | 1,110.5 | 1,072.9 | 2,126 | 139 | 108 of 2,146 (5.03 %) | 4,657 | 82.83 | 39 | 5,180 | none 2,928 · idle 2,291 · power cap 51 | none 2,047 · power cap 51 · idle 48 |
| 1 | 5,270 | 1,657.0 | 978.0 | 1,094.9 | 2,151 | 139 | 105 of 2,190 (4.79 %) | 4,498 | 82.59 | 37 | 5,224 | none 3,153 · idle 2,063 · power cap 54 | none 2,083 · power cap 54 · idle 53 |
| 2 | 5,270 | 1,646.5 | 988.5 | 1,110.0 | 2,151 | 139 | 122 of 2,220 (5.50 %) | 4,498 | 82.82 | 32 | 5,082 | none 3,122 · idle 2,084 · power cap 64 | none 2,105 · power cap 64 · idle 51 |
| 3 | 5,270 | 1,644.0 | 991.0 | 1,120.0 | 2,100 | 139 | 107 of 2,240 (4.78 %) | 4,551 | 81.70 | 31 | 5,240 | none 3,127 · idle 2,090 · power cap 53 | none 2,134 · idle 54 · power cap 52 |
| 4 | 5,270 | 1,606.5 | 1,028.5 | 1,076.5 | 2,113 | 139 | 118 of 2,153 (5.48 %) | 4,551 | 81.84 | 36 | 4,338 | none 3,063 · idle 2,166 · power cap 41 | none 2,061 · idle 52 · power cap 40 |
| 5 | 5,270 | 1,648.0 | 987.0 | 1,115.0 | 2,100 | 139 | 128 of 2,230 (5.74 %) | 4,608 | 79.30 | 32 | 5,098 | none 3,135 · idle 2,088 · power cap 47 | none 2,127 · idle 57 · power cap 46 |
| 6 | 5,270 | 1,613.0 | 1,022.0 | 1,077.0 | 2,151 | 139 | 128 of 2,154 (5.94 %) | 4,455 | 87.19 | 41 | 4,496 | none 3,067 · idle 2,172 · power cap 31 | none 2,070 · idle 54 · power cap 30 |
| 7 | 5,270 | 1,598.5 | 1,036.5 | 1,090.0 | 2,151 | 139 | 126 of 2,180 (5.78 %) | 4,455 | 81.73 | 37 | 5,110 | none 3,042 · idle 2,167 · power cap 61 | none 2,071 · power cap 56 · idle 53 |
| 8 | 5,270 | 1,422.5 | 1,212.5 | 1,069.5 | 2,126 | 139 | 100 of 2,139 (4.68 %) | 4,657 | 83.29 | 35 | 5,858 | none 2,750 · idle 2,486 · power cap 34 | none 2,049 · idle 57 · power cap 33 |

## Busy stretches below 1,800 MHz, 2.0 s or longer

2 stretches.

| card | from | to | s | samples | lowest MHz | median MHz | P-states | reasons |
|---:|---|---|---:|---:|---:|---:|---|---|
| 2 | 2026-11-26 11:12:03.822 | 2026-11-26 11:12:05.822 | 2.0 | 4 | 1,354 | 1,354 | P0 | none |
| 0 | 2026-11-26 11:38:33.322 | 2026-11-26 11:38:35.322 | 2.0 | 4 | 1,354 | 1,354 | P0 | idle, none |

## Log events

- 2026-11-26 11:04:01.310 card_log start: pid 5639, driver 580.178.04, 9 cards (NVIDIA P106-090), interval 0.5 s; g0 limit 75 W, memory offset 1300, core offset 256; g1 limit 75 W, memory offset 1000, core offset 256; g2 limit 75 W, memory offset 1000, core offset 269; g3 limit 75 W, memory offset 1100, core offset 231; g4 limit 75 W, memory offset 1100, core offset 206; g5 limit 75 W, memory offset 1200, core offset 206; g6 limit 75 W, memory offset 900, core offset 282; g7 limit 75 W, memory offset 900, core offset 256; g8 limit 75 W, memory offset 1300, core offset 231
- 2026-11-26 11:47:55.981 card_log stop (signal): 5270 samples, 182.25 s CPU (user+sys), sampling 36.4 ms median, 584.5 ms max, 0 samples late by more than one interval; g0 limit 75 W, memory offset 1300, core offset 256; g1 limit 75 W, memory offset 1000, core offset 256; g2 limit 75 W, memory offset 1000, core offset 269; g3 limit 75 W, memory offset 1100, core offset 231; g4 limit 75 W, memory offset 1100, core offset 206; g5 limit 75 W, memory offset 1200, core offset 206; g6 limit 75 W, memory offset 900, core offset 282; g7 limit 75 W, memory offset 900, core offset 256; g8 limit 75 W, memory offset 1300, core offset 231
