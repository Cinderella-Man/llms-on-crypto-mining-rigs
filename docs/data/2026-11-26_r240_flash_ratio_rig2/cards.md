# Cards: cards.csv

5,059 samples from 2026-11-26 15:56:54.718 to 2026-11-26 16:39:03.718 (2,529.0 s); 9 cards; a sample every 0.500 s (median), longest gap 0.51 s, 0 gaps over 2.0 s (0.0 s unlogged).
Busy: utilization above 0 % (a kernel ran in NVML's last sample period). Low: an SM clock below 1,800 MHz.

| card | samples | P0 s | P8 s | busy s | SM MHz busy, median | lowest | busy below 1,800 MHz | memory MHz busy, median | peak W | peak °C | most MiB | reasons, all samples | reasons, busy samples |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0 | 5,059 | 1,421.0 | 1,108.5 | 1,007.5 | 2,113 | 139 | 112 of 2,015 (5.56 %) | 4,657 | 83.47 | 46 | 5,180 | none 2,732 · idle 2,298 · power cap 29 | none 1,933 · idle 53 · power cap 29 |
| 1 | 5,059 | 1,553.5 | 976.0 | 1,012.5 | 2,138 | 139 | 105 of 2,025 (5.19 %) | 4,498 | 81.82 | 43 | 5,224 | none 2,956 · idle 2,073 · power cap 30 | none 1,941 · idle 54 · power cap 30 |
| 2 | 5,059 | 1,539.0 | 990.5 | 1,022.5 | 2,151 | 139 | 117 of 2,045 (5.72 %) | 4,498 | 83.29 | 39 | 5,082 | none 2,915 · idle 2,107 · power cap 37 | none 1,958 · idle 54 · power cap 33 |
| 3 | 5,059 | 1,542.0 | 987.5 | 1,024.0 | 2,100 | 139 | 114 of 2,048 (5.57 %) | 4,551 | 82.52 | 38 | 5,240 | none 2,936 · idle 2,100 · power cap 23 | none 1,972 · idle 53 · power cap 23 |
| 4 | 5,059 | 1,503.0 | 1,026.5 | 976.5 | 2,100 | 139 | 125 of 1,953 (6.40 %) | 4,551 | 82.57 | 43 | 4,338 | none 2,853 · idle 2,179 · power cap 27 | none 1,877 · idle 50 · power cap 26 |
| 5 | 5,059 | 1,542.5 | 987.0 | 1,017.0 | 2,100 | 139 | 118 of 2,034 (5.80 %) | 4,608 | 84.69 | 39 | 5,098 | none 2,917 · idle 2,111 · power cap 31 | none 1,955 · idle 48 · power cap 31 |
| 6 | 5,059 | 1,509.0 | 1,020.5 | 972.0 | 2,151 | 139 | 113 of 1,944 (5.81 %) | 4,455 | 86.69 | 49 | 4,496 | none 2,860 · idle 2,183 · power cap 16 | none 1,882 · idle 48 · power cap 14 |
| 7 | 5,059 | 1,485.0 | 1,044.5 | 988.5 | 2,138 | 139 | 107 of 1,977 (5.41 %) | 4,455 | 80.72 | 44 | 5,110 | none 2,846 · idle 2,191 · power cap 22 | none 1,905 · idle 50 · power cap 22 |
| 8 | 5,059 | 1,315.0 | 1,214.5 | 968.0 | 2,126 | 139 | 83 of 1,936 (4.29 %) | 4,657 | 81.89 | 41 | 5,698 | none 2,563 · idle 2,475 · power cap 21 | none 1,867 · idle 49 · power cap 20 |

## Busy stretches below 1,800 MHz, 2.0 s or longer

1 stretches.

| card | from | to | s | samples | lowest MHz | median MHz | P-states | reasons |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | 2026-11-26 16:23:42.218 | 2026-11-26 16:23:44.218 | 2.0 | 4 | 1,354 | 1,354 | P0 | none |

## Log events

- 2026-11-26 15:56:54.710 card_log start: pid 29374, driver 580.178.04, 9 cards (NVIDIA P106-090), interval 0.5 s; g0 limit 75 W, memory offset 1300, core offset 256; g1 limit 75 W, memory offset 1000, core offset 256; g2 limit 75 W, memory offset 1000, core offset 269; g3 limit 75 W, memory offset 1100, core offset 231; g4 limit 75 W, memory offset 1100, core offset 206; g5 limit 75 W, memory offset 1200, core offset 206; g6 limit 75 W, memory offset 900, core offset 282; g7 limit 75 W, memory offset 900, core offset 256; g8 limit 75 W, memory offset 1300, core offset 231
- 2026-11-26 16:39:04.031 card_log stop (signal): 5059 samples, 175.90 s CPU (user+sys), sampling 36.7 ms median, 436.5 ms max, 0 samples late by more than one interval; g0 limit 75 W, memory offset 1300, core offset 256; g1 limit 75 W, memory offset 1000, core offset 256; g2 limit 75 W, memory offset 1000, core offset 269; g3 limit 75 W, memory offset 1100, core offset 231; g4 limit 75 W, memory offset 1100, core offset 206; g5 limit 75 W, memory offset 1200, core offset 206; g6 limit 75 W, memory offset 900, core offset 282; g7 limit 75 W, memory offset 900, core offset 256; g8 limit 75 W, memory offset 1300, core offset 231
