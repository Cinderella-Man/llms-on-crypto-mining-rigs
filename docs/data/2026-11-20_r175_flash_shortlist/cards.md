# Cards: cards.csv (the lane's card log)

3,425 samples from 2026-11-20 18:58:30.697 to 2026-11-20 19:27:02.697 (1,712.0 s); 9 cards; a sample every 0.500 s (median), longest gap 0.52 s, 0 gaps over 2.0 s (0.0 s unlogged).
Busy: utilization above 0 % (a kernel ran in NVML's last sample period). Low: an SM clock below 1,800 MHz.

| card | samples | P0 s | P8 s | busy s | SM MHz busy, median | lowest | busy below 1,800 MHz | memory MHz busy, median | peak W | peak °C | most MiB | reasons, all samples | reasons, busy samples |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0 | 3,425 | 912.0 | 800.5 | 629.0 | 2,126 | 139 | 74 of 1,258 (5.88 %) | 4,657 | 76.98 | 38 | 5,180 | none 1,756 · idle 1,639 · power cap 30 | none 1,194 · idle 35 · power cap 29 |
| 1 | 3,425 | 1,014.5 | 698.0 | 640.5 | 2,151 | 139 | 77 of 1,281 (6.01 %) | 4,498 | 82.02 | 34 | 5,224 | none 1,934 · idle 1,464 · power cap 27 | none 1,217 · idle 37 · power cap 27 |
| 2 | 3,425 | 1,006.5 | 706.0 | 650.5 | 2,164 | 139 | 89 of 1,301 (6.84 %) | 4,498 | 81.74 | 30 | 5,082 | none 1,906 · idle 1,490 · power cap 29 | none 1,230 · idle 42 · power cap 29 |
| 3 | 3,425 | 1,005.0 | 707.5 | 652.0 | 2,100 | 139 | 75 of 1,304 (5.75 %) | 4,551 | 82.21 | 28 | 5,240 | none 1,911 · idle 1,491 · power cap 23 | none 1,243 · idle 38 · power cap 23 |
| 4 | 3,425 | 977.5 | 735.0 | 624.0 | 2,113 | 139 | 77 of 1,248 (6.17 %) | 4,551 | 77.22 | 33 | 4,338 | none 1,861 · idle 1,544 · power cap 20 | none 1,193 · idle 36 · power cap 19 |
| 5 | 3,425 | 1,006.5 | 706.0 | 646.5 | 2,113 | 139 | 79 of 1,293 (6.11 %) | 4,608 | 81.47 | 30 | 5,098 | none 1,915 · idle 1,486 · power cap 24 | none 1,235 · idle 34 · power cap 24 |
| 6 | 3,425 | 980.5 | 732.0 | 618.0 | 2,164 | 139 | 87 of 1,236 (7.04 %) | 4,455 | 78.95 | 39 | 4,496 | none 1,869 · idle 1,546 · power cap 10 | none 1,191 · idle 37 · power cap 8 |
| 7 | 3,425 | 967.0 | 745.5 | 635.5 | 2,164 | 139 | 78 of 1,271 (6.14 %) | 4,455 | 81.68 | 34 | 5,110 | none 1,842 · idle 1,553 · power cap 30 | none 1,205 · idle 36 · power cap 30 |
| 8 | 3,425 | 842.5 | 870.0 | 625.0 | 2,138 | 139 | 68 of 1,250 (5.44 %) | 4,657 | 81.38 | 32 | 5,858 | idle 1,782 · none 1,627 · power cap 16 | none 1,197 · idle 37 · power cap 16 |

## Busy stretches below 1,800 MHz, 2.0 s or longer

2 stretches.

| card | from | to | s | samples | lowest MHz | median MHz | P-states | reasons |
|---:|---|---|---:|---:|---:|---:|---|---|
| 2 | 2026-11-20 19:12:47.197 | 2026-11-20 19:12:49.200 | 2.0 | 4 | 1,075 | 1,354 | P0 | idle, none |
| 0 | 2026-11-20 19:12:47.697 | 2026-11-20 19:12:49.697 | 2.0 | 4 | 1,354 | 1,354 | P0 | idle, none |

## Log events

- 2026-11-20 18:58:30.689 card_log start: pid 45563, driver 580.178.04, 9 cards (NVIDIA P106-090), interval 0.5 s; g0 limit 75 W, memory offset 1300, core offset 256; g1 limit 75 W, memory offset 1000, core offset 256; g2 limit 75 W, memory offset 1000, core offset 269; g3 limit 75 W, memory offset 1100, core offset 231; g4 limit 75 W, memory offset 1100, core offset 206; g5 limit 75 W, memory offset 1200, core offset 206; g6 limit 75 W, memory offset 900, core offset 282; g7 limit 75 W, memory offset 900, core offset 256; g8 limit 75 W, memory offset 1300, core offset 231
- 2026-11-20 19:27:03.196 card_log stop (signal): 3425 samples, 120.29 s CPU (user+sys), sampling 36.5 ms median, 289.7 ms max, 0 samples late by more than one interval; g0 limit 75 W, memory offset 1300, core offset 256; g1 limit 75 W, memory offset 1000, core offset 256; g2 limit 75 W, memory offset 1000, core offset 269; g3 limit 75 W, memory offset 1100, core offset 231; g4 limit 75 W, memory offset 1100, core offset 206; g5 limit 75 W, memory offset 1200, core offset 206; g6 limit 75 W, memory offset 900, core offset 282; g7 limit 75 W, memory offset 900, core offset 256; g8 limit 75 W, memory offset 1300, core offset 231
