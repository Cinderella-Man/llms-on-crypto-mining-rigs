# Cards: cards.csv

6,590 samples from 2026-11-26 19:16:29.126 to 2026-11-26 20:11:23.627 (3,294.5 s); 9 cards; a sample every 0.500 s (median), longest gap 0.51 s, 0 gaps over 2.0 s (0.0 s unlogged).
Busy: utilization above 0 % (a kernel ran in NVML's last sample period). Low: an SM clock below 1,800 MHz.

| card | samples | P0 s | P8 s | busy s | SM MHz busy, median | lowest | busy below 1,800 MHz | memory MHz busy, median | peak W | peak °C | most MiB | reasons, all samples | reasons, busy samples |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0 | 6,590 | 1,874.5 | 1,420.5 | 1,361.5 | 2,113 | 139 | 134 of 2,723 (4.92 %) | 4,657 | 79.22 | 49 | 5,180 | none 3,609 · idle 2,938 · power cap 43 | none 2,619 · idle 61 · power cap 43 |
| 1 | 6,590 | 2,039.5 | 1,255.5 | 1,365.0 | 2,138 | 139 | 135 of 2,730 (4.95 %) | 4,498 | 86.29 | 46 | 5,224 | none 3,902 · idle 2,642 · power cap 46 | none 2,622 · idle 63 · power cap 45 |
| 2 | 6,590 | 2,024.5 | 1,270.5 | 1,374.5 | 2,151 | 139 | 145 of 2,749 (5.27 %) | 4,498 | 83.58 | 42 | 5,082 | none 3,882 · idle 2,677 · power cap 31 | none 2,654 · idle 65 · power cap 30 |
| 3 | 6,590 | 2,022.5 | 1,272.5 | 1,369.0 | 2,088 | 139 | 141 of 2,738 (5.15 %) | 4,551 | 79.59 | 40 | 5,240 | none 3,884 · idle 2,686 · power cap 20 | none 2,656 · idle 62 · power cap 20 |
| 4 | 6,590 | 1,976.5 | 1,318.5 | 1,310.5 | 2,100 | 139 | 147 of 2,621 (5.61 %) | 4,551 | 80.45 | 46 | 4,338 | none 3,758 · idle 2,786 · power cap 46 | none 2,516 · idle 60 · power cap 45 |
| 5 | 6,590 | 2,025.0 | 1,270.0 | 1,365.5 | 2,088 | 139 | 160 of 2,731 (5.86 %) | 4,608 | 78.77 | 42 | 5,098 | none 3,865 · idle 2,692 · power cap 33 | none 2,632 · idle 66 · power cap 33 |
| 6 | 6,590 | 1,986.0 | 1,309.0 | 1,313.5 | 2,138 | 139 | 154 of 2,627 (5.86 %) | 4,455 | 84.32 | 51 | 4,496 | none 3,776 · idle 2,782 · power cap 32 | none 2,529 · idle 67 · power cap 31 |
| 7 | 6,590 | 1,958.0 | 1,337.0 | 1,336.5 | 2,138 | 139 | 150 of 2,673 (5.61 %) | 4,455 | 76.89 | 47 | 5,110 | none 3,754 · idle 2,794 · power cap 42 | none 2,563 · idle 68 · power cap 42 |
| 8 | 6,590 | 1,735.0 | 1,560.0 | 1,324.5 | 2,126 | 139 | 127 of 2,649 (4.79 %) | 4,657 | 81.70 | 43 | 5,698 | none 3,370 · idle 3,204 · power cap 16 | none 2,563 · idle 70 · power cap 16 |

## Busy stretches below 1,800 MHz, 2.0 s or longer

3 stretches.

| card | from | to | s | samples | lowest MHz | median MHz | P-states | reasons |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | 2026-11-26 19:37:01.127 | 2026-11-26 19:37:03.127 | 2.0 | 4 | 1,354 | 1,354 | P0 | none |
| 2 | 2026-11-26 19:37:01.127 | 2026-11-26 19:37:03.127 | 2.0 | 4 | 1,354 | 1,354 | P0 | none |
| 1 | 2026-11-26 19:49:37.627 | 2026-11-26 19:49:39.627 | 2.0 | 4 | 1,354 | 1,354 | P0 | none |

## Log events

- 2026-11-26 19:16:29.119 card_log start: pid 41890, driver 580.178.04, 9 cards (NVIDIA P106-090), interval 0.5 s; g0 limit 75 W, memory offset 1300, core offset 256; g1 limit 75 W, memory offset 1000, core offset 256; g2 limit 75 W, memory offset 1000, core offset 269; g3 limit 75 W, memory offset 1100, core offset 231; g4 limit 75 W, memory offset 1100, core offset 206; g5 limit 75 W, memory offset 1200, core offset 206; g6 limit 75 W, memory offset 900, core offset 282; g7 limit 75 W, memory offset 900, core offset 256; g8 limit 75 W, memory offset 1300, core offset 231
- 2026-11-26 20:11:23.972 card_log stop (signal): 6590 samples, 226.85 s CPU (user+sys), sampling 36.6 ms median, 323.9 ms max, 0 samples late by more than one interval; g0 limit 75 W, memory offset 1300, core offset 256; g1 limit 75 W, memory offset 1000, core offset 256; g2 limit 75 W, memory offset 1000, core offset 269; g3 limit 75 W, memory offset 1100, core offset 231; g4 limit 75 W, memory offset 1100, core offset 206; g5 limit 75 W, memory offset 1200, core offset 206; g6 limit 75 W, memory offset 900, core offset 282; g7 limit 75 W, memory offset 900, core offset 256; g8 limit 75 W, memory offset 1300, core offset 231
