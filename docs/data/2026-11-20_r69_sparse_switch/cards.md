# Cards: cards.csv (the lane's card log)

6,272 samples from 2026-11-20 10:10:03.959 to 2026-11-20 11:02:19.460 (3,135.5 s); 9 cards; a sample every 0.500 s (median), longest gap 0.51 s, 0 gaps over 2.0 s (0.0 s unlogged).
Busy: utilization above 0 % (a kernel ran in NVML's last sample period). Low: an SM clock below 1,800 MHz.

| card | samples | P0 s | P8 s | busy s | SM MHz busy, median | lowest | busy below 1,800 MHz | memory MHz busy, median | peak W | peak °C | most MiB | reasons, all samples | reasons, busy samples |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0 | 6,272 | 2,017.5 | 1,118.5 | 1,394.5 | 2,113 | 139 | 129 of 2,789 (4.63 %) | 4,657 | 83.11 | 51 | 5,178 | none 3,819 · idle 2,291 · power cap 162 | none 2,582 · power cap 160 · idle 47 |
| 1 | 6,272 | 2,155.0 | 981.0 | 1,450.0 | 2,138 | 139 | 132 of 2,900 (4.55 %) | 4,498 | 85.21 | 46 | 5,222 | none 4,083 · idle 2,044 · power cap 145 | none 2,708 · power cap 144 · idle 48 |
| 2 | 6,272 | 2,143.5 | 992.5 | 1,488.5 | 2,138 | 139 | 152 of 2,977 (5.11 %) | 4,498 | 83.95 | 41 | 5,078 | none 4,032 · idle 2,082 · power cap 158 | none 2,764 · power cap 157 · idle 56 |
| 3 | 6,272 | 2,150.5 | 985.5 | 1,518.0 | 2,088 | 139 | 133 of 3,036 (4.38 %) | 4,551 | 81.96 | 40 | 5,238 | none 4,035 · idle 2,078 · power cap 159 | none 2,829 · power cap 158 · idle 49 |
| 4 | 6,272 | 2,108.5 | 1,027.5 | 1,469.0 | 2,100 | 139 | 192 of 2,938 (6.54 %) | 4,551 | 84.08 | 45 | 4,336 | none 3,997 · idle 2,157 · power cap 118 | none 2,769 · power cap 115 · idle 54 |
| 5 | 6,272 | 2,147.0 | 989.0 | 1,514.5 | 2,088 | 139 | 161 of 3,029 (5.32 %) | 4,608 | 83.19 | 41 | 5,094 | none 4,021 · idle 2,082 · power cap 169 | none 2,818 · power cap 162 · idle 49 |
| 6 | 6,272 | 2,119.0 | 1,017.0 | 1,466.5 | 2,151 | 139 | 178 of 2,933 (6.07 %) | 4,455 | 87.75 | 50 | 4,494 | none 4,023 · idle 2,161 · power cap 88 | none 2,796 · power cap 85 · idle 52 |
| 7 | 6,272 | 2,094.5 | 1,041.5 | 1,482.0 | 2,138 | 139 | 137 of 2,964 (4.62 %) | 4,455 | 83.54 | 45 | 5,106 | none 3,938 · idle 2,169 · power cap 165 | none 2,757 · power cap 160 · idle 47 |
| 8 | 6,272 | 1,920.5 | 1,215.5 | 1,413.5 | 2,113 | 139 | 118 of 2,827 (4.17 %) | 4,657 | 84.17 | 42 | 6,018 | none 3,723 · idle 2,479 · power cap 70 | none 2,713 · power cap 67 · idle 47 |

## Busy stretches below 1,800 MHz, 2.0 s or longer

2 stretches.

| card | from | to | s | samples | lowest MHz | median MHz | P-states | reasons |
|---:|---|---|---:|---:|---:|---:|---|---|
| 0 | 2026-11-20 10:18:00.960 | 2026-11-20 10:18:02.960 | 2.0 | 4 | 1,354 | 1,354 | P0 | none |
| 2 | 2026-11-20 10:18:00.960 | 2026-11-20 10:18:02.960 | 2.0 | 4 | 1,354 | 1,354 | P0 | none |

## Log events

- 2026-11-20 10:10:03.952 card_log start: pid 4156, driver 580.178.04, 9 cards (NVIDIA P106-090), interval 0.5 s; g0 limit 75 W, memory offset 1300, core offset 256; g1 limit 75 W, memory offset 1000, core offset 256; g2 limit 75 W, memory offset 1000, core offset 269; g3 limit 75 W, memory offset 1100, core offset 231; g4 limit 75 W, memory offset 1100, core offset 206; g5 limit 75 W, memory offset 1200, core offset 206; g6 limit 75 W, memory offset 900, core offset 282; g7 limit 75 W, memory offset 900, core offset 256; g8 limit 75 W, memory offset 1300, core offset 231
- 2026-11-20 11:02:19.978 card_log stop (signal): 6272 samples, 220.54 s CPU (user+sys), sampling 37.8 ms median, 385.5 ms max, 0 samples late by more than one interval; g0 limit 75 W, memory offset 1300, core offset 256; g1 limit 75 W, memory offset 1000, core offset 256; g2 limit 75 W, memory offset 1000, core offset 269; g3 limit 75 W, memory offset 1100, core offset 231; g4 limit 75 W, memory offset 1100, core offset 206; g5 limit 75 W, memory offset 1200, core offset 206; g6 limit 75 W, memory offset 900, core offset 282; g7 limit 75 W, memory offset 900, core offset 256; g8 limit 75 W, memory offset 1300, core offset 231
