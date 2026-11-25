# Cards: cards.csv

4,338 samples from 2026-11-25 18:09:01.641 to 2026-11-25 18:45:10.141 (2,168.5 s); 9 cards; a sample every 0.500 s (median), longest gap 0.51 s, 0 gaps over 2.0 s (0.0 s unlogged).
Busy: utilization above 0 % (a kernel ran in NVML's last sample period). Low: an SM clock below 1,800 MHz.

| card | samples | P0 s | P8 s | busy s | SM MHz busy, median | lowest | busy below 1,800 MHz | memory MHz busy, median | peak W | peak °C | most MiB | reasons, all samples | reasons, busy samples |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0 | 4,338 | 1,015.0 | 1,154.0 | 708.0 | 1,847 | 139 | 146 of 1,416 (10.31 %) | 4,455 | 76.29 | 44 | 5,176 | idle 2,365 · none 1,939 · power cap 34 | none 1,344 · idle 38 · power cap 34 |
| 1 | 4,338 | 1,103.5 | 1,065.5 | 718.5 | 1,847 | 139 | 169 of 1,437 (11.76 %) | 4,498 | 80.13 | 48 | 5,222 | idle 2,217 · none 2,079 · power cap 42 | none 1,352 · idle 43 · power cap 42 |
| 2 | 4,338 | 1,100.5 | 1,068.5 | 723.0 | 1,898 | 139 | 133 of 1,446 (9.20 %) | 4,551 | 77.18 | 40 | 5,080 | idle 2,223 · none 2,085 · power cap 30 | none 1,376 · idle 40 · power cap 30 |
| 3 | 4,338 | 1,104.5 | 1,064.5 | 728.0 | 1,885 | 139 | 143 of 1,456 (9.82 %) | 4,608 | 79.73 | 46 | 5,238 | idle 2,219 · none 2,082 · power cap 37 | none 1,377 · idle 42 · power cap 37 |
| 4 | 4,338 | 1,069.5 | 1,099.5 | 705.0 | 1,873 | 139 | 144 of 1,410 (10.21 %) | 4,551 | 79.06 | 39 | 4,336 | idle 2,290 · none 2,026 · power cap 22 | none 1,345 · idle 44 · power cap 21 |
| 5 | 4,338 | 1,098.5 | 1,070.5 | 729.5 | 1,911 | 139 | 127 of 1,459 (8.70 %) | 4,657 | 80.53 | 40 | 5,096 | idle 2,226 · none 2,075 · power cap 37 | none 1,382 · idle 41 · power cap 36 |
| 6 | 4,338 | 1,071.5 | 1,097.5 | 694.5 | 1,873 | 139 | 118 of 1,389 (8.50 %) | 4,006 | 78.19 | 40 | 4,494 | idle 2,287 · none 2,037 · power cap 14 | none 1,335 · idle 40 · power cap 14 |
| 7 | 4,338 | 1,091.5 | 1,077.5 | 715.5 | 1,911 | 139 | 144 of 1,431 (10.06 %) | 4,551 | 79.83 | 43 | 5,108 | idle 2,237 · none 2,066 · power cap 35 | none 1,351 · idle 47 · power cap 33 |
| 8 | 4,338 | 942.0 | 1,227.0 | 695.5 | 1,873 | 139 | 84 of 1,391 (6.04 %) | 4,455 | 78.17 | 43 | 5,856 | idle 2,491 · none 1,827 · power cap 20 | none 1,332 · idle 39 · power cap 20 |

## Busy stretches below 1,800 MHz, 2.0 s or longer

30 stretches.

| card | from | to | s | samples | lowest MHz | median MHz | P-states | reasons |
|---:|---|---|---:|---:|---:|---:|---|---|
| 0 | 2026-11-25 18:20:58.641 | 2026-11-25 18:21:00.642 | 2.0 | 4 | 1,784 | 1,784 | P0 | none, power cap |
| 1 | 2026-11-25 18:20:58.641 | 2026-11-25 18:21:01.141 | 2.5 | 5 | 1,746 | 1,759 | P0 | none, power cap |
| 4 | 2026-11-25 18:21:01.141 | 2026-11-25 18:21:03.643 | 2.5 | 5 | 1,771 | 1,797 | P0 | none, power cap |
| 7 | 2026-11-25 18:21:03.142 | 2026-11-25 18:21:05.641 | 2.5 | 5 | 1,708 | 1,784 | P0 | none, power cap |
| 1 | 2026-11-25 18:21:37.142 | 2026-11-25 18:21:40.142 | 3.0 | 6 | 1,683 | 1,765 | P0 | none, power cap |
| 2 | 2026-11-25 18:21:39.141 | 2026-11-25 18:21:41.141 | 2.0 | 4 | 1,708 | 1,734 | P0 | none, power cap |
| 7 | 2026-11-25 18:21:41.641 | 2026-11-25 18:21:43.641 | 2.0 | 4 | 1,733 | 1,746 | P0 | none, power cap |
| 1 | 2026-11-25 18:28:25.641 | 2026-11-25 18:28:27.642 | 2.0 | 4 | 1,733 | 1,746 | P0 | none, power cap |
| 1 | 2026-11-25 18:28:32.143 | 2026-11-25 18:28:35.141 | 3.0 | 6 | 1,708 | 1,752 | P0 | none, power cap |
| 2 | 2026-11-25 18:28:32.641 | 2026-11-25 18:28:35.141 | 2.5 | 5 | 1,721 | 1,759 | P0 | none, power cap |
| 5 | 2026-11-25 18:28:34.641 | 2026-11-25 18:28:36.642 | 2.0 | 4 | 1,733 | 1,772 | P0 | none, power cap |
| 0 | 2026-11-25 18:29:16.642 | 2026-11-25 18:29:18.642 | 2.0 | 4 | 1,708 | 1,740 | P0 | none, power cap |
| 1 | 2026-11-25 18:29:16.642 | 2026-11-25 18:29:19.641 | 3.0 | 6 | 1,670 | 1,727 | P0 | none, power cap |
| 7 | 2026-11-25 18:29:21.141 | 2026-11-25 18:29:24.141 | 3.0 | 6 | 1,670 | 1,778 | P0 | none |
| 0 | 2026-11-25 18:36:04.641 | 2026-11-25 18:36:07.141 | 2.5 | 5 | 1,759 | 1,784 | P0 | none, power cap |
| 1 | 2026-11-25 18:36:05.142 | 2026-11-25 18:36:08.141 | 3.0 | 6 | 1,746 | 1,765 | P0 | none, power cap |
| 0 | 2026-11-25 18:36:43.141 | 2026-11-25 18:36:46.641 | 3.5 | 7 | 1,721 | 1,746 | P0 | none, power cap |
| 1 | 2026-11-25 18:36:43.641 | 2026-11-25 18:36:48.141 | 4.5 | 9 | 1,670 | 1,721 | P0 | none, power cap |
| 2 | 2026-11-25 18:36:44.642 | 2026-11-25 18:36:48.141 | 3.5 | 7 | 1,721 | 1,771 | P0 | none, power cap |
| 7 | 2026-11-25 18:36:49.642 | 2026-11-25 18:36:52.142 | 2.5 | 5 | 1,695 | 1,759 | P0 | none |
| 7 | 2026-11-25 18:36:55.641 | 2026-11-25 18:36:57.641 | 2.0 | 4 | 1,695 | 1,720 | P0 | none |
| 1 | 2026-11-25 18:43:39.141 | 2026-11-25 18:43:42.641 | 3.5 | 7 | 1,670 | 1,746 | P0 | none, power cap |
| 0 | 2026-11-25 18:43:39.642 | 2026-11-25 18:43:42.141 | 2.5 | 5 | 1,771 | 1,784 | P0 | none, power cap |
| 3 | 2026-11-25 18:43:42.141 | 2026-11-25 18:43:44.141 | 2.0 | 4 | 1,733 | 1,797 | P0 | none, power cap |
| 1 | 2026-11-25 18:44:18.141 | 2026-11-25 18:44:21.141 | 3.0 | 6 | 1,695 | 1,714 | P0 | none, power cap |
| 3 | 2026-11-25 18:44:18.641 | 2026-11-25 18:44:22.641 | 4.0 | 8 | 1,733 | 1,771 | P0 | none, power cap |
| 4 | 2026-11-25 18:44:21.141 | 2026-11-25 18:44:23.142 | 2.0 | 4 | 1,733 | 1,771 | P0 | none, power cap |
| 5 | 2026-11-25 18:44:22.141 | 2026-11-25 18:44:24.141 | 2.0 | 4 | 1,771 | 1,784 | P0 | none, power cap |
| 0 | 2026-11-25 18:44:22.641 | 2026-11-25 18:44:24.641 | 2.0 | 4 | 1,759 | 1,771 | P0 | none, power cap |
| 2 | 2026-11-25 18:44:24.141 | 2026-11-25 18:44:26.141 | 2.0 | 4 | 1,746 | 1,759 | P0 | none |

## Log events

- 2026-11-25 18:09:01.630 card_log start: pid 19390, driver 580.178.04, 9 cards (NVIDIA P106-090), interval 0.5 s; g0 limit 75 W, memory offset 900, core offset 0; g1 limit 75 W, memory offset 1000, core offset 0; g2 limit 75 W, memory offset 1100, core offset 0; g3 limit 75 W, memory offset 1200, core offset 0; g4 limit 75 W, memory offset 1100, core offset 0; g5 limit 75 W, memory offset 1300, core offset 0; g6 limit 75 W, memory offset 0, core offset 0; g7 limit 75 W, memory offset 1100, core offset 0; g8 limit 75 W, memory offset 900, core offset 0
- 2026-11-25 18:45:10.643 card_log stop (signal): 4338 samples, 167.13 s CPU (user+sys), sampling 41.0 ms median, 302.4 ms max, 0 samples late by more than one interval; g0 limit 75 W, memory offset 900, core offset 0; g1 limit 75 W, memory offset 1000, core offset 0; g2 limit 75 W, memory offset 1100, core offset 0; g3 limit 75 W, memory offset 1200, core offset 0; g4 limit 75 W, memory offset 1100, core offset 0; g5 limit 75 W, memory offset 1300, core offset 0; g6 limit 75 W, memory offset 0, core offset 0; g7 limit 75 W, memory offset 1100, core offset 0; g8 limit 75 W, memory offset 900, core offset 0
