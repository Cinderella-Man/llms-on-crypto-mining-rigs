# Cards: cards.csv

24,127 samples from 2026-11-19 14:40:22.612 to 2026-11-19 18:01:25.612 (12,063.0 s); 9 cards; a sample every 0.500 s (median), longest gap 0.61 s, 0 gaps over 2.0 s (0.0 s unlogged).
Busy: utilization above 0 % (a kernel ran in NVML's last sample period). Low: an SM clock below 1,800 MHz.

| card | samples | P0 s | P8 s | busy s | SM MHz busy, median | lowest | busy below 1,800 MHz | memory MHz busy, median | peak W | peak °C | most MiB | reasons, all samples | reasons, busy samples |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0 | 24,127 | 4,615.0 | 7,448.5 | 2,296.0 | 2,113 | 1,354 | 130 of 4,592 (2.83 %) | 4,657 | 81.64 | 40 | 5,808 | idle 15,036 · none 8,880 · power cap 211 | none 4,404 · power cap 188 |
| 1 | 24,127 | 4,613.0 | 7,450.5 | 2,295.0 | 2,151 | 1,354 | 165 of 4,590 (3.59 %) | 4,498 | 91.92 | 37 | 5,828 | idle 15,043 · none 8,858 · power cap 226 | none 4,388 · power cap 202 |
| 2 | 24,127 | 4,613.5 | 7,450.0 | 2,298.0 | 2,151 | 1,354 | 156 of 4,596 (3.39 %) | 4,498 | 88.03 | 33 | 5,754 | idle 15,060 · none 8,879 · power cap 188 | none 4,424 · power cap 172 |
| 3 | 24,127 | 4,620.5 | 7,443.0 | 2,299.5 | 2,100 | 1,354 | 181 of 4,599 (3.94 %) | 4,551 | 89.40 | 32 | 5,828 | idle 15,056 · none 8,886 · power cap 185 | none 4,426 · power cap 173 |
| 4 | 24,127 | 4,663.0 | 7,400.5 | 2,277.5 | 2,113 | 1,354 | 135 of 4,555 (2.96 %) | 4,551 | 80.44 | 37 | 6,136 | idle 14,967 · none 8,968 · power cap 192 | none 4,386 · power cap 169 |
| 5 | 24,127 | 4,599.5 | 7,464.0 | 2,181.5 | 2,100 | 1,354 | 134 of 4,363 (3.07 %) | 4,608 | 81.02 | 33 | 5,544 | idle 15,102 · none 8,914 · power cap 111 | none 4,267 · power cap 96 |
| 6 | 24,127 | 4,624.0 | 7,439.5 | 2,170.5 | 2,164 | 1,354 | 132 of 4,341 (3.04 %) | 4,455 | 81.08 | 41 | 5,706 | idle 15,046 · none 9,008 · power cap 73 | none 4,281 · power cap 60 |
| 7 | 24,127 | 4,612.0 | 7,451.5 | 2,159.5 | 2,151 | 1,354 | 141 of 4,319 (3.26 %) | 4,455 | 81.00 | 36 | 5,782 | idle 15,102 · none 8,876 · power cap 149 | none 4,197 · power cap 122 |
| 8 | 24,127 | 4,319.5 | 7,744.0 | 2,116.0 | 2,126 | 1,354 | 113 of 4,232 (2.67 %) | 4,657 | 84.11 | 36 | 5,188 | idle 15,577 · none 8,392 · power cap 158 | none 4,094 · power cap 138 |

## Busy stretches below 1,800 MHz, 2.0 s or longer

None: no card was busy below 1,800 MHz for 2.0 s in a row.

## Log events

- 2026-11-19 14:40:22.604 card_log start: pid 4540, driver 580.178.04, 9 cards (NVIDIA P106-090), interval 0.5 s; g0 limit 75 W, memory offset 1300, core offset 256; g1 limit 75 W, memory offset 1000, core offset 256; g2 limit 75 W, memory offset 1000, core offset 269; g3 limit 75 W, memory offset 1100, core offset 231; g4 limit 75 W, memory offset 1100, core offset 206; g5 limit 75 W, memory offset 1200, core offset 206; g6 limit 75 W, memory offset 900, core offset 282; g7 limit 75 W, memory offset 900, core offset 256; g8 limit 75 W, memory offset 1300, core offset 231
- 2026-11-19 18:01:25.968 card_log stop (signal): 24127 samples, 841.24 s CPU (user+sys), sampling 36.3 ms median, 272.3 ms max, 0 samples late by more than one interval; g0 limit 75 W, memory offset 1300, core offset 256; g1 limit 75 W, memory offset 1000, core offset 256; g2 limit 75 W, memory offset 1000, core offset 269; g3 limit 75 W, memory offset 1100, core offset 231; g4 limit 75 W, memory offset 1100, core offset 206; g5 limit 75 W, memory offset 1200, core offset 206; g6 limit 75 W, memory offset 900, core offset 282; g7 limit 75 W, memory offset 900, core offset 256; g8 limit 75 W, memory offset 1300, core offset 231
