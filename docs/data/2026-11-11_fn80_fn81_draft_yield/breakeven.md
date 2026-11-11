step3 128.92 ms, drafting pass 3.645 ms (est.)

## Each draft length against d3, pooled (est. costs)

| arm | tokens a step | ratio to d3 | 95 % over requests | break-even column, ms | its 95 % range | tok/s at 8 ms | tok/s at 10 ms | tok/s at 12 ms | tok/s at 15.36 ms | tok/s at 21.36 ms |
|---|---:|---:|---|---:|---|---:|---:|---:|---:|---:|
| d3 | 3.2401 | 1 | | | | 25.13 | 25.13 | 25.13 | 25.13 | 25.13 |
| d4 | 3.7028 | 1.1428 | 1.1130 to 1.1710 | 14.77 | 10.92 to 18.39 | 26.34 | 25.97 | 25.61 | 25.03 | 24.06 |
| d5 | 4.1257 | 1.2733 | 1.2155 to 1.3212 | 13.97 | 10.25 to 17.06 | 27.11 | 26.41 | 25.75 | 24.72 | 23.06 |
| d6 | 4.5037 | 1.3900 | 1.3398 to 1.4364 | 13.11 | 10.96 to 15.11 | 27.49 | 26.51 | 25.61 | 24.22 | 22.08 |
| d7 | 4.5024 | 1.3896 | 1.3452 to 1.4368 | 8.91 | 7.48 to 10.43 | 25.65 | 24.54 | 23.51 | 21.97 | 19.67 |

## The same text: each load's verify steps cut to a shorter draft (accepted + 1 per verify step)

Cumulative: the length L against 3 on the same steps; marginal: L against L - 1, with every column before it at the same cost.

| load | L | tokens a verify step | ratio to 3 | break-even column against 3, ms | added by L | break-even column for L against L - 1, ms |
|---|---:|---:|---:|---:|---:|---:|
| d3-1 | 3 | 3.2697 | 1 | | | |
| d4-1 | 3 | 3.2182 | 1 | | | |
| d4-1 | 4 | 3.7456 | 1.1639 | 17.48 | +0.5274 | 17.48 |
| d5-1 | 3 | 3.2017 | 1 | | | |
| d5-1 | 4 | 3.7245 | 1.1633 | 17.41 | +0.5228 | 17.41 |
| d5-1 | 5 | 4.1709 | 1.3027 | 15.87 | +0.4464 | 13.91 |
| d6-1 | 3 | 3.2137 | 1 | | | |
| d6-1 | 4 | 3.7345 | 1.1621 | 17.25 | +0.5208 | 17.25 |
| d6-1 | 5 | 4.1843 | 1.3020 | 15.82 | +0.4498 | 14.01 |
| d6-1 | 6 | 4.5711 | 1.4224 | 14.51 | +0.3868 | 10.98 |
| d7-1 | 3 | 3.1059 | 1 | | | |
| d7-1 | 4 | 3.5799 | 1.1526 | 16.03 | +0.4740 | 16.03 |
| d7-1 | 5 | 3.9713 | 1.2786 | 14.32 | +0.3914 | 12.18 |
| d7-1 | 6 | 4.2993 | 1.3842 | 12.87 | +0.3280 | 9.11 |
| d7-1 | 7 | 4.5729 | 1.4723 | 11.58 | +0.2736 | 6.50 |
