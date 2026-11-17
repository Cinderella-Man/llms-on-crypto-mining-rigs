# How every graph evaluation ran, pass by pass (<RES>/gp-e1-d4096.csv)

| piece | rows | evaluations on card 0 | host_ms span | request |
|---|---|---|---|---|
| 0 | 1019 | 79 | 71452.1 to 82196.0 | d4096p1 |
| 1 | 999 | 77 | 86234.3 to 94880.2 | d4096p2 |

## Evaluations by how

| request | main passes | main: eager | capture | replay | other | draft evaluations | draft: eager | capture | replay | other |
|---|---|---|---|---|---|---|---|---|---|---|
| d4096p1 | 79 | 27 | 18 | 666 | 0 | 308 | 4 | 4 | 300 | 0 |
| d4096p2 | 77 | 18 | 9 | 666 | 0 | 306 | 0 | 0 | 306 | 0 |

## Main passes, d4096p1 against d4096p2

How on cards 0 to 8 (E eager, C capture, R replay, - not written); GPU ms summed over the cards written.

| pass | d4096p1 how | d4096p1 GPU ms | d4096p2 how | d4096p2 GPU ms | difference ms |
|---|---|---|---|---|---|
| 0 (prompt) | CCCCCCCCC | 1962.81 | EEEEEEEEE | 97.75 | +1865.06 |
| 1 (first verify) | EEEEEEEEE | 237.57 | EEEEEEEEE | 101.53 | +136.03 |
| 2 | EEEEEEEEE | 221.89 | CCCCCCCCC | 93.65 | +128.24 |
| 3 | CCCCCCCCC | 99.00 | RRRRRRRRR | 95.30 | +3.71 |
| 4 | RRRRRRRRR | 95.36 | RRRRRRRRR | 95.32 | +0.04 |
| 5 | RRRRRRRRR | 95.42 | RRRRRRRRR | 93.82 | +1.59 |
| 6 | RRRRRRRRR | 93.69 | RRRRRRRRR | 93.68 | +0.01 |
| 7 | RRRRRRRRR | 93.62 | RRRRRRRRR | 94.52 | -0.90 |
| 8 | RRRRRRRRR | 94.34 | RRRRRRRRR | 94.46 | -0.12 |
| 9 | RRRRRRRRR | 94.65 | RRRRRRRRR | 94.59 | +0.06 |
| 10 | RRRRRRRRR | 94.47 | RRRRRRRRR | 94.83 | -0.36 |
| 11 | RRRRRRRRR | 94.83 | RRRRRRRRR | 92.64 | +2.19 |
| 12 | RRRRRRRRR | 92.53 | RRRRRRRRR | 92.78 | -0.25 |
| 13 | RRRRRRRRR | 92.83 | RRRRRRRRR | 94.52 | -1.69 |
| 14 | RRRRRRRRR | 94.23 | RRRRRRRRR | 93.86 | +0.37 |
| 15 | RRRRRRRRR | 93.81 | RRRRRRRRR | 94.27 | -0.46 |
| 16 | RRRRRRRRR | 94.28 | RRRRRRRRR | 94.39 | -0.11 |
| 17 | RRRRRRRRR | 94.26 | RRRRRRRRR | 94.03 | +0.23 |
| 18 | RRRRRRRRR | 93.88 | RRRRRRRRR | 94.99 | -1.11 |
| 19 | RRRRRRRRR | 94.80 | RRRRRRRRR | 94.54 | +0.26 |
| 20 | RRRRRRRRR | 94.37 | RRRRRRRRR | 94.66 | -0.29 |
| 21 | RRRRRRRRR | 94.48 | RRRRRRRRR | 94.24 | +0.24 |
| 22 | RRRRRRRRR | 94.05 | RRRRRRRRR | 93.86 | +0.19 |
| 23 | RRRRRRRRR | 93.47 | RRRRRRRRR | 94.40 | -0.93 |
| 24 | RRRRRRRRR | 94.31 | RRRRRRRRR | 94.24 | +0.07 |
| 25 | RRRRRRRRR | 94.12 | RRRRRRRRR | 94.23 | -0.11 |
| 26 | RRRRRRRRR | 93.85 | RRRRRRRRR | 91.58 | +2.27 |
| 27 | RRRRRRRRR | 91.62 | RRRRRRRRR | 93.93 | -2.31 |
| 28 | RRRRRRRRR | 93.92 | RRRRRRRRR | 95.01 | -1.09 |
| 29 | RRRRRRRRR | 94.70 | RRRRRRRRR | 92.09 | +2.62 |
| 30 | RRRRRRRRR | 91.89 | RRRRRRRRR | 92.17 | -0.28 |
| 31 | RRRRRRRRR | 92.10 | RRRRRRRRR | 92.59 | -0.49 |
| 32 | RRRRRRRRR | 92.39 | RRRRRRRRR | 93.19 | -0.81 |
| 33 | RRRRRRRRR | 93.06 | RRRRRRRRR | 94.23 | -1.17 |
| 34 | RRRRRRRRR | 94.05 | RRRRRRRRR | 93.19 | +0.87 |
| 35 | RRRRRRRRR | 93.13 | RRRRRRRRR | 95.22 | -2.09 |
| 36 | RRRRRRRRR | 94.99 | RRRRRRRRR | 94.48 | +0.51 |
| 37 | RRRRRRRRR | 94.32 | RRRRRRRRR | 93.09 | +1.22 |
| 38 | RRRRRRRRR | 92.78 | RRRRRRRRR | 93.46 | -0.67 |
| 39 | RRRRRRRRR | 93.51 | RRRRRRRRR | 93.31 | +0.20 |
| 40 | RRRRRRRRR | 93.35 | RRRRRRRRR | 93.41 | -0.06 |
| 41 | RRRRRRRRR | 93.37 | RRRRRRRRR | 95.77 | -2.40 |
| 42 | RRRRRRRRR | 95.66 | RRRRRRRRR | 95.44 | +0.22 |
| 43 | RRRRRRRRR | 95.35 | RRRRRRRRR | 95.61 | -0.26 |
| 44 | RRRRRRRRR | 95.61 | RRRRRRRRR | 95.32 | +0.29 |
| 45 | RRRRRRRRR | 95.25 | RRRRRRRRR | 93.48 | +1.77 |
| 46 | RRRRRRRRR | 93.44 | RRRRRRRRR | 95.18 | -1.75 |
| 47 | RRRRRRRRR | 95.03 | RRRRRRRRR | 95.53 | -0.50 |
| 48 | RRRRRRRRR | 95.36 | RRRRRRRRR | 95.46 | -0.10 |
| 49 | RRRRRRRRR | 95.41 | RRRRRRRRR | 95.92 | -0.51 |
| 50 | RRRRRRRRR | 95.75 | RRRRRRRRR | 94.77 | +0.99 |
| 51 | RRRRRRRRR | 94.63 | RRRRRRRRR | 95.88 | -1.25 |
| 52 | RRRRRRRRR | 95.67 | RRRRRRRRR | 96.39 | -0.73 |
| 53 | RRRRRRRRR | 96.30 | RRRRRRRRR | 91.87 | +4.43 |
| 54 | RRRRRRRRR | 91.73 | RRRRRRRRR | 92.96 | -1.23 |
| 55 | RRRRRRRRR | 92.79 | RRRRRRRRR | 96.03 | -3.24 |
| 56 | RRRRRRRRR | 95.93 | RRRRRRRRR | 95.59 | +0.34 |
| 57 | RRRRRRRRR | 95.67 | RRRRRRRRR | 93.33 | +2.35 |
| 58 | RRRRRRRRR | 93.17 | RRRRRRRRR | 96.17 | -3.00 |
| 59 | RRRRRRRRR | 96.14 | RRRRRRRRR | 95.54 | +0.60 |
| 60 | RRRRRRRRR | 95.46 | RRRRRRRRR | 95.36 | +0.10 |
| 61 | RRRRRRRRR | 95.31 | RRRRRRRRR | 94.22 | +1.09 |
| 62 | RRRRRRRRR | 94.03 | RRRRRRRRR | 94.54 | -0.51 |
| 63 | RRRRRRRRR | 94.49 | RRRRRRRRR | 93.46 | +1.03 |
| 64 | RRRRRRRRR | 93.27 | RRRRRRRRR | 94.26 | -0.99 |
| 65 | RRRRRRRRR | 94.02 | RRRRRRRRR | 93.69 | +0.34 |
| 66 | RRRRRRRRR | 93.61 | RRRRRRRRR | 92.68 | +0.93 |
| 67 | RRRRRRRRR | 92.62 | RRRRRRRRR | 93.35 | -0.73 |
| 68 | RRRRRRRRR | 93.14 | RRRRRRRRR | 94.65 | -1.51 |
| 69 | RRRRRRRRR | 94.62 | RRRRRRRRR | 95.85 | -1.23 |
| 70 | RRRRRRRRR | 95.52 | RRRRRRRRR | 93.05 | +2.47 |
| 71 | RRRRRRRRR | 92.89 | RRRRRRRRR | 93.73 | -0.84 |
| 72 | RRRRRRRRR | 93.86 | RRRRRRRRR | 94.52 | -0.66 |
| 73 | RRRRRRRRR | 94.51 | RRRRRRRRR | 94.51 | +0.00 |
| 74 | RRRRRRRRR | 94.48 | RRRRRRRRR | 94.14 | +0.34 |
| 75 | RRRRRRRRR | 94.28 | RRRRRRRRR | 95.28 | -1.00 |
| 76 (d4096p2 last) | RRRRRRRRR | 95.19 | RRRRRRRRR | 95.82 | -0.63 |
| 77 | RRRRRRRRR | 95.85 |  | nan | +nan |
| 78 (d4096p1 last) | EEEEEEEEE | 186.94 |  | nan | +nan |

## The first verify pass and the last pass, card by card

GPU ms of each card's evaluation, how in parentheses.

| pass | request | card 0 | card 1 | card 2 | card 3 | card 4 | card 5 | card 6 | card 7 | card 8 |
|---|---|---|---|---|---|---|---|---|---|---|
| first verify (1) | d4096p1 | 26.75 (E) | 31.45 (E) | 28.79 (E) | 25.77 (E) | 21.01 (E) | 30.15 (E) | 28.69 (E) | 30.77 (E) | 14.20 (E) |
| first verify (1) | d4096p2 | 12.44 (E) | 12.76 (E) | 11.66 (E) | 12.67 (E) | 9.77 (E) | 11.33 (E) | 10.68 (E) | 11.13 (E) | 9.09 (E) |
| last (78) | d4096p1 | 21.45 (E) | 24.23 (E) | 21.77 (E) | 22.63 (E) | 17.62 (E) | 23.33 (E) | 20.43 (E) | 21.84 (E) | 13.64 (E) |
| last (76) | d4096p2 | 11.40 (R) | 11.77 (R) | 11.00 (R) | 12.03 (R) | 9.62 (R) | 10.87 (R) | 10.10 (R) | 10.80 (R) | 8.24 (R) |
| second verify (2) | d4096p1 | 26.50 (E) | 28.59 (E) | 25.26 (E) | 27.09 (E) | 20.95 (E) | 26.95 (E) | 23.90 (E) | 24.92 (E) | 17.72 (E) |
| second verify (2) | d4096p2 | 12.57 (C) | 11.60 (C) | 10.62 (C) | 11.54 (C) | 8.98 (C) | 10.26 (C) | 9.84 (C) | 10.07 (C) | 8.19 (C) |

## Passes not replayed on every card

| request | pass | how | GPU ms | cards eager | cards captured |
|---|---|---|---|---|---|
| d4096p1 | 0 | CCCCCCCCC | 1962.81 | 0 | 9 |
| d4096p1 | 1 | EEEEEEEEE | 237.57 | 9 | 0 |
| d4096p1 | 2 | EEEEEEEEE | 221.89 | 9 | 0 |
| d4096p1 | 3 | CCCCCCCCC | 99.00 | 0 | 9 |
| d4096p1 | 78 (last) | EEEEEEEEE | 186.94 | 9 | 0 |
| d4096p2 | 0 | EEEEEEEEE | 97.75 | 9 | 0 |
| d4096p2 | 1 | EEEEEEEEE | 101.53 | 9 | 0 |
| d4096p2 | 2 | CCCCCCCCC | 93.65 | 0 | 9 |

## The draft's evaluations that ran eagerly or were captured

| request | draft evaluation (order) | how | graph | GPU ms | after main pass |
|---|---|---|---|---|---|
| d4096p1 | 0 | capture | 4 | 10.01 | 1 |
| d4096p1 | 2 | eager | 6 | 7.22 | 1 |
| d4096p1 | 3 | capture | 6 | 4.23 | 1 |
| d4096p1 | 34 | eager | 7 | 9.50 | 9 |
| d4096p1 | 35 | capture | 7 | 7.78 | 9 |
| d4096p1 | 42 | eager | 6 | 5.93 | 11 |
| d4096p1 | 43 | capture | 6 | 3.44 | 11 |
| d4096p1 | 307 | eager | 11 | 0.93 | 78 |
