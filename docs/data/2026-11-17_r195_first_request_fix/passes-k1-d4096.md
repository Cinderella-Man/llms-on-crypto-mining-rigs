# How every graph evaluation ran, pass by pass (<RES>/gp-k1-d4096.csv)

| piece | rows | evaluations on card 0 | host_ms span | request |
|---|---|---|---|---|
| 0 | 1019 | 79 | 70778.7 to 81415.3 | d4096p1 |
| 1 | 999 | 77 | 85463.8 to 94113.9 | d4096p2 |

## Evaluations by how

| request | main passes | main: eager | capture | replay | other | draft evaluations | draft: eager | capture | replay | other |
|---|---|---|---|---|---|---|---|---|---|---|
| d4096p1 | 79 | 27 | 18 | 666 | 0 | 308 | 4 | 4 | 300 | 0 |
| d4096p2 | 77 | 18 | 9 | 666 | 0 | 306 | 0 | 0 | 306 | 0 |

## Main passes, d4096p1 against d4096p2

How on cards 0 to 8 (E eager, C capture, R replay, - not written); GPU ms summed over the cards written.

| pass | d4096p1 how | d4096p1 GPU ms | d4096p2 how | d4096p2 GPU ms | difference ms |
|---|---|---|---|---|---|
| 0 (prompt) | CCCCCCCCC | 1637.05 | EEEEEEEEE | 97.97 | +1539.08 |
| 1 (first verify) | EEEEEEEEE | 240.17 | EEEEEEEEE | 101.64 | +138.53 |
| 2 | EEEEEEEEE | 114.64 | CCCCCCCCC | 93.64 | +21.00 |
| 3 | CCCCCCCCC | 99.40 | RRRRRRRRR | 95.39 | +4.00 |
| 4 | RRRRRRRRR | 95.28 | RRRRRRRRR | 95.33 | -0.05 |
| 5 | RRRRRRRRR | 95.48 | RRRRRRRRR | 93.92 | +1.56 |
| 6 | RRRRRRRRR | 93.93 | RRRRRRRRR | 93.63 | +0.31 |
| 7 | RRRRRRRRR | 93.65 | RRRRRRRRR | 94.46 | -0.81 |
| 8 | RRRRRRRRR | 94.49 | RRRRRRRRR | 94.45 | +0.04 |
| 9 | RRRRRRRRR | 94.61 | RRRRRRRRR | 94.56 | +0.05 |
| 10 | RRRRRRRRR | 94.54 | RRRRRRRRR | 94.82 | -0.28 |
| 11 | RRRRRRRRR | 95.03 | RRRRRRRRR | 92.66 | +2.36 |
| 12 | RRRRRRRRR | 92.68 | RRRRRRRRR | 92.80 | -0.13 |
| 13 | RRRRRRRRR | 92.87 | RRRRRRRRR | 94.35 | -1.47 |
| 14 | RRRRRRRRR | 94.33 | RRRRRRRRR | 93.79 | +0.55 |
| 15 | RRRRRRRRR | 93.83 | RRRRRRRRR | 94.20 | -0.37 |
| 16 | RRRRRRRRR | 94.36 | RRRRRRRRR | 94.62 | -0.25 |
| 17 | RRRRRRRRR | 94.48 | RRRRRRRRR | 94.04 | +0.45 |
| 18 | RRRRRRRRR | 93.94 | RRRRRRRRR | 94.83 | -0.89 |
| 19 | RRRRRRRRR | 94.91 | RRRRRRRRR | 94.64 | +0.27 |
| 20 | RRRRRRRRR | 94.60 | RRRRRRRRR | 94.37 | +0.24 |
| 21 | RRRRRRRRR | 94.60 | RRRRRRRRR | 94.21 | +0.39 |
| 22 | RRRRRRRRR | 94.12 | RRRRRRRRR | 93.66 | +0.46 |
| 23 | RRRRRRRRR | 93.68 | RRRRRRRRR | 94.45 | -0.77 |
| 24 | RRRRRRRRR | 94.35 | RRRRRRRRR | 94.04 | +0.31 |
| 25 | RRRRRRRRR | 94.13 | RRRRRRRRR | 94.14 | -0.01 |
| 26 | RRRRRRRRR | 94.11 | RRRRRRRRR | 91.60 | +2.51 |
| 27 | RRRRRRRRR | 91.82 | RRRRRRRRR | 93.96 | -2.14 |
| 28 | RRRRRRRRR | 93.96 | RRRRRRRRR | 94.69 | -0.73 |
| 29 | RRRRRRRRR | 94.96 | RRRRRRRRR | 92.09 | +2.87 |
| 30 | RRRRRRRRR | 91.94 | RRRRRRRRR | 92.33 | -0.38 |
| 31 | RRRRRRRRR | 92.37 | RRRRRRRRR | 92.58 | -0.21 |
| 32 | RRRRRRRRR | 92.66 | RRRRRRRRR | 93.02 | -0.36 |
| 33 | RRRRRRRRR | 93.25 | RRRRRRRRR | 94.36 | -1.11 |
| 34 | RRRRRRRRR | 94.11 | RRRRRRRRR | 93.19 | +0.92 |
| 35 | RRRRRRRRR | 93.39 | RRRRRRRRR | 95.25 | -1.86 |
| 36 | RRRRRRRRR | 95.05 | RRRRRRRRR | 94.28 | +0.77 |
| 37 | RRRRRRRRR | 94.43 | RRRRRRRRR | 93.00 | +1.44 |
| 38 | RRRRRRRRR | 92.84 | RRRRRRRRR | 93.51 | -0.67 |
| 39 | RRRRRRRRR | 93.38 | RRRRRRRRR | 93.32 | +0.06 |
| 40 | RRRRRRRRR | 93.35 | RRRRRRRRR | 93.41 | -0.06 |
| 41 | RRRRRRRRR | 93.51 | RRRRRRRRR | 95.52 | -2.01 |
| 42 | RRRRRRRRR | 95.70 | RRRRRRRRR | 95.45 | +0.24 |
| 43 | RRRRRRRRR | 95.50 | RRRRRRRRR | 95.74 | -0.24 |
| 44 | RRRRRRRRR | 95.71 | RRRRRRRRR | 95.49 | +0.22 |
| 45 | RRRRRRRRR | 95.50 | RRRRRRRRR | 93.58 | +1.91 |
| 46 | RRRRRRRRR | 93.45 | RRRRRRRRR | 95.24 | -1.78 |
| 47 | RRRRRRRRR | 95.16 | RRRRRRRRR | 95.58 | -0.42 |
| 48 | RRRRRRRRR | 95.43 | RRRRRRRRR | 95.36 | +0.07 |
| 49 | RRRRRRRRR | 95.51 | RRRRRRRRR | 95.77 | -0.26 |
| 50 | RRRRRRRRR | 95.99 | RRRRRRRRR | 94.61 | +1.39 |
| 51 | RRRRRRRRR | 94.77 | RRRRRRRRR | 95.75 | -0.97 |
| 52 | RRRRRRRRR | 95.92 | RRRRRRRRR | 96.43 | -0.51 |
| 53 | RRRRRRRRR | 96.54 | RRRRRRRRR | 91.85 | +4.69 |
| 54 | RRRRRRRRR | 91.81 | RRRRRRRRR | 92.92 | -1.10 |
| 55 | RRRRRRRRR | 93.04 | RRRRRRRRR | 95.94 | -2.90 |
| 56 | RRRRRRRRR | 95.99 | RRRRRRRRR | 95.57 | +0.42 |
| 57 | RRRRRRRRR | 95.77 | RRRRRRRRR | 93.30 | +2.47 |
| 58 | RRRRRRRRR | 93.18 | RRRRRRRRR | 96.11 | -2.93 |
| 59 | RRRRRRRRR | 96.22 | RRRRRRRRR | 95.56 | +0.67 |
| 60 | RRRRRRRRR | 95.56 | RRRRRRRRR | 95.25 | +0.31 |
| 61 | RRRRRRRRR | 95.42 | RRRRRRRRR | 93.98 | +1.44 |
| 62 | RRRRRRRRR | 94.13 | RRRRRRRRR | 94.43 | -0.30 |
| 63 | RRRRRRRRR | 94.47 | RRRRRRRRR | 93.35 | +1.12 |
| 64 | RRRRRRRRR | 93.26 | RRRRRRRRR | 94.09 | -0.83 |
| 65 | RRRRRRRRR | 94.15 | RRRRRRRRR | 93.81 | +0.35 |
| 66 | RRRRRRRRR | 93.76 | RRRRRRRRR | 92.73 | +1.03 |
| 67 | RRRRRRRRR | 92.81 | RRRRRRRRR | 93.31 | -0.50 |
| 68 | RRRRRRRRR | 93.32 | RRRRRRRRR | 94.69 | -1.37 |
| 69 | RRRRRRRRR | 94.69 | RRRRRRRRR | 95.56 | -0.87 |
| 70 | RRRRRRRRR | 95.58 | RRRRRRRRR | 92.89 | +2.68 |
| 71 | RRRRRRRRR | 92.89 | RRRRRRRRR | 93.93 | -1.04 |
| 72 | RRRRRRRRR | 93.91 | RRRRRRRRR | 94.40 | -0.49 |
| 73 | RRRRRRRRR | 94.55 | RRRRRRRRR | 94.51 | +0.03 |
| 74 | RRRRRRRRR | 94.64 | RRRRRRRRR | 94.16 | +0.48 |
| 75 | RRRRRRRRR | 94.16 | RRRRRRRRR | 95.26 | -1.10 |
| 76 (d4096p2 last) | RRRRRRRRR | 95.10 | RRRRRRRRR | 95.84 | -0.74 |
| 77 | RRRRRRRRR | 95.84 |  | nan | +nan |
| 78 (d4096p1 last) | EEEEEEEEE | 187.51 |  | nan | +nan |

## The first verify pass and the last pass, card by card

GPU ms of each card's evaluation, how in parentheses.

| pass | request | card 0 | card 1 | card 2 | card 3 | card 4 | card 5 | card 6 | card 7 | card 8 |
|---|---|---|---|---|---|---|---|---|---|---|
| first verify (1) | d4096p1 | 27.75 (E) | 32.44 (E) | 28.71 (E) | 25.88 (E) | 21.03 (E) | 30.94 (E) | 28.24 (E) | 30.89 (E) | 14.30 (E) |
| first verify (1) | d4096p2 | 12.54 (E) | 12.80 (E) | 11.59 (E) | 12.73 (E) | 9.75 (E) | 11.31 (E) | 10.70 (E) | 11.09 (E) | 9.13 (E) |
| last (78) | d4096p1 | 21.39 (E) | 24.39 (E) | 21.36 (E) | 22.41 (E) | 17.76 (E) | 23.48 (E) | 21.16 (E) | 21.84 (E) | 13.72 (E) |
| last (76) | d4096p2 | 11.32 (R) | 11.84 (R) | 11.02 (R) | 11.97 (R) | 9.59 (R) | 10.88 (R) | 10.15 (R) | 10.80 (R) | 8.27 (R) |
| second verify (2) | d4096p1 | 12.59 (E) | 12.79 (E) | 11.60 (E) | 12.68 (E) | 9.79 (E) | 12.35 (E) | 11.98 (E) | 13.40 (E) | 17.47 (E) |
| second verify (2) | d4096p2 | 12.52 (C) | 11.56 (C) | 10.63 (C) | 11.51 (C) | 8.97 (C) | 10.30 (C) | 9.81 (C) | 10.09 (C) | 8.24 (C) |

## Passes not replayed on every card

| request | pass | how | GPU ms | cards eager | cards captured |
|---|---|---|---|---|---|
| d4096p1 | 0 | CCCCCCCCC | 1637.05 | 0 | 9 |
| d4096p1 | 1 | EEEEEEEEE | 240.17 | 9 | 0 |
| d4096p1 | 2 | EEEEEEEEE | 114.64 | 9 | 0 |
| d4096p1 | 3 | CCCCCCCCC | 99.40 | 0 | 9 |
| d4096p1 | 78 (last) | EEEEEEEEE | 187.51 | 9 | 0 |
| d4096p2 | 0 | EEEEEEEEE | 97.97 | 9 | 0 |
| d4096p2 | 1 | EEEEEEEEE | 101.64 | 9 | 0 |
| d4096p2 | 2 | CCCCCCCCC | 93.64 | 0 | 9 |

## The draft's evaluations that ran eagerly or were captured

| request | draft evaluation (order) | how | graph | GPU ms | after main pass |
|---|---|---|---|---|---|
| d4096p1 | 0 | capture | 4 | 10.06 | 1 |
| d4096p1 | 2 | eager | 6 | 7.26 | 1 |
| d4096p1 | 3 | capture | 6 | 4.17 | 1 |
| d4096p1 | 34 | eager | 7 | 9.50 | 9 |
| d4096p1 | 35 | capture | 7 | 7.81 | 9 |
| d4096p1 | 42 | eager | 6 | 5.96 | 11 |
| d4096p1 | 43 | capture | 6 | 3.44 | 11 |
| d4096p1 | 307 | eager | 11 | 0.91 | 78 |
