# How every graph evaluation ran, pass by pass (gp-g1.csv)

| piece | rows | evaluations on card 0 | host_ms span | request |
|---|---|---|---|---|
| 0 | 215 | 17 | 23376.0 to 28135.8 | - |
| 1 | 90 | 9 | 32732.3 to 61665.1 | - |
| 2 | 1008 | 78 | 67045.7 to 76245.8 | d4096p1 |
| 3 | 988 | 76 | 80286.0 to 88866.1 | d4096p2 |

## Evaluations by how

| request | main passes | main: eager | capture | replay | other | draft evaluations | draft: eager | capture | replay | other |
|---|---|---|---|---|---|---|---|---|---|---|
| d4096p1 | 78 | 27 | 9 | 666 | 0 | 306 | 4 | 3 | 299 | 0 |
| d4096p2 | 76 | 18 | 9 | 657 | 0 | 304 | 0 | 0 | 304 | 0 |

## Main passes, d4096p1 against d4096p2

How on cards 0 to 8 (E eager, C capture, R replay, - not written); GPU ms summed over the cards written.

| pass | d4096p1 how | d4096p1 GPU ms | d4096p2 how | d4096p2 GPU ms | difference ms |
|---|---|---|---|---|---|
| 0 (prompt) | EEEEEEEEE | 235.65 | EEEEEEEEE | 118.65 | +117.00 |
| 1 (first verify) | EEEEEEEEE | 219.92 | EEEEEEEEE | 118.12 | +101.80 |
| 2 | CCCCCCCCC | 102.09 | CCCCCCCCC | 103.57 | -1.48 |
| 3 | RRRRRRRRR | 95.18 | RRRRRRRRR | 100.79 | -5.61 |
| 4 | RRRRRRRRR | 95.39 | RRRRRRRRR | 98.33 | -2.94 |
| 5 | RRRRRRRRR | 93.86 | RRRRRRRRR | 94.42 | -0.56 |
| 6 | RRRRRRRRR | 93.59 | RRRRRRRRR | 93.57 | +0.02 |
| 7 | RRRRRRRRR | 94.32 | RRRRRRRRR | 94.47 | -0.16 |
| 8 | RRRRRRRRR | 94.46 | RRRRRRRRR | 94.35 | +0.11 |
| 9 | RRRRRRRRR | 94.28 | RRRRRRRRR | 94.53 | -0.25 |
| 10 | RRRRRRRRR | 94.73 | RRRRRRRRR | 94.77 | -0.04 |
| 11 | RRRRRRRRR | 92.53 | RRRRRRRRR | 92.41 | +0.11 |
| 12 | RRRRRRRRR | 92.75 | RRRRRRRRR | 92.88 | -0.12 |
| 13 | RRRRRRRRR | 94.16 | RRRRRRRRR | 94.43 | -0.26 |
| 14 | RRRRRRRRR | 93.70 | RRRRRRRRR | 93.73 | -0.03 |
| 15 | RRRRRRRRR | 94.07 | RRRRRRRRR | 94.16 | -0.08 |
| 16 | RRRRRRRRR | 94.45 | RRRRRRRRR | 94.35 | +0.10 |
| 17 | RRRRRRRRR | 93.85 | RRRRRRRRR | 93.84 | +0.01 |
| 18 | RRRRRRRRR | 94.63 | RRRRRRRRR | 94.61 | +0.02 |
| 19 | RRRRRRRRR | 94.50 | RRRRRRRRR | 94.42 | +0.08 |
| 20 | RRRRRRRRR | 94.49 | RRRRRRRRR | 94.48 | +0.00 |
| 21 | RRRRRRRRR | 93.92 | RRRRRRRRR | 93.96 | -0.04 |
| 22 | RRRRRRRRR | 93.59 | RRRRRRRRR | 93.48 | +0.11 |
| 23 | RRRRRRRRR | 94.17 | RRRRRRRRR | 94.27 | -0.10 |
| 24 | RRRRRRRRR | 94.04 | RRRRRRRRR | 93.95 | +0.09 |
| 25 | RRRRRRRRR | 93.82 | RRRRRRRRR | 94.06 | -0.24 |
| 26 | RRRRRRRRR | 91.74 | RRRRRRRRR | 91.52 | +0.22 |
| 27 | RRRRRRRRR | 93.79 | RRRRRRRRR | 93.90 | -0.11 |
| 28 | RRRRRRRRR | 94.68 | RRRRRRRRR | 94.76 | -0.08 |
| 29 | RRRRRRRRR | 91.79 | RRRRRRRRR | 92.04 | -0.24 |
| 30 | RRRRRRRRR | 92.08 | RRRRRRRRR | 92.21 | -0.13 |
| 31 | RRRRRRRRR | 92.34 | RRRRRRRRR | 92.45 | -0.12 |
| 32 | RRRRRRRRR | 93.16 | RRRRRRRRR | 93.18 | -0.02 |
| 33 | RRRRRRRRR | 93.96 | RRRRRRRRR | 94.10 | -0.15 |
| 34 | RRRRRRRRR | 93.21 | RRRRRRRRR | 93.28 | -0.06 |
| 35 | RRRRRRRRR | 95.03 | RRRRRRRRR | 95.00 | +0.03 |
| 36 | RRRRRRRRR | 94.21 | RRRRRRRRR | 94.25 | -0.04 |
| 37 | RRRRRRRRR | 92.61 | RRRRRRRRR | 92.73 | -0.12 |
| 38 | RRRRRRRRR | 93.30 | RRRRRRRRR | 93.36 | -0.06 |
| 39 | RRRRRRRRR | 93.13 | RRRRRRRRR | 93.27 | -0.14 |
| 40 | RRRRRRRRR | 93.43 | RRRRRRRRR | 93.43 | +0.00 |
| 41 | RRRRRRRRR | 95.36 | RRRRRRRRR | 95.45 | -0.09 |
| 42 | RRRRRRRRR | 95.30 | RRRRRRRRR | 95.29 | +0.01 |
| 43 | RRRRRRRRR | 95.48 | RRRRRRRRR | 95.50 | -0.01 |
| 44 | RRRRRRRRR | 95.24 | RRRRRRRRR | 95.32 | -0.08 |
| 45 | RRRRRRRRR | 93.25 | RRRRRRRRR | 93.37 | -0.12 |
| 46 | RRRRRRRRR | 95.00 | RRRRRRRRR | 95.05 | -0.05 |
| 47 | RRRRRRRRR | 95.20 | RRRRRRRRR | 95.53 | -0.33 |
| 48 | RRRRRRRRR | 95.27 | RRRRRRRRR | 95.43 | -0.16 |
| 49 | RRRRRRRRR | 95.65 | RRRRRRRRR | 95.64 | +0.02 |
| 50 | RRRRRRRRR | 94.67 | RRRRRRRRR | 94.43 | +0.24 |
| 51 | RRRRRRRRR | 95.76 | RRRRRRRRR | 95.68 | +0.08 |
| 52 | RRRRRRRRR | 96.34 | RRRRRRRRR | 96.31 | +0.02 |
| 53 | RRRRRRRRR | 91.58 | RRRRRRRRR | 91.62 | -0.05 |
| 54 | RRRRRRRRR | 93.03 | RRRRRRRRR | 92.86 | +0.17 |
| 55 | RRRRRRRRR | 95.59 | RRRRRRRRR | 95.85 | -0.26 |
| 56 | RRRRRRRRR | 95.65 | RRRRRRRRR | 95.45 | +0.19 |
| 57 | RRRRRRRRR | 93.15 | RRRRRRRRR | 93.12 | +0.03 |
| 58 | RRRRRRRRR | 96.07 | RRRRRRRRR | 96.24 | -0.17 |
| 59 | RRRRRRRRR | 95.36 | RRRRRRRRR | 95.44 | -0.08 |
| 60 | RRRRRRRRR | 95.20 | RRRRRRRRR | 95.34 | -0.14 |
| 61 | RRRRRRRRR | 93.95 | RRRRRRRRR | 94.03 | -0.08 |
| 62 | RRRRRRRRR | 94.50 | RRRRRRRRR | 94.35 | +0.15 |
| 63 | RRRRRRRRR | 93.30 | RRRRRRRRR | 93.18 | +0.12 |
| 64 | RRRRRRRRR | 93.95 | RRRRRRRRR | 94.02 | -0.08 |
| 65 | RRRRRRRRR | 93.62 | RRRRRRRRR | 93.50 | +0.12 |
| 66 | RRRRRRRRR | 92.69 | RRRRRRRRR | 92.50 | +0.18 |
| 67 | RRRRRRRRR | 92.98 | RRRRRRRRR | 93.14 | -0.16 |
| 68 | RRRRRRRRR | 94.43 | RRRRRRRRR | 94.57 | -0.13 |
| 69 | RRRRRRRRR | 95.39 | RRRRRRRRR | 95.47 | -0.08 |
| 70 | RRRRRRRRR | 92.86 | RRRRRRRRR | 92.85 | +0.01 |
| 71 | RRRRRRRRR | 93.82 | RRRRRRRRR | 93.85 | -0.03 |
| 72 | RRRRRRRRR | 94.23 | RRRRRRRRR | 94.28 | -0.05 |
| 73 | RRRRRRRRR | 94.38 | RRRRRRRRR | 94.46 | -0.08 |
| 74 | RRRRRRRRR | 94.09 | RRRRRRRRR | 94.13 | -0.03 |
| 75 (d4096p2 last) | RRRRRRRRR | 95.22 | RRRRRRRRR | 95.29 | -0.07 |
| 76 | RRRRRRRRR | 95.63 |  | nan | +nan |
| 77 (d4096p1 last) | EEEEEEEEE | 188.19 |  | nan | +nan |

## The first verify pass and the last pass, card by card

GPU ms of each card's evaluation, how in parentheses.

| pass | request | card 0 | card 1 | card 2 | card 3 | card 4 | card 5 | card 6 | card 7 | card 8 |
|---|---|---|---|---|---|---|---|---|---|---|
| first verify (1) | d4096p1 | 26.15 (E) | 28.26 (E) | 25.89 (E) | 27.25 (E) | 20.66 (E) | 27.08 (E) | 23.51 (E) | 24.02 (E) | 17.11 (E) |
| first verify (1) | d4096p2 | 12.41 (E) | 16.02 (E) | 13.15 (E) | 16.28 (E) | 11.74 (E) | 13.46 (E) | 12.69 (E) | 13.31 (E) | 9.05 (E) |
| last (77) | d4096p1 | 21.67 (E) | 24.82 (E) | 21.45 (E) | 22.70 (E) | 17.63 (E) | 23.71 (E) | 20.88 (E) | 21.88 (E) | 13.43 (E) |
| last (75) | d4096p2 | 11.60 (R) | 11.76 (R) | 11.17 (R) | 11.99 (R) | 9.23 (R) | 10.73 (R) | 10.01 (R) | 10.62 (R) | 8.18 (R) |
| second verify (2) | d4096p1 | 16.86 (C) | 11.90 (C) | 11.24 (C) | 12.17 (C) | 9.31 (C) | 10.83 (C) | 10.40 (C) | 10.69 (C) | 8.68 (C) |
| second verify (2) | d4096p2 | 12.52 (C) | 13.70 (C) | 11.32 (C) | 13.98 (C) | 9.98 (C) | 11.56 (C) | 10.95 (C) | 11.44 (C) | 8.12 (C) |

## Passes not replayed on every card

| request | pass | how | GPU ms | cards eager | cards captured |
|---|---|---|---|---|---|
| d4096p1 | 0 | EEEEEEEEE | 235.65 | 9 | 0 |
| d4096p1 | 1 | EEEEEEEEE | 219.92 | 9 | 0 |
| d4096p1 | 2 | CCCCCCCCC | 102.09 | 0 | 9 |
| d4096p1 | 77 (last) | EEEEEEEEE | 188.19 | 9 | 0 |
| d4096p2 | 0 | EEEEEEEEE | 118.65 | 9 | 0 |
| d4096p2 | 1 | EEEEEEEEE | 118.12 | 9 | 0 |
| d4096p2 | 2 | CCCCCCCCC | 103.57 | 0 | 9 |

## The draft's evaluations that ran eagerly or were captured

| request | draft evaluation (order) | how | graph | GPU ms | after main pass |
|---|---|---|---|---|---|
| d4096p1 | 1 | eager | 6 | 6.64 | 0 |
| d4096p1 | 2 | capture | 6 | 4.19 | 0 |
| d4096p1 | 33 | eager | 7 | 9.51 | 8 |
| d4096p1 | 34 | capture | 7 | 7.77 | 8 |
| d4096p1 | 40 | eager | 6 | 5.62 | 10 |
| d4096p1 | 41 | capture | 6 | 3.45 | 10 |
| d4096p1 | 305 | eager | 11 | 0.94 | 77 |
