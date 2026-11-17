# How every graph evaluation ran, pass by pass (<RES>/gp-e1-d6144.csv)

| piece | rows | evaluations on card 0 | host_ms span | request |
|---|---|---|---|---|
| 0 | 914 | 71 | 142206.7 to 151913.0 | d6144p1 |
| 1 | 881 | 68 | 155959.2 to 164005.3 | d6144p2 |

## Evaluations by how

| request | main passes | main: eager | capture | replay | other | draft evaluations | draft: eager | capture | replay | other |
|---|---|---|---|---|---|---|---|---|---|---|
| d6144p1 | 71 | 27 | 18 | 594 | 0 | 275 | 4 | 5 | 266 | 0 |
| d6144p2 | 68 | 18 | 9 | 585 | 0 | 269 | 0 | 0 | 269 | 0 |

## Main passes, d6144p1 against d6144p2

How on cards 0 to 8 (E eager, C capture, R replay, - not written); GPU ms summed over the cards written.

| pass | d6144p1 how | d6144p1 GPU ms | d6144p2 how | d6144p2 GPU ms | difference ms |
|---|---|---|---|---|---|
| 0 (prompt) | CCCCCCCCC | 1176.78 | EEEEEEEEE | 103.57 | +1073.21 |
| 1 (first verify) | EEEEEEEEE | 220.00 | EEEEEEEEE | 107.78 | +112.22 |
| 2 | EEEEEEEEE | 224.90 | CCCCCCCCC | 99.90 | +125.00 |
| 3 | CCCCCCCCC | 104.99 | RRRRRRRRR | 98.77 | +6.22 |
| 4 | RRRRRRRRR | 98.87 | RRRRRRRRR | 99.74 | -0.87 |
| 5 | RRRRRRRRR | 98.58 | RRRRRRRRR | 98.89 | -0.31 |
| 6 | RRRRRRRRR | 99.51 | RRRRRRRRR | 98.19 | +1.32 |
| 7 | RRRRRRRRR | 98.86 | RRRRRRRRR | 98.60 | +0.26 |
| 8 | RRRRRRRRR | 98.13 | RRRRRRRRR | 100.66 | -2.53 |
| 9 | RRRRRRRRR | 98.60 | RRRRRRRRR | 100.81 | -2.21 |
| 10 | RRRRRRRRR | 100.53 | RRRRRRRRR | 101.46 | -0.93 |
| 11 | RRRRRRRRR | 100.61 | RRRRRRRRR | 100.61 | -0.01 |
| 12 | RRRRRRRRR | 101.56 | RRRRRRRRR | 100.66 | +0.89 |
| 13 | RRRRRRRRR | 100.62 | RRRRRRRRR | 100.31 | +0.30 |
| 14 | RRRRRRRRR | 100.67 | RRRRRRRRR | 101.72 | -1.05 |
| 15 | RRRRRRRRR | 100.37 | RRRRRRRRR | 99.35 | +1.02 |
| 16 | RRRRRRRRR | 101.57 | RRRRRRRRR | 98.40 | +3.17 |
| 17 | RRRRRRRRR | 99.20 | RRRRRRRRR | 99.04 | +0.16 |
| 18 | RRRRRRRRR | 98.27 | RRRRRRRRR | 98.38 | -0.12 |
| 19 | RRRRRRRRR | 98.95 | RRRRRRRRR | 100.57 | -1.62 |
| 20 | RRRRRRRRR | 98.12 | RRRRRRRRR | 101.44 | -3.32 |
| 21 | RRRRRRRRR | 100.40 | RRRRRRRRR | 99.80 | +0.60 |
| 22 | RRRRRRRRR | 101.36 | RRRRRRRRR | 100.58 | +0.78 |
| 23 | RRRRRRRRR | 99.93 | RRRRRRRRR | 99.38 | +0.55 |
| 24 | RRRRRRRRR | 100.55 | RRRRRRRRR | 99.39 | +1.16 |
| 25 | RRRRRRRRR | 99.43 | RRRRRRRRR | 101.23 | -1.80 |
| 26 | RRRRRRRRR | 99.30 | RRRRRRRRR | 98.86 | +0.44 |
| 27 | RRRRRRRRR | 101.01 | RRRRRRRRR | 100.92 | +0.09 |
| 28 | RRRRRRRRR | 98.84 | RRRRRRRRR | 100.27 | -1.43 |
| 29 | RRRRRRRRR | 100.86 | RRRRRRRRR | 99.52 | +1.34 |
| 30 | RRRRRRRRR | 99.91 | RRRRRRRRR | 98.16 | +1.74 |
| 31 | RRRRRRRRR | 99.49 | RRRRRRRRR | 99.80 | -0.31 |
| 32 | RRRRRRRRR | 97.93 | RRRRRRRRR | 99.44 | -1.50 |
| 33 | RRRRRRRRR | 99.69 | RRRRRRRRR | 99.59 | +0.09 |
| 34 | RRRRRRRRR | 99.37 | RRRRRRRRR | 99.40 | -0.03 |
| 35 | RRRRRRRRR | 99.68 | RRRRRRRRR | 99.50 | +0.18 |
| 36 | RRRRRRRRR | 99.31 | RRRRRRRRR | 99.40 | -0.09 |
| 37 | RRRRRRRRR | 99.62 | RRRRRRRRR | 99.55 | +0.07 |
| 38 | RRRRRRRRR | 99.42 | RRRRRRRRR | 99.00 | +0.42 |
| 39 | RRRRRRRRR | 99.52 | RRRRRRRRR | 100.50 | -0.98 |
| 40 | RRRRRRRRR | 98.86 | RRRRRRRRR | 100.84 | -1.98 |
| 41 | RRRRRRRRR | 100.52 | RRRRRRRRR | 100.47 | +0.05 |
| 42 | RRRRRRRRR | 100.73 | RRRRRRRRR | 98.79 | +1.94 |
| 43 | RRRRRRRRR | 100.35 | RRRRRRRRR | 99.81 | +0.54 |
| 44 | RRRRRRRRR | 98.78 | RRRRRRRRR | 98.63 | +0.15 |
| 45 | RRRRRRRRR | 99.58 | RRRRRRRRR | 99.92 | -0.35 |
| 46 | RRRRRRRRR | 98.58 | RRRRRRRRR | 99.58 | -1.00 |
| 47 | RRRRRRRRR | 99.76 | RRRRRRRRR | 98.59 | +1.17 |
| 48 | RRRRRRRRR | 99.37 | RRRRRRRRR | 98.52 | +0.85 |
| 49 | RRRRRRRRR | 98.58 | RRRRRRRRR | 98.67 | -0.09 |
| 50 | RRRRRRRRR | 98.42 | RRRRRRRRR | 98.88 | -0.46 |
| 51 | RRRRRRRRR | 98.56 | RRRRRRRRR | 98.87 | -0.31 |
| 52 | RRRRRRRRR | 98.77 | RRRRRRRRR | 98.51 | +0.26 |
| 53 | RRRRRRRRR | 98.94 | RRRRRRRRR | 98.80 | +0.14 |
| 54 | RRRRRRRRR | 98.56 | RRRRRRRRR | 98.67 | -0.11 |
| 55 | RRRRRRRRR | 98.63 | RRRRRRRRR | 99.82 | -1.19 |
| 56 | RRRRRRRRR | 98.72 | RRRRRRRRR | 99.47 | -0.75 |
| 57 | RRRRRRRRR | 99.71 | RRRRRRRRR | 97.84 | +1.87 |
| 58 | RRRRRRRRR | 99.28 | RRRRRRRRR | 98.07 | +1.21 |
| 59 | RRRRRRRRR | 97.78 | RRRRRRRRR | 99.01 | -1.23 |
| 60 | RRRRRRRRR | 98.00 | RRRRRRRRR | 99.25 | -1.25 |
| 61 | RRRRRRRRR | 98.85 | RRRRRRRRR | 98.64 | +0.21 |
| 62 | RRRRRRRRR | 99.15 | RRRRRRRRR | 99.88 | -0.73 |
| 63 | RRRRRRRRR | 98.57 | RRRRRRRRR | 100.27 | -1.70 |
| 64 | RRRRRRRRR | 99.81 | RRRRRRRRR | 99.94 | -0.13 |
| 65 | RRRRRRRRR | 100.22 | RRRRRRRRR | 101.59 | -1.38 |
| 66 | RRRRRRRRR | 99.83 | RRRRRRRRR | 99.67 | +0.16 |
| 67 (d6144p2 last) | RRRRRRRRR | 101.49 | RRRRRRRRR | 99.40 | +2.09 |
| 68 | RRRRRRRRR | 99.59 |  | nan | +nan |
| 69 | RRRRRRRRR | 99.15 |  | nan | +nan |
| 70 (d6144p1 last) | EEEEEEEEE | 186.17 |  | nan | +nan |

## The first verify pass and the last pass, card by card

GPU ms of each card's evaluation, how in parentheses.

| pass | request | card 0 | card 1 | card 2 | card 3 | card 4 | card 5 | card 6 | card 7 | card 8 |
|---|---|---|---|---|---|---|---|---|---|---|
| first verify (1) | d6144p1 | 28.85 (E) | 28.66 (E) | 25.23 (E) | 27.18 (E) | 21.52 (E) | 26.28 (E) | 23.94 (E) | 25.64 (E) | 12.69 (E) |
| first verify (1) | d6144p2 | 13.01 (E) | 13.46 (E) | 12.01 (E) | 13.54 (E) | 10.12 (E) | 11.72 (E) | 11.79 (E) | 11.72 (E) | 10.41 (E) |
| last (70) | d6144p1 | 20.61 (E) | 24.39 (E) | 22.79 (E) | 22.46 (E) | 16.91 (E) | 24.33 (E) | 20.45 (E) | 21.68 (E) | 12.56 (E) |
| last (67) | d6144p2 | 11.70 (R) | 12.60 (R) | 11.36 (R) | 12.61 (R) | 9.74 (R) | 10.97 (R) | 10.88 (R) | 10.87 (R) | 8.66 (R) |
| second verify (2) | d6144p1 | 26.12 (E) | 28.69 (E) | 25.62 (E) | 27.27 (E) | 20.98 (E) | 27.76 (E) | 24.55 (E) | 25.99 (E) | 17.91 (E) |
| second verify (2) | d6144p2 | 13.39 (C) | 12.65 (C) | 11.10 (C) | 12.49 (C) | 9.52 (C) | 10.81 (C) | 10.66 (C) | 10.57 (C) | 8.72 (C) |

## Passes not replayed on every card

| request | pass | how | GPU ms | cards eager | cards captured |
|---|---|---|---|---|---|
| d6144p1 | 0 | CCCCCCCCC | 1176.78 | 0 | 9 |
| d6144p1 | 1 | EEEEEEEEE | 220.00 | 9 | 0 |
| d6144p1 | 2 | EEEEEEEEE | 224.90 | 9 | 0 |
| d6144p1 | 3 | CCCCCCCCC | 104.99 | 0 | 9 |
| d6144p1 | 70 (last) | EEEEEEEEE | 186.17 | 9 | 0 |
| d6144p2 | 0 | EEEEEEEEE | 103.57 | 9 | 0 |
| d6144p2 | 1 | EEEEEEEEE | 107.78 | 9 | 0 |
| d6144p2 | 2 | CCCCCCCCC | 99.90 | 0 | 9 |

## The draft's evaluations that ran eagerly or were captured

| request | draft evaluation (order) | how | graph | GPU ms | after main pass |
|---|---|---|---|---|---|
| d6144p1 | 0 | capture | 4 | 10.10 | 1 |
| d6144p1 | 1 | capture | 5 | 1.33 | 1 |
| d6144p1 | 2 | eager | 6 | 5.73 | 1 |
| d6144p1 | 3 | capture | 6 | 3.53 | 1 |
| d6144p1 | 10 | eager | 7 | 9.37 | 3 |
| d6144p1 | 11 | capture | 7 | 7.70 | 3 |
| d6144p1 | 18 | eager | 6 | 5.85 | 5 |
| d6144p1 | 19 | capture | 6 | 3.48 | 5 |
| d6144p1 | 274 | eager | 13 | 1.46 | 70 |
