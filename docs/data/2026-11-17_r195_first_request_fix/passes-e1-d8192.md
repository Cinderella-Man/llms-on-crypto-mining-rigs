# How every graph evaluation ran, pass by pass (<RES>/gp-e1-d8192.csv)

| piece | rows | evaluations on card 0 | host_ms span | request |
|---|---|---|---|---|
| 0 | 906 | 70 | 224493.3 to 233899.2 | d8192p1 |
| 1 | 883 | 68 | 237950.6 to 245897.7 | d8192p2 |

## Evaluations by how

| request | main passes | main: eager | capture | replay | other | draft evaluations | draft: eager | capture | replay | other |
|---|---|---|---|---|---|---|---|---|---|---|
| d8192p1 | 70 | 18 | 9 | 603 | 0 | 276 | 3 | 3 | 270 | 0 |
| d8192p2 | 68 | 18 | 9 | 585 | 0 | 271 | 0 | 0 | 271 | 0 |

## Main passes, d8192p1 against d8192p2

How on cards 0 to 8 (E eager, C capture, R replay, - not written); GPU ms summed over the cards written.

| pass | d8192p1 how | d8192p1 GPU ms | d8192p2 how | d8192p2 GPU ms | difference ms |
|---|---|---|---|---|---|
| 0 (prompt) | EEEEEEEEE | 223.17 | EEEEEEEEE | 106.71 | +116.45 |
| 1 (first verify) | EEEEEEEEE | 227.96 | EEEEEEEEE | 105.30 | +122.66 |
| 2 | CCCCCCCCC | 102.53 | CCCCCCCCC | 97.11 | +5.42 |
| 3 | RRRRRRRRR | 95.43 | RRRRRRRRR | 95.56 | -0.13 |
| 4 | RRRRRRRRR | 96.79 | RRRRRRRRR | 96.85 | -0.06 |
| 5 | RRRRRRRRR | 96.44 | RRRRRRRRR | 95.89 | +0.55 |
| 6 | RRRRRRRRR | 96.14 | RRRRRRRRR | 99.07 | -2.93 |
| 7 | RRRRRRRRR | 98.93 | RRRRRRRRR | 95.66 | +3.28 |
| 8 | RRRRRRRRR | 95.45 | RRRRRRRRR | 97.76 | -2.31 |
| 9 | RRRRRRRRR | 97.77 | RRRRRRRRR | 98.96 | -1.20 |
| 10 | RRRRRRRRR | 98.91 | RRRRRRRRR | 96.53 | +2.39 |
| 11 | RRRRRRRRR | 96.52 | RRRRRRRRR | 97.62 | -1.10 |
| 12 | RRRRRRRRR | 97.47 | RRRRRRRRR | 98.55 | -1.08 |
| 13 | RRRRRRRRR | 98.58 | RRRRRRRRR | 97.62 | +0.95 |
| 14 | RRRRRRRRR | 97.55 | RRRRRRRRR | 98.55 | -1.01 |
| 15 | RRRRRRRRR | 98.49 | RRRRRRRRR | 96.86 | +1.62 |
| 16 | RRRRRRRRR | 96.91 | RRRRRRRRR | 97.32 | -0.40 |
| 17 | RRRRRRRRR | 97.47 | RRRRRRRRR | 97.83 | -0.36 |
| 18 | RRRRRRRRR | 97.59 | RRRRRRRRR | 97.07 | +0.52 |
| 19 | RRRRRRRRR | 97.09 | RRRRRRRRR | 99.13 | -2.04 |
| 20 | RRRRRRRRR | 99.19 | RRRRRRRRR | 99.30 | -0.11 |
| 21 | RRRRRRRRR | 99.11 | RRRRRRRRR | 97.60 | +1.50 |
| 22 | RRRRRRRRR | 97.74 | RRRRRRRRR | 96.87 | +0.86 |
| 23 | RRRRRRRRR | 97.14 | RRRRRRRRR | 98.00 | -0.86 |
| 24 | RRRRRRRRR | 97.84 | RRRRRRRRR | 97.13 | +0.71 |
| 25 | RRRRRRRRR | 97.07 | RRRRRRRRR | 98.30 | -1.23 |
| 26 | RRRRRRRRR | 98.00 | RRRRRRRRR | 97.86 | +0.13 |
| 27 | RRRRRRRRR | 97.84 | RRRRRRRRR | 96.98 | +0.86 |
| 28 | RRRRRRRRR | 97.01 | RRRRRRRRR | 97.93 | -0.92 |
| 29 | RRRRRRRRR | 97.84 | RRRRRRRRR | 97.40 | +0.44 |
| 30 | RRRRRRRRR | 97.45 | RRRRRRRRR | 97.30 | +0.15 |
| 31 | RRRRRRRRR | 97.14 | RRRRRRRRR | 97.24 | -0.10 |
| 32 | RRRRRRRRR | 96.87 | RRRRRRRRR | 97.91 | -1.03 |
| 33 | RRRRRRRRR | 97.88 | RRRRRRRRR | 96.85 | +1.03 |
| 34 | RRRRRRRRR | 96.83 | RRRRRRRRR | 97.67 | -0.84 |
| 35 | RRRRRRRRR | 97.74 | RRRRRRRRR | 97.29 | +0.45 |
| 36 | RRRRRRRRR | 97.10 | RRRRRRRRR | 96.52 | +0.58 |
| 37 | RRRRRRRRR | 96.73 | RRRRRRRRR | 97.92 | -1.19 |
| 38 | RRRRRRRRR | 97.79 | RRRRRRRRR | 96.97 | +0.82 |
| 39 | RRRRRRRRR | 96.87 | RRRRRRRRR | 95.81 | +1.06 |
| 40 | RRRRRRRRR | 95.68 | RRRRRRRRR | 95.50 | +0.18 |
| 41 | RRRRRRRRR | 95.63 | RRRRRRRRR | 96.88 | -1.26 |
| 42 | RRRRRRRRR | 96.86 | RRRRRRRRR | 97.20 | -0.34 |
| 43 | RRRRRRRRR | 97.29 | RRRRRRRRR | 97.26 | +0.03 |
| 44 | RRRRRRRRR | 97.36 | RRRRRRRRR | 98.08 | -0.71 |
| 45 | RRRRRRRRR | 98.02 | RRRRRRRRR | 98.20 | -0.18 |
| 46 | RRRRRRRRR | 98.16 | RRRRRRRRR | 98.90 | -0.74 |
| 47 | RRRRRRRRR | 98.83 | RRRRRRRRR | 99.36 | -0.54 |
| 48 | RRRRRRRRR | 99.46 | RRRRRRRRR | 98.06 | +1.40 |
| 49 | RRRRRRRRR | 98.12 | RRRRRRRRR | 97.37 | +0.76 |
| 50 | RRRRRRRRR | 97.39 | RRRRRRRRR | 97.77 | -0.38 |
| 51 | RRRRRRRRR | 97.91 | RRRRRRRRR | 96.89 | +1.02 |
| 52 | RRRRRRRRR | 96.65 | RRRRRRRRR | 98.42 | -1.77 |
| 53 | RRRRRRRRR | 98.27 | RRRRRRRRR | 98.05 | +0.22 |
| 54 | RRRRRRRRR | 97.88 | RRRRRRRRR | 97.82 | +0.06 |
| 55 | RRRRRRRRR | 97.75 | RRRRRRRRR | 97.58 | +0.17 |
| 56 | RRRRRRRRR | 97.43 | RRRRRRRRR | 98.07 | -0.64 |
| 57 | RRRRRRRRR | 97.95 | RRRRRRRRR | 97.45 | +0.50 |
| 58 | RRRRRRRRR | 97.55 | RRRRRRRRR | 98.92 | -1.37 |
| 59 | RRRRRRRRR | 98.87 | RRRRRRRRR | 97.86 | +1.01 |
| 60 | RRRRRRRRR | 97.92 | RRRRRRRRR | 97.59 | +0.33 |
| 61 | RRRRRRRRR | 97.57 | RRRRRRRRR | 97.21 | +0.36 |
| 62 | RRRRRRRRR | 97.02 | RRRRRRRRR | 98.17 | -1.15 |
| 63 | RRRRRRRRR | 98.37 | RRRRRRRRR | 96.59 | +1.78 |
| 64 | RRRRRRRRR | 96.72 | RRRRRRRRR | 97.80 | -1.08 |
| 65 | RRRRRRRRR | 97.75 | RRRRRRRRR | 97.98 | -0.23 |
| 66 | RRRRRRRRR | 97.89 | RRRRRRRRR | 96.93 | +0.96 |
| 67 (d8192p2 last) | RRRRRRRRR | 96.93 | RRRRRRRRR | 97.80 | -0.86 |
| 68 | RRRRRRRRR | 97.68 |  | nan | +nan |
| 69 (d8192p1 last) | RRRRRRRRR | 97.79 |  | nan | +nan |

## The first verify pass and the last pass, card by card

GPU ms of each card's evaluation, how in parentheses.

| pass | request | card 0 | card 1 | card 2 | card 3 | card 4 | card 5 | card 6 | card 7 | card 8 |
|---|---|---|---|---|---|---|---|---|---|---|
| first verify (1) | d8192p1 | 26.94 (E) | 28.89 (E) | 25.51 (E) | 27.75 (E) | 21.17 (E) | 27.92 (E) | 25.23 (E) | 26.22 (E) | 18.33 (E) |
| first verify (1) | d8192p2 | 12.75 (E) | 13.25 (E) | 11.93 (E) | 13.29 (E) | 10.03 (E) | 11.53 (E) | 11.51 (E) | 11.59 (E) | 9.40 (E) |
| last (69) | d8192p1 | 11.83 (R) | 12.39 (R) | 11.19 (R) | 12.33 (R) | 9.57 (R) | 10.96 (R) | 10.56 (R) | 10.53 (R) | 8.44 (R) |
| last (67) | d8192p2 | 11.73 (R) | 12.56 (R) | 11.29 (R) | 12.19 (R) | 9.57 (R) | 10.75 (R) | 10.52 (R) | 10.62 (R) | 8.56 (R) |
| second verify (2) | d8192p1 | 14.04 (C) | 12.81 (C) | 11.40 (C) | 12.75 (C) | 9.81 (C) | 11.08 (C) | 10.67 (C) | 10.95 (C) | 9.02 (C) |
| second verify (2) | d8192p2 | 12.96 (C) | 12.22 (C) | 10.86 (C) | 12.04 (C) | 9.34 (C) | 10.55 (C) | 10.26 (C) | 10.41 (C) | 8.48 (C) |

## Passes not replayed on every card

| request | pass | how | GPU ms | cards eager | cards captured |
|---|---|---|---|---|---|
| d8192p1 | 0 | EEEEEEEEE | 223.17 | 9 | 0 |
| d8192p1 | 1 | EEEEEEEEE | 227.96 | 9 | 0 |
| d8192p1 | 2 | CCCCCCCCC | 102.53 | 0 | 9 |
| d8192p2 | 0 | EEEEEEEEE | 106.71 | 9 | 0 |
| d8192p2 | 1 | EEEEEEEEE | 105.30 | 9 | 0 |
| d8192p2 | 2 | CCCCCCCCC | 97.11 | 0 | 9 |

## The draft's evaluations that ran eagerly or were captured

| request | draft evaluation (order) | how | graph | GPU ms | after main pass |
|---|---|---|---|---|---|
| d8192p1 | 1 | eager | 6 | 5.96 | 0 |
| d8192p1 | 2 | capture | 6 | 3.52 | 0 |
| d8192p1 | 21 | eager | 7 | 9.48 | 5 |
| d8192p1 | 22 | capture | 7 | 7.67 | 5 |
| d8192p1 | 28 | eager | 6 | 6.04 | 7 |
| d8192p1 | 29 | capture | 6 | 3.55 | 7 |
