# How every graph evaluation ran, pass by pass (<RES>/gp-k1-d8192.csv)

| piece | rows | evaluations on card 0 | host_ms span | request |
|---|---|---|---|---|
| 0 | 906 | 70 | 224938.1 to 234327.0 | d8192p1 |
| 1 | 883 | 68 | 238377.6 to 246333.1 | d8192p2 |

## Evaluations by how

| request | main passes | main: eager | capture | replay | other | draft evaluations | draft: eager | capture | replay | other |
|---|---|---|---|---|---|---|---|---|---|---|
| d8192p1 | 70 | 18 | 9 | 603 | 0 | 276 | 3 | 3 | 270 | 0 |
| d8192p2 | 68 | 18 | 9 | 585 | 0 | 271 | 0 | 0 | 271 | 0 |

## Main passes, d8192p1 against d8192p2

How on cards 0 to 8 (E eager, C capture, R replay, - not written); GPU ms summed over the cards written.

| pass | d8192p1 how | d8192p1 GPU ms | d8192p2 how | d8192p2 GPU ms | difference ms |
|---|---|---|---|---|---|
| 0 (prompt) | EEEEEEEEE | 113.22 | EEEEEEEEE | 106.77 | +6.45 |
| 1 (first verify) | EEEEEEEEE | 227.31 | EEEEEEEEE | 105.30 | +122.01 |
| 2 | CCCCCCCCC | 102.77 | CCCCCCCCC | 97.17 | +5.60 |
| 3 | RRRRRRRRR | 95.52 | RRRRRRRRR | 95.49 | +0.03 |
| 4 | RRRRRRRRR | 96.96 | RRRRRRRRR | 96.84 | +0.12 |
| 5 | RRRRRRRRR | 96.61 | RRRRRRRRR | 95.93 | +0.69 |
| 6 | RRRRRRRRR | 96.32 | RRRRRRRRR | 99.09 | -2.77 |
| 7 | RRRRRRRRR | 99.04 | RRRRRRRRR | 95.46 | +3.58 |
| 8 | RRRRRRRRR | 95.45 | RRRRRRRRR | 97.67 | -2.22 |
| 9 | RRRRRRRRR | 97.77 | RRRRRRRRR | 98.93 | -1.16 |
| 10 | RRRRRRRRR | 99.11 | RRRRRRRRR | 96.56 | +2.55 |
| 11 | RRRRRRRRR | 96.65 | RRRRRRRRR | 97.66 | -1.01 |
| 12 | RRRRRRRRR | 97.58 | RRRRRRRRR | 98.55 | -0.97 |
| 13 | RRRRRRRRR | 98.53 | RRRRRRRRR | 97.85 | +0.69 |
| 14 | RRRRRRRRR | 97.64 | RRRRRRRRR | 98.67 | -1.03 |
| 15 | RRRRRRRRR | 98.52 | RRRRRRRRR | 96.94 | +1.58 |
| 16 | RRRRRRRRR | 96.90 | RRRRRRRRR | 97.57 | -0.67 |
| 17 | RRRRRRRRR | 97.49 | RRRRRRRRR | 97.78 | -0.29 |
| 18 | RRRRRRRRR | 97.79 | RRRRRRRRR | 96.94 | +0.85 |
| 19 | RRRRRRRRR | 97.20 | RRRRRRRRR | 99.10 | -1.90 |
| 20 | RRRRRRRRR | 99.22 | RRRRRRRRR | 99.25 | -0.03 |
| 21 | RRRRRRRRR | 99.20 | RRRRRRRRR | 97.63 | +1.58 |
| 22 | RRRRRRRRR | 97.68 | RRRRRRRRR | 97.29 | +0.39 |
| 23 | RRRRRRRRR | 97.01 | RRRRRRRRR | 98.02 | -1.01 |
| 24 | RRRRRRRRR | 97.96 | RRRRRRRRR | 97.22 | +0.74 |
| 25 | RRRRRRRRR | 97.23 | RRRRRRRRR | 98.16 | -0.94 |
| 26 | RRRRRRRRR | 98.19 | RRRRRRRRR | 98.00 | +0.19 |
| 27 | RRRRRRRRR | 97.99 | RRRRRRRRR | 96.86 | +1.14 |
| 28 | RRRRRRRRR | 97.16 | RRRRRRRRR | 97.67 | -0.51 |
| 29 | RRRRRRRRR | 97.87 | RRRRRRRRR | 97.25 | +0.62 |
| 30 | RRRRRRRRR | 97.30 | RRRRRRRRR | 97.46 | -0.16 |
| 31 | RRRRRRRRR | 97.54 | RRRRRRRRR | 97.23 | +0.31 |
| 32 | RRRRRRRRR | 97.21 | RRRRRRRRR | 97.97 | -0.76 |
| 33 | RRRRRRRRR | 98.03 | RRRRRRRRR | 96.93 | +1.10 |
| 34 | RRRRRRRRR | 97.14 | RRRRRRRRR | 97.80 | -0.66 |
| 35 | RRRRRRRRR | 97.82 | RRRRRRRRR | 97.25 | +0.56 |
| 36 | RRRRRRRRR | 97.16 | RRRRRRRRR | 96.64 | +0.52 |
| 37 | RRRRRRRRR | 96.66 | RRRRRRRRR | 97.98 | -1.32 |
| 38 | RRRRRRRRR | 97.91 | RRRRRRRRR | 96.95 | +0.96 |
| 39 | RRRRRRRRR | 96.85 | RRRRRRRRR | 95.79 | +1.06 |
| 40 | RRRRRRRRR | 96.08 | RRRRRRRRR | 95.68 | +0.40 |
| 41 | RRRRRRRRR | 95.71 | RRRRRRRRR | 96.87 | -1.16 |
| 42 | RRRRRRRRR | 96.97 | RRRRRRRRR | 97.32 | -0.34 |
| 43 | RRRRRRRRR | 97.32 | RRRRRRRRR | 97.41 | -0.09 |
| 44 | RRRRRRRRR | 97.49 | RRRRRRRRR | 98.01 | -0.52 |
| 45 | RRRRRRRRR | 98.21 | RRRRRRRRR | 98.17 | +0.04 |
| 46 | RRRRRRRRR | 98.34 | RRRRRRRRR | 98.79 | -0.45 |
| 47 | RRRRRRRRR | 98.87 | RRRRRRRRR | 99.44 | -0.57 |
| 48 | RRRRRRRRR | 99.62 | RRRRRRRRR | 97.97 | +1.65 |
| 49 | RRRRRRRRR | 98.15 | RRRRRRRRR | 97.48 | +0.67 |
| 50 | RRRRRRRRR | 97.45 | RRRRRRRRR | 98.01 | -0.56 |
| 51 | RRRRRRRRR | 97.95 | RRRRRRRRR | 96.82 | +1.13 |
| 52 | RRRRRRRRR | 96.93 | RRRRRRRRR | 98.40 | -1.47 |
| 53 | RRRRRRRRR | 98.46 | RRRRRRRRR | 97.86 | +0.60 |
| 54 | RRRRRRRRR | 98.06 | RRRRRRRRR | 97.80 | +0.26 |
| 55 | RRRRRRRRR | 97.82 | RRRRRRRRR | 97.43 | +0.39 |
| 56 | RRRRRRRRR | 97.48 | RRRRRRRRR | 98.02 | -0.54 |
| 57 | RRRRRRRRR | 98.14 | RRRRRRRRR | 97.51 | +0.63 |
| 58 | RRRRRRRRR | 97.73 | RRRRRRRRR | 98.97 | -1.24 |
| 59 | RRRRRRRRR | 98.94 | RRRRRRRRR | 98.04 | +0.91 |
| 60 | RRRRRRRRR | 98.04 | RRRRRRRRR | 97.62 | +0.42 |
| 61 | RRRRRRRRR | 97.52 | RRRRRRRRR | 97.05 | +0.47 |
| 62 | RRRRRRRRR | 97.31 | RRRRRRRRR | 98.31 | -1.00 |
| 63 | RRRRRRRRR | 98.23 | RRRRRRRRR | 96.72 | +1.52 |
| 64 | RRRRRRRRR | 96.76 | RRRRRRRRR | 97.85 | -1.09 |
| 65 | RRRRRRRRR | 97.90 | RRRRRRRRR | 97.95 | -0.05 |
| 66 | RRRRRRRRR | 98.07 | RRRRRRRRR | 97.05 | +1.02 |
| 67 (d8192p2 last) | RRRRRRRRR | 97.00 | RRRRRRRRR | 97.82 | -0.82 |
| 68 | RRRRRRRRR | 97.97 |  | nan | +nan |
| 69 (d8192p1 last) | RRRRRRRRR | 97.84 |  | nan | +nan |

## The first verify pass and the last pass, card by card

GPU ms of each card's evaluation, how in parentheses.

| pass | request | card 0 | card 1 | card 2 | card 3 | card 4 | card 5 | card 6 | card 7 | card 8 |
|---|---|---|---|---|---|---|---|---|---|---|
| first verify (1) | d8192p1 | 26.84 (E) | 29.51 (E) | 26.17 (E) | 27.55 (E) | 21.25 (E) | 27.11 (E) | 24.91 (E) | 25.54 (E) | 18.42 (E) |
| first verify (1) | d8192p2 | 12.69 (E) | 13.28 (E) | 11.94 (E) | 13.33 (E) | 9.99 (E) | 11.50 (E) | 11.51 (E) | 11.61 (E) | 9.45 (E) |
| last (69) | d8192p1 | 11.87 (R) | 12.35 (R) | 11.16 (R) | 12.40 (R) | 9.56 (R) | 10.99 (R) | 10.56 (R) | 10.52 (R) | 8.43 (R) |
| last (67) | d8192p2 | 11.72 (R) | 12.54 (R) | 11.25 (R) | 12.36 (R) | 9.56 (R) | 10.73 (R) | 10.49 (R) | 10.63 (R) | 8.55 (R) |
| second verify (2) | d8192p1 | 14.19 (C) | 12.76 (C) | 11.47 (C) | 12.78 (C) | 9.78 (C) | 11.09 (C) | 10.69 (C) | 10.97 (C) | 9.02 (C) |
| second verify (2) | d8192p2 | 12.93 (C) | 12.17 (C) | 10.86 (C) | 12.15 (C) | 9.32 (C) | 10.59 (C) | 10.26 (C) | 10.40 (C) | 8.49 (C) |

## Passes not replayed on every card

| request | pass | how | GPU ms | cards eager | cards captured |
|---|---|---|---|---|---|
| d8192p1 | 0 | EEEEEEEEE | 113.22 | 9 | 0 |
| d8192p1 | 1 | EEEEEEEEE | 227.31 | 9 | 0 |
| d8192p1 | 2 | CCCCCCCCC | 102.77 | 0 | 9 |
| d8192p2 | 0 | EEEEEEEEE | 106.77 | 9 | 0 |
| d8192p2 | 1 | EEEEEEEEE | 105.30 | 9 | 0 |
| d8192p2 | 2 | CCCCCCCCC | 97.17 | 0 | 9 |

## The draft's evaluations that ran eagerly or were captured

| request | draft evaluation (order) | how | graph | GPU ms | after main pass |
|---|---|---|---|---|---|
| d8192p1 | 1 | eager | 6 | 5.99 | 0 |
| d8192p1 | 2 | capture | 6 | 3.51 | 0 |
| d8192p1 | 21 | eager | 7 | 6.93 | 5 |
| d8192p1 | 22 | capture | 7 | 7.65 | 5 |
| d8192p1 | 28 | eager | 6 | 6.06 | 7 |
| d8192p1 | 29 | capture | 6 | 3.52 | 7 |
