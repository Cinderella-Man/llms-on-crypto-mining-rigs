# How every graph evaluation ran, pass by pass (<RES>/gp-k1-d6144.csv)

| piece | rows | evaluations on card 0 | host_ms span | request |
|---|---|---|---|---|
| 0 | 914 | 71 | 141923.9 to 151506.4 | d6144p1 |
| 1 | 881 | 68 | 155548.7 to 163630.7 | d6144p2 |

## Evaluations by how

| request | main passes | main: eager | capture | replay | other | draft evaluations | draft: eager | capture | replay | other |
|---|---|---|---|---|---|---|---|---|---|---|
| d6144p1 | 71 | 27 | 18 | 594 | 0 | 275 | 4 | 5 | 266 | 0 |
| d6144p2 | 68 | 18 | 9 | 585 | 0 | 269 | 0 | 0 | 269 | 0 |

## Main passes, d6144p1 against d6144p2

How on cards 0 to 8 (E eager, C capture, R replay, - not written); GPU ms summed over the cards written.

| pass | d6144p1 how | d6144p1 GPU ms | d6144p2 how | d6144p2 GPU ms | difference ms |
|---|---|---|---|---|---|
| 0 (prompt) | CCCCCCCCC | 1167.62 | EEEEEEEEE | 104.97 | +1062.65 |
| 1 (first verify) | EEEEEEEEE | 107.68 | EEEEEEEEE | 108.54 | -0.86 |
| 2 | EEEEEEEEE | 117.33 | CCCCCCCCC | 101.62 | +15.70 |
| 3 | CCCCCCCCC | 105.35 | RRRRRRRRR | 99.86 | +5.50 |
| 4 | RRRRRRRRR | 98.84 | RRRRRRRRR | 100.02 | -1.18 |
| 5 | RRRRRRRRR | 98.66 | RRRRRRRRR | 99.01 | -0.35 |
| 6 | RRRRRRRRR | 99.65 | RRRRRRRRR | 98.27 | +1.37 |
| 7 | RRRRRRRRR | 98.89 | RRRRRRRRR | 98.79 | +0.10 |
| 8 | RRRRRRRRR | 98.19 | RRRRRRRRR | 100.80 | -2.60 |
| 9 | RRRRRRRRR | 98.70 | RRRRRRRRR | 100.92 | -2.22 |
| 10 | RRRRRRRRR | 100.61 | RRRRRRRRR | 101.61 | -1.00 |
| 11 | RRRRRRRRR | 100.79 | RRRRRRRRR | 100.70 | +0.08 |
| 12 | RRRRRRRRR | 101.52 | RRRRRRRRR | 100.76 | +0.77 |
| 13 | RRRRRRRRR | 100.64 | RRRRRRRRR | 100.53 | +0.11 |
| 14 | RRRRRRRRR | 100.64 | RRRRRRRRR | 101.87 | -1.23 |
| 15 | RRRRRRRRR | 100.44 | RRRRRRRRR | 99.21 | +1.23 |
| 16 | RRRRRRRRR | 101.64 | RRRRRRRRR | 98.31 | +3.34 |
| 17 | RRRRRRRRR | 99.25 | RRRRRRRRR | 99.15 | +0.10 |
| 18 | RRRRRRRRR | 98.36 | RRRRRRRRR | 98.39 | -0.04 |
| 19 | RRRRRRRRR | 98.98 | RRRRRRRRR | 100.57 | -1.59 |
| 20 | RRRRRRRRR | 98.44 | RRRRRRRRR | 101.60 | -3.16 |
| 21 | RRRRRRRRR | 100.50 | RRRRRRRRR | 100.02 | +0.48 |
| 22 | RRRRRRRRR | 101.38 | RRRRRRRRR | 100.68 | +0.70 |
| 23 | RRRRRRRRR | 99.86 | RRRRRRRRR | 99.50 | +0.36 |
| 24 | RRRRRRRRR | 100.59 | RRRRRRRRR | 99.35 | +1.24 |
| 25 | RRRRRRRRR | 99.44 | RRRRRRRRR | 101.36 | -1.91 |
| 26 | RRRRRRRRR | 99.25 | RRRRRRRRR | 98.84 | +0.41 |
| 27 | RRRRRRRRR | 101.20 | RRRRRRRRR | 100.94 | +0.26 |
| 28 | RRRRRRRRR | 98.73 | RRRRRRRRR | 100.31 | -1.58 |
| 29 | RRRRRRRRR | 100.84 | RRRRRRRRR | 99.78 | +1.06 |
| 30 | RRRRRRRRR | 100.14 | RRRRRRRRR | 98.39 | +1.75 |
| 31 | RRRRRRRRR | 99.63 | RRRRRRRRR | 99.95 | -0.32 |
| 32 | RRRRRRRRR | 97.98 | RRRRRRRRR | 99.59 | -1.62 |
| 33 | RRRRRRRRR | 99.85 | RRRRRRRRR | 99.83 | +0.02 |
| 34 | RRRRRRRRR | 99.41 | RRRRRRRRR | 99.60 | -0.19 |
| 35 | RRRRRRRRR | 99.65 | RRRRRRRRR | 99.62 | +0.02 |
| 36 | RRRRRRRRR | 99.31 | RRRRRRRRR | 99.47 | -0.15 |
| 37 | RRRRRRRRR | 99.50 | RRRRRRRRR | 99.76 | -0.26 |
| 38 | RRRRRRRRR | 99.25 | RRRRRRRRR | 99.07 | +0.18 |
| 39 | RRRRRRRRR | 99.64 | RRRRRRRRR | 100.55 | -0.91 |
| 40 | RRRRRRRRR | 98.99 | RRRRRRRRR | 101.03 | -2.04 |
| 41 | RRRRRRRRR | 100.53 | RRRRRRRRR | 100.46 | +0.07 |
| 42 | RRRRRRRRR | 100.88 | RRRRRRRRR | 98.99 | +1.89 |
| 43 | RRRRRRRRR | 100.41 | RRRRRRRRR | 99.68 | +0.73 |
| 44 | RRRRRRRRR | 98.71 | RRRRRRRRR | 98.82 | -0.11 |
| 45 | RRRRRRRRR | 99.62 | RRRRRRRRR | 100.15 | -0.54 |
| 46 | RRRRRRRRR | 98.65 | RRRRRRRRR | 99.55 | -0.90 |
| 47 | RRRRRRRRR | 99.88 | RRRRRRRRR | 98.62 | +1.26 |
| 48 | RRRRRRRRR | 99.40 | RRRRRRRRR | 98.71 | +0.68 |
| 49 | RRRRRRRRR | 98.63 | RRRRRRRRR | 98.87 | -0.24 |
| 50 | RRRRRRRRR | 98.49 | RRRRRRRRR | 98.99 | -0.50 |
| 51 | RRRRRRRRR | 98.70 | RRRRRRRRR | 99.06 | -0.36 |
| 52 | RRRRRRRRR | 98.57 | RRRRRRRRR | 98.57 | -0.00 |
| 53 | RRRRRRRRR | 99.12 | RRRRRRRRR | 98.97 | +0.15 |
| 54 | RRRRRRRRR | 98.55 | RRRRRRRRR | 98.93 | -0.38 |
| 55 | RRRRRRRRR | 98.65 | RRRRRRRRR | 99.82 | -1.17 |
| 56 | RRRRRRRRR | 98.73 | RRRRRRRRR | 99.38 | -0.65 |
| 57 | RRRRRRRRR | 99.87 | RRRRRRRRR | 97.97 | +1.90 |
| 58 | RRRRRRRRR | 99.52 | RRRRRRRRR | 98.32 | +1.20 |
| 59 | RRRRRRRRR | 97.76 | RRRRRRRRR | 99.01 | -1.24 |
| 60 | RRRRRRRRR | 98.12 | RRRRRRRRR | 99.34 | -1.21 |
| 61 | RRRRRRRRR | 98.91 | RRRRRRRRR | 98.70 | +0.20 |
| 62 | RRRRRRRRR | 99.08 | RRRRRRRRR | 99.90 | -0.81 |
| 63 | RRRRRRRRR | 98.70 | RRRRRRRRR | 100.31 | -1.61 |
| 64 | RRRRRRRRR | 99.69 | RRRRRRRRR | 99.93 | -0.23 |
| 65 | RRRRRRRRR | 100.34 | RRRRRRRRR | 101.75 | -1.41 |
| 66 | RRRRRRRRR | 99.73 | RRRRRRRRR | 99.74 | -0.01 |
| 67 (d6144p2 last) | RRRRRRRRR | 101.43 | RRRRRRRRR | 99.49 | +1.94 |
| 68 | RRRRRRRRR | 99.67 |  | nan | +nan |
| 69 | RRRRRRRRR | 99.20 |  | nan | +nan |
| 70 (d6144p1 last) | EEEEEEEEE | 187.11 |  | nan | +nan |

## The first verify pass and the last pass, card by card

GPU ms of each card's evaluation, how in parentheses.

| pass | request | card 0 | card 1 | card 2 | card 3 | card 4 | card 5 | card 6 | card 7 | card 8 |
|---|---|---|---|---|---|---|---|---|---|---|
| first verify (1) | d6144p1 | 13.09 (E) | 13.77 (E) | 12.45 (E) | 13.17 (E) | 9.97 (E) | 12.34 (E) | 12.56 (E) | 12.30 (E) | 8.03 (E) |
| first verify (1) | d6144p2 | 12.81 (E) | 13.49 (E) | 11.98 (E) | 13.62 (E) | 10.13 (E) | 11.73 (E) | 11.77 (E) | 11.72 (E) | 11.29 (E) |
| last (70) | d6144p1 | 20.95 (E) | 24.57 (E) | 22.41 (E) | 22.37 (E) | 16.93 (E) | 24.69 (E) | 20.52 (E) | 21.99 (E) | 12.66 (E) |
| last (67) | d6144p2 | 11.75 (R) | 12.57 (R) | 11.39 (R) | 12.68 (R) | 9.75 (R) | 10.94 (R) | 10.92 (R) | 10.86 (R) | 8.64 (R) |
| second verify (2) | d6144p1 | 12.50 (E) | 13.35 (E) | 11.90 (E) | 13.45 (E) | 10.07 (E) | 12.58 (E) | 12.56 (E) | 12.66 (E) | 18.25 (E) |
| second verify (2) | d6144p2 | 13.34 (C) | 12.62 (C) | 11.17 (C) | 12.54 (C) | 9.51 (C) | 10.82 (C) | 10.65 (C) | 10.62 (C) | 10.35 (C) |

## Passes not replayed on every card

| request | pass | how | GPU ms | cards eager | cards captured |
|---|---|---|---|---|---|
| d6144p1 | 0 | CCCCCCCCC | 1167.62 | 0 | 9 |
| d6144p1 | 1 | EEEEEEEEE | 107.68 | 9 | 0 |
| d6144p1 | 2 | EEEEEEEEE | 117.33 | 9 | 0 |
| d6144p1 | 3 | CCCCCCCCC | 105.35 | 0 | 9 |
| d6144p1 | 70 (last) | EEEEEEEEE | 187.11 | 9 | 0 |
| d6144p2 | 0 | EEEEEEEEE | 104.97 | 9 | 0 |
| d6144p2 | 1 | EEEEEEEEE | 108.54 | 9 | 0 |
| d6144p2 | 2 | CCCCCCCCC | 101.62 | 0 | 9 |

## The draft's evaluations that ran eagerly or were captured

| request | draft evaluation (order) | how | graph | GPU ms | after main pass |
|---|---|---|---|---|---|
| d6144p1 | 0 | capture | 4 | 10.05 | 1 |
| d6144p1 | 1 | capture | 5 | 1.32 | 1 |
| d6144p1 | 2 | eager | 6 | 5.90 | 1 |
| d6144p1 | 3 | capture | 6 | 3.57 | 1 |
| d6144p1 | 10 | eager | 7 | 6.93 | 3 |
| d6144p1 | 11 | capture | 7 | 7.67 | 3 |
| d6144p1 | 18 | eager | 6 | 5.93 | 5 |
| d6144p1 | 19 | capture | 6 | 3.48 | 5 |
| d6144p1 | 274 | eager | 13 | 1.48 | 70 |
