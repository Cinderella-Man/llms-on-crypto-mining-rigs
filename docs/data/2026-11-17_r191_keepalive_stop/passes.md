# How every graph evaluation ran, pass by pass (<OUT>/gp-k1.csv)

| piece | rows | evaluations on card 0 | host_ms span | request |
|---|---|---|---|---|
| 0 | 215 | 17 | 23208.9 to 30370.2 | - |
| 1 | 90 | 9 | 35743.3 to 64709.6 | - |
| 2 | 1008 | 78 | 70336.5 to 80139.9 | d4096p1 |
| 3 | 988 | 76 | 84182.5 to 92722.1 | d4096p2 |

## Evaluations by how

| request | main passes | main: eager | capture | replay | other | draft evaluations | draft: eager | capture | replay | other |
|---|---|---|---|---|---|---|---|---|---|---|
| d4096p1 | 78 | 27 | 9 | 666 | 0 | 306 | 4 | 3 | 299 | 0 |
| d4096p2 | 76 | 18 | 9 | 657 | 0 | 304 | 0 | 0 | 304 | 0 |

## Main passes, d4096p1 against d4096p2

How on cards 0 to 8 (E eager, C capture, R replay, - not written); GPU ms summed over the cards written.

| pass | d4096p1 how | d4096p1 GPU ms | d4096p2 how | d4096p2 GPU ms | difference ms |
|---|---|---|---|---|---|
| 0 (prompt) | EEEEEEEEE | 102.29 | EEEEEEEEE | 100.57 | +1.71 |
| 1 (first verify) | EEEEEEEEE | 100.67 | EEEEEEEEE | 102.94 | -2.27 |
| 2 | CCCCCCCCC | 98.78 | CCCCCCCCC | 94.07 | +4.71 |
| 3 | RRRRRRRRR | 95.19 | RRRRRRRRR | 95.95 | -0.75 |
| 4 | RRRRRRRRR | 95.17 | RRRRRRRRR | 95.02 | +0.16 |
| 5 | RRRRRRRRR | 93.87 | RRRRRRRRR | 93.55 | +0.32 |
| 6 | RRRRRRRRR | 93.62 | RRRRRRRRR | 93.48 | +0.14 |
| 7 | RRRRRRRRR | 94.35 | RRRRRRRRR | 94.12 | +0.23 |
| 8 | RRRRRRRRR | 94.37 | RRRRRRRRR | 94.14 | +0.23 |
| 9 | RRRRRRRRR | 94.31 | RRRRRRRRR | 94.54 | -0.23 |
| 10 | RRRRRRRRR | 94.73 | RRRRRRRRR | 94.70 | +0.03 |
| 11 | RRRRRRRRR | 92.35 | RRRRRRRRR | 92.48 | -0.13 |
| 12 | RRRRRRRRR | 92.56 | RRRRRRRRR | 92.57 | -0.01 |
| 13 | RRRRRRRRR | 94.11 | RRRRRRRRR | 94.34 | -0.24 |
| 14 | RRRRRRRRR | 93.71 | RRRRRRRRR | 93.51 | +0.19 |
| 15 | RRRRRRRRR | 93.92 | RRRRRRRRR | 94.05 | -0.13 |
| 16 | RRRRRRRRR | 94.15 | RRRRRRRRR | 94.22 | -0.07 |
| 17 | RRRRRRRRR | 93.80 | RRRRRRRRR | 93.72 | +0.07 |
| 18 | RRRRRRRRR | 94.60 | RRRRRRRRR | 94.50 | +0.10 |
| 19 | RRRRRRRRR | 94.34 | RRRRRRRRR | 94.28 | +0.06 |
| 20 | RRRRRRRRR | 94.19 | RRRRRRRRR | 94.34 | -0.14 |
| 21 | RRRRRRRRR | 93.87 | RRRRRRRRR | 93.96 | -0.09 |
| 22 | RRRRRRRRR | 93.37 | RRRRRRRRR | 93.61 | -0.24 |
| 23 | RRRRRRRRR | 94.13 | RRRRRRRRR | 94.24 | -0.11 |
| 24 | RRRRRRRRR | 93.95 | RRRRRRRRR | 93.89 | +0.06 |
| 25 | RRRRRRRRR | 93.93 | RRRRRRRRR | 93.74 | +0.19 |
| 26 | RRRRRRRRR | 91.28 | RRRRRRRRR | 91.43 | -0.15 |
| 27 | RRRRRRRRR | 93.69 | RRRRRRRRR | 93.72 | -0.03 |
| 28 | RRRRRRRRR | 94.61 | RRRRRRRRR | 94.59 | +0.02 |
| 29 | RRRRRRRRR | 91.48 | RRRRRRRRR | 91.90 | -0.43 |
| 30 | RRRRRRRRR | 91.96 | RRRRRRRRR | 92.02 | -0.07 |
| 31 | RRRRRRRRR | 92.27 | RRRRRRRRR | 92.24 | +0.03 |
| 32 | RRRRRRRRR | 92.70 | RRRRRRRRR | 92.91 | -0.21 |
| 33 | RRRRRRRRR | 93.86 | RRRRRRRRR | 94.00 | -0.13 |
| 34 | RRRRRRRRR | 92.89 | RRRRRRRRR | 93.03 | -0.14 |
| 35 | RRRRRRRRR | 94.93 | RRRRRRRRR | 95.09 | -0.16 |
| 36 | RRRRRRRRR | 94.28 | RRRRRRRRR | 94.20 | +0.08 |
| 37 | RRRRRRRRR | 92.54 | RRRRRRRRR | 92.63 | -0.10 |
| 38 | RRRRRRRRR | 93.19 | RRRRRRRRR | 93.33 | -0.15 |
| 39 | RRRRRRRRR | 93.07 | RRRRRRRRR | 93.20 | -0.13 |
| 40 | RRRRRRRRR | 93.31 | RRRRRRRRR | 93.15 | +0.16 |
| 41 | RRRRRRRRR | 95.45 | RRRRRRRRR | 95.38 | +0.08 |
| 42 | RRRRRRRRR | 95.16 | RRRRRRRRR | 95.21 | -0.05 |
| 43 | RRRRRRRRR | 95.42 | RRRRRRRRR | 95.51 | -0.09 |
| 44 | RRRRRRRRR | 95.10 | RRRRRRRRR | 95.07 | +0.03 |
| 45 | RRRRRRRRR | 93.32 | RRRRRRRRR | 93.41 | -0.09 |
| 46 | RRRRRRRRR | 94.95 | RRRRRRRRR | 94.87 | +0.08 |
| 47 | RRRRRRRRR | 95.21 | RRRRRRRRR | 95.16 | +0.05 |
| 48 | RRRRRRRRR | 95.12 | RRRRRRRRR | 95.19 | -0.07 |
| 49 | RRRRRRRRR | 95.50 | RRRRRRRRR | 95.39 | +0.11 |
| 50 | RRRRRRRRR | 94.41 | RRRRRRRRR | 94.51 | -0.09 |
| 51 | RRRRRRRRR | 95.51 | RRRRRRRRR | 95.57 | -0.06 |
| 52 | RRRRRRRRR | 96.06 | RRRRRRRRR | 96.14 | -0.08 |
| 53 | RRRRRRRRR | 91.46 | RRRRRRRRR | 91.58 | -0.12 |
| 54 | RRRRRRRRR | 92.70 | RRRRRRRRR | 92.69 | +0.01 |
| 55 | RRRRRRRRR | 95.71 | RRRRRRRRR | 95.79 | -0.09 |
| 56 | RRRRRRRRR | 95.42 | RRRRRRRRR | 95.39 | +0.04 |
| 57 | RRRRRRRRR | 93.02 | RRRRRRRRR | 93.06 | -0.04 |
| 58 | RRRRRRRRR | 95.94 | RRRRRRRRR | 95.84 | +0.11 |
| 59 | RRRRRRRRR | 95.32 | RRRRRRRRR | 95.35 | -0.04 |
| 60 | RRRRRRRRR | 95.20 | RRRRRRRRR | 95.08 | +0.12 |
| 61 | RRRRRRRRR | 93.87 | RRRRRRRRR | 93.83 | +0.05 |
| 62 | RRRRRRRRR | 94.32 | RRRRRRRRR | 94.35 | -0.03 |
| 63 | RRRRRRRRR | 93.08 | RRRRRRRRR | 93.08 | -0.00 |
| 64 | RRRRRRRRR | 93.85 | RRRRRRRRR | 93.84 | +0.01 |
| 65 | RRRRRRRRR | 93.49 | RRRRRRRRR | 93.49 | +0.01 |
| 66 | RRRRRRRRR | 92.44 | RRRRRRRRR | 92.40 | +0.05 |
| 67 | RRRRRRRRR | 93.06 | RRRRRRRRR | 92.98 | +0.08 |
| 68 | RRRRRRRRR | 94.50 | RRRRRRRRR | 94.39 | +0.11 |
| 69 | RRRRRRRRR | 95.32 | RRRRRRRRR | 95.31 | +0.01 |
| 70 | RRRRRRRRR | 92.84 | RRRRRRRRR | 92.72 | +0.12 |
| 71 | RRRRRRRRR | 93.65 | RRRRRRRRR | 93.80 | -0.15 |
| 72 | RRRRRRRRR | 94.12 | RRRRRRRRR | 94.32 | -0.20 |
| 73 | RRRRRRRRR | 94.17 | RRRRRRRRR | 94.44 | -0.27 |
| 74 | RRRRRRRRR | 94.04 | RRRRRRRRR | 93.94 | +0.10 |
| 75 (d4096p2 last) | RRRRRRRRR | 94.86 | RRRRRRRRR | 94.88 | -0.02 |
| 76 | RRRRRRRRR | 95.78 |  | nan | +nan |
| 77 (d4096p1 last) | EEEEEEEEE | 70.26 |  | nan | +nan |

## The first verify pass and the last pass, card by card

GPU ms of each card's evaluation, how in parentheses.

| pass | request | card 0 | card 1 | card 2 | card 3 | card 4 | card 5 | card 6 | card 7 | card 8 |
|---|---|---|---|---|---|---|---|---|---|---|
| first verify (1) | d4096p1 | 12.51 (E) | 12.46 (E) | 11.59 (E) | 12.70 (E) | 9.53 (E) | 11.23 (E) | 10.63 (E) | 11.05 (E) | 8.96 (E) |
| first verify (1) | d4096p2 | 12.40 (E) | 12.59 (E) | 13.20 (E) | 12.81 (E) | 9.62 (E) | 11.37 (E) | 10.72 (E) | 11.15 (E) | 9.07 (E) |
| last (77) | d4096p1 | 8.84 (E) | 8.49 (E) | 7.86 (E) | 8.38 (E) | 6.57 (E) | 8.08 (E) | 7.44 (E) | 8.05 (E) | 6.54 (E) |
| last (75) | d4096p2 | 11.38 (R) | 11.74 (R) | 11.11 (R) | 11.95 (R) | 9.15 (R) | 10.72 (R) | 10.01 (R) | 10.64 (R) | 8.17 (R) |
| second verify (2) | d4096p1 | 13.58 (C) | 11.86 (C) | 11.24 (C) | 12.18 (C) | 9.31 (C) | 10.86 (C) | 10.40 (C) | 10.68 (C) | 8.67 (C) |
| second verify (2) | d4096p2 | 12.53 (C) | 11.31 (C) | 11.28 (C) | 11.59 (C) | 8.84 (C) | 10.35 (C) | 9.87 (C) | 10.15 (C) | 8.16 (C) |

## Passes not replayed on every card

| request | pass | how | GPU ms | cards eager | cards captured |
|---|---|---|---|---|---|
| d4096p1 | 0 | EEEEEEEEE | 102.29 | 9 | 0 |
| d4096p1 | 1 | EEEEEEEEE | 100.67 | 9 | 0 |
| d4096p1 | 2 | CCCCCCCCC | 98.78 | 0 | 9 |
| d4096p1 | 77 (last) | EEEEEEEEE | 70.26 | 9 | 0 |
| d4096p2 | 0 | EEEEEEEEE | 100.57 | 9 | 0 |
| d4096p2 | 1 | EEEEEEEEE | 102.94 | 9 | 0 |
| d4096p2 | 2 | CCCCCCCCC | 94.07 | 0 | 9 |

## The draft's evaluations that ran eagerly or were captured

| request | draft evaluation (order) | how | graph | GPU ms | after main pass |
|---|---|---|---|---|---|
| d4096p1 | 1 | eager | 6 | 1003.87 | 0 |
| d4096p1 | 2 | capture | 6 | 4.19 | 0 |
| d4096p1 | 33 | eager | 7 | 6.88 | 8 |
| d4096p1 | 34 | capture | 7 | 7.72 | 8 |
| d4096p1 | 40 | eager | 6 | 3.29 | 10 |
| d4096p1 | 41 | capture | 6 | 3.44 | 10 |
| d4096p1 | 305 | eager | 11 | 0.90 | 77 |
