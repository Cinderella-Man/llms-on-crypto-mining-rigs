# R191 against R192: the same pair, EAGER=0 against EAGER=1

## The passes that are not replayed on every card

How on cards 0 to 8 (E kernel by kernel, C captured, R replayed, - not written) and GPU ms summed over the cards written. "First − repeat" is the first request's excess over its repeat in the same run.

| pass | EAGER=1 first | GPU ms | EAGER=1 repeat | GPU ms | first − repeat | EAGER=0 first | GPU ms | EAGER=0 repeat | GPU ms | first − repeat |
|---|---|---:|---|---:|---:|---|---:|---|---:|---:|
| 0: the prompt's last 4 tokens | EEEEEEEEE  | 235.65 | EEEEEEEEE  | 118.65 | +117.00 | EEEEEEEEE  | 102.29 | EEEEEEEEE  | 100.57 | +1.71 |
| 1: the first verify pass | EEEEEEEEE  | 219.92 | EEEEEEEEE  | 118.12 | +101.80 | EEEEEEEEE  | 100.67 | EEEEEEEEE  | 102.94 | -2.27 |
| 2: the second verify pass | CCCCCCCCC  | 102.09 | CCCCCCCCC  | 103.58 | -1.48 | CCCCCCCCC  | 98.78 | CCCCCCCCC  | 94.07 | +4.71 |
| last pass | EEEEEEEEE (77) | 188.19 | not written (76) |  |  | EEEEEEEEE (77) | 70.26 | not written (76) |  |  |

Replayed passes (every card R), the median GPU ms over the nine cards per pass:

| run | first request | repeat | passes (first, repeat) |
|---|---:|---:|---|
| EAGER=1 | 94.17 | 94.27 | 74, 73 |
| EAGER=0 | 94.12 | 94.12 | 74, 73 |

## The first verify pass and the prompt's last 4 tokens, card by card

GPU ms of each card's evaluation (how in parentheses): the main model's first and second evaluation on each card in the request (card 8: its first and second evaluations that are not the draft's are found by graph_passes.py; here card 8 is shown as the pass total minus cards 0 to 7).

| pass | run | request | card 0 | card 1 | card 2 | card 3 | card 4 | card 5 | card 6 | card 7 | card 8 | sum |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prompt's last 4 | EAGER=1 | first | 27.39 (E) | 31.73 (E) | 29.34 (E) | 26.14 (E) | 20.39 (E) | 30.69 (E) | 27.83 (E) | 28.42 (E) | 13.71 | 235.65 |
| prompt's last 4 | EAGER=1 | repeat | 11.90 (E) | 15.39 (E) | 14.29 (E) | 15.61 (E) | 12.41 (E) | 14.59 (E) | 13.60 (E) | 13.28 (E) | 7.59 | 118.65 |
| prompt's last 4 | EAGER=0 | first | 11.78 (E) | 12.04 (E) | 11.27 (E) | 12.31 (E) | 9.50 (E) | 11.49 (E) | 10.83 (E) | 13.39 (E) | 9.68 | 102.29 |
| prompt's last 4 | EAGER=0 | repeat | 11.81 (E) | 12.01 (E) | 14.27 (E) | 12.32 (E) | 9.44 (E) | 11.30 (E) | 10.70 (E) | 11.15 (E) | 7.58 | 100.57 |
| first verify | EAGER=1 | first | 26.15 (E) | 28.26 (E) | 25.89 (E) | 27.25 (E) | 20.66 (E) | 27.08 (E) | 23.51 (E) | 24.02 (E) | 17.11 | 219.92 |
| first verify | EAGER=1 | repeat | 12.41 (E) | 16.02 (E) | 13.15 (E) | 16.28 (E) | 11.74 (E) | 13.46 (E) | 12.69 (E) | 13.31 (E) | 9.05 | 118.12 |
| first verify | EAGER=0 | first | 12.51 (E) | 12.46 (E) | 11.59 (E) | 12.70 (E) | 9.53 (E) | 11.23 (E) | 10.63 (E) | 11.05 (E) | 8.96 | 100.67 |
| first verify | EAGER=0 | repeat | 12.40 (E) | 12.59 (E) | 13.20 (E) | 12.81 (E) | 9.62 (E) | 11.37 (E) | 10.72 (E) | 11.15 (E) | 9.07 | 102.94 |

## The keep-alive's counts per request

From the line the server prints after each reply (the difference to the reply before). At EAGER=0 the switch counts the evaluations that run kernel by kernel and acts on none: "acted on" and the stops stay 0, and shapes new to a card are not tracked.

| run | request | gen tokens | kernel by kernel | acted on (new shapes) | block stops | new sizes outside a graph | their stops | graphs reused (llama) | debug: warm-up done | warm-up reset | update failed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EAGER=1 | warmup 0 at 0 | 32 | 72 | 64 | 2 | 689 | 54 | 13 | 13 | 1 | 0 |
| EAGER=1 | prime 0 at 4096 | 1 | 9 | 9 | 9 | 90 | 28 | 13 | 0 | 0 | 72 |
| EAGER=1 | decode 1 at 4096 | 256 | 31 | 30 | 30 | 0 | 0 | 88 | 12 | 3 | 15 |
| EAGER=1 | decode 2 at 4096 | 256 | 28 | 0 | 0 | 0 | 0 | 163 | 9 | 9 | 0 |
| EAGER=0 | warmup 0 at 0 | 32 | 72 | 0 | 0 | 0 | 0 | 13 | 13 | 1 | 0 |
| EAGER=0 | prime 0 at 4096 | 1 | 9 | 0 | 0 | 0 | 0 | 13 | 0 | 0 | 72 |
| EAGER=0 | decode 1 at 4096 | 256 | 31 | 0 | 0 | 0 | 0 | 88 | 12 | 3 | 15 |
| EAGER=0 | decode 2 at 4096 | 256 | 27 | 0 | 0 | 0 | 0 | 163 | 10 | 9 | 0 |

## Request times in the timed load (ms)

| request | EAGER=1 prompt ms | EAGER=1 gen ms | EAGER=0 prompt ms | EAGER=0 gen ms | prompt: EAGER=0 − EAGER=1 | gen: EAGER=0 − EAGER=1 | reply md5 the same |
|---|---:|---:|---:|---:|---:|---:|---|
| warmup 0 at 0 | 3096.93 | 2827.71 | 4747.53 | 4485.55 | +1650.60 | +1657.83 | yes |
| prime 0 at 4096 | 29278.01 | 0.00 | 29116.17 | 0.00 | -161.84 | +0.00 | yes |
| decode 1 at 4096 | 1616.85 | 8947.48 | 1730.72 | 9685.70 | +113.87 | +738.22 | yes |
| decode 2 at 4096 | 149.31 | 8656.51 | 132.13 | 8632.35 | -17.18 | -24.16 | yes |

The first request against its repeat in each run:

| run | first gen ms | repeat gen ms | generation: first − repeat | first prompt ms | repeat prompt ms | drafted, accepted (first; repeat) | same reply |
|---|---:|---:|---:|---:|---:|---|---|
| EAGER=1 | 8947.48 | 8656.51 | +290.97 | 1616.85 | 149.31 | 229, 178; 229, 178 | yes |
| EAGER=0 | 9685.70 | 8632.35 | +1053.35 | 1730.72 | 132.13 | 229, 178; 229, 178 | yes |

## Each evaluation against the same one in the other run (a wait of about 1 s for a running block would show here)

Per piece of the graph-mode profile, each card's evaluations are paired in order with the other run's (EAGER=0 against EAGER=1; the same requests, so the same evaluations). A launch that waits for a running block is inside its evaluation's event pair. Listed: the evaluations of each run, the longest kernel-by-kernel one, and the largest increase of any evaluation; every pair that grew by more than 200 ms is listed below.

| piece | evaluations (EAGER=1; EAGER=0) | kernel by kernel | longest kernel-by-kernel ms (EAGER=1; EAGER=0) | largest increase ms | where (card, how, order on the card) | pairs over +200 ms |
|---|---|---|---|---:|---|---:|
| warm-up request (depth 0) | 215; 215 | 42; 42 | 303.99; 1009.84 | +999.50 | card 8, eager/eager, 2 | 4 |
| prime (4,032 tokens) | 90; 90 | 9; 9 | 30.89; 14.06 | +26.70 | card 0, capture/capture, 0 | 0 |
| first request | 1008; 1008 | 31; 31 | 31.73; 1003.87 | +997.23 | card 8, eager/eager, 2 | 1 |
| repeat | 988; 988 | 18; 18 | 16.28; 14.27 | +0.60 | card 2, replay/replay, 3 | 0 |

Every pair that grew by more than 200 ms:

| piece | card | order on the card | how (EAGER=1/EAGER=0) | EAGER=1 ms | EAGER=0 ms | increase |
|---|---:|---:|---|---:|---:|---:|
| warm-up request (depth 0) | 5 | 1 | eager/eager | 31.80 | 247.80 | +216.00 |
| warm-up request (depth 0) | 6 | 1 | eager/eager | 31.59 | 236.99 | +205.39 |
| warm-up request (depth 0) | 7 | 1 | eager/eager | 30.99 | 266.72 | +235.73 |
| warm-up request (depth 0) | 8 | 2 | eager/eager | 10.34 | 1009.84 | +999.50 |
| first request | 8 | 2 | eager/eager | 6.64 | 1003.87 | +997.23 |
