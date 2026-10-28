| run | passed | min per task (median) | min per task (mean) | prompt s per task | decode s per task | generation tok/s | first token p50 s | p90 s | max s | restores of the previous prompt's end | re-read tokens | re-read s | requests that kept cached ids (tokens kept) | end-of-prompt checkpoints per request |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| d4 | 12/12 | 1.07 | 2.22 | 25.3 | 104.7 | 45.9 | 2.61 | 6.18 | 59.8 | 15 | 16,954 | 94.4 | 0 (0) | 1.15 |
| q8-d4b | 12/12 | 1.42 | 2.52 | 29.0 | 119.4 | 44.6 | 2.56 | 6.97 | 80.9 | 16 | 29,083 | 150.3 | 0 (0) | 1.16 |
| base | 12/12 | 1.19 | 1.89 | 23.6 | 86.0 | 45.0 | 2.30 | 6.48 | 82.5 | 13 | 17,294 | 90.8 | 0 (0) | 1.16 |
| tr | 12/12 | 1.23 | 2.02 | 18.6 | 98.5 | 43.7 | 2.48 | 6.10 | 12.7 | 11 | 0 | 0.0 | 23 (138,676) | 1.15 |
| trtail | 12/12 | 1.05 | 1.62 | 16.2 | 77.6 | 45.7 | 2.38 | 5.98 | 12.2 | 11 | 0 | 0.0 | 15 (55,038) | 0.32 |

| run | requests processing ≤ 64 tokens | their prompt ms (median) | requests processing 65–256 | their prompt ms (median) |
|---|---|---|---|---|
| d4 | 18 | 668 | 12 | 1377 |
| q8-d4b | 24 | 696 | 6 | 1483 |
| base | 20 | 739 | 11 | 1495 |
| tr | 25 | 684 | 13 | 1534 |
| trtail | 22 | 379 | 9 | 1359 |
