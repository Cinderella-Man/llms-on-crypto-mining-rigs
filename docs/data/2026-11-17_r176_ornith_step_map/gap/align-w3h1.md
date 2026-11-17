# <OUT>/st-w3h1.jsonl beside <OUT>/hp-w3h1.csv

- main context's ubatches in the trace: 455; card 0's evaluations in the profile: 455; the draft context's ubatches in the trace: 1281

| shift (profile index - trace index) | pairs | lower bound, ms | upper bound, ms | fits |
|---:|---:|---:|---:|---|
| -4 | 451 | 20183.886 (ubatch 258) | 1637.160 (ubatch 175) | no |
| -3 | 452 | 19438.782 (ubatch 258) | 1494.169 (ubatch 175) | no |
| -2 | 453 | 14423.122 (ubatch 258) | 1351.549 (ubatch 173) | no |
| -1 | 454 | 13660.221 (ubatch 258) | 1208.558 (ubatch 173) | no |
| 0 | 455 | 9787.668 (ubatch 258) | 1066.245 (ubatch 94) | no |
| 1 | 454 | 923.924 (ubatch 23) | 926.376 (ubatch 143) | yes |
| 2 | 453 | 784.123 (ubatch 93) | -6241.912 (ubatch 257) | no |
| 3 | 452 | 642.012 (ubatch 171) | -8199.455 (ubatch 224) | no |
| 4 | 451 | 498.973 (ubatch 170) | -11884.241 (ubatch 255) | no |

- shift 1 (454 pairs): card 0's evaluations before the trace's first ubatch are the load's warm-up; a trace ubatch past the profile's last row is one the profile did not write before the exit
- card 0 bounds the offset (trace ms = profile ms + offset) to 923.924 … 926.376 ms, width 2.452 ms: from below by ubatch 23's inputs set, from above by ubatch 143's graphs queued
- the most evaluations inside a window, 5367, at offsets 923.989 … 924.214 ms (46 grid points of 0.005 ms, of 491); the offset: their middle, 924.101

| card | in a main ubatch | in a draft ubatch | in none |
|---|---:|---:|---:|
| 0 | 454 | 0 | 0 |
| 1 | 454 | 0 | 0 |
| 2 | 454 | 0 | 0 |
| 3 | 454 | 0 | 0 |
| 4 | 454 | 0 | 0 |
| 5 | 454 | 0 | 0 |
| 6 | 454 | 0 | 0 |
| 7 | 454 | 0 | 0 |
| 8 | 455 | 1280 | 0 |

t0 = 25107697.286
