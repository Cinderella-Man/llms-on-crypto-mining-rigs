# <OUT>/st-k1.jsonl beside <OUT>/hp-k1.csv

- main context's ubatches in the trace: 495; card 0's evaluations in the profile: 495; the draft context's ubatches in the trace: 1814

| shift (profile index - trace index) | pairs | lower bound, ms | upper bound, ms | fits |
|---:|---:|---:|---:|---|
| -4 | 491 | 31758.472 (ubatch 4) | 1048.562 (ubatch 15) | no |
| -3 | 492 | 31644.053 (ubatch 3) | 944.225 (ubatch 15) | no |
| -2 | 493 | 31396.667 (ubatch 2) | 841.891 (ubatch 8) | no |
| -1 | 494 | 30588.787 (ubatch 1) | 738.305 (ubatch 15) | no |
| 0 | 495 | 28320.675 (ubatch 0) | 636.678 (ubatch 6) | no |
| 1 | 494 | 533.100 (ubatch 337) | 536.455 (ubatch 303) | yes |
| 2 | 493 | 431.762 (ubatch 5) | -4812.579 (ubatch 354) | no |
| 3 | 492 | 328.172 (ubatch 13) | -9960.392 (ubatch 26) | no |
| 4 | 491 | 226.483 (ubatch 5) | -11809.115 (ubatch 352) | no |

- shift 1 (494 pairs): card 0's evaluations before the trace's first ubatch are the load's warm-up; a trace ubatch past the profile's last row is one the profile did not write before the exit
- card 0 bounds the offset (trace ms = profile ms + offset) to 533.100 … 536.455 ms, width 3.355 ms: from below by ubatch 337's inputs set, from above by ubatch 303's graphs queued
- the most evaluations inside a window, 6260, at offsets 533.315 … 533.450 ms (28 grid points of 0.005 ms, of 671); the offset: their middle, 533.383

| card | in a main ubatch | in a draft ubatch | in none |
|---|---:|---:|---:|
| 0 | 494 | 0 | 0 |
| 1 | 494 | 0 | 0 |
| 2 | 494 | 0 | 0 |
| 3 | 494 | 0 | 0 |
| 4 | 494 | 0 | 0 |
| 5 | 494 | 0 | 0 |
| 6 | 494 | 0 | 0 |
| 7 | 494 | 0 | 0 |
| 8 | 495 | 1813 | 0 |

t0 = 19987777.380
