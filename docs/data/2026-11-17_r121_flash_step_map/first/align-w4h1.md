# <OUT>/st-w4h1.jsonl beside <OUT>/hp-w4h1.csv

- main context's ubatches in the trace: 819; card 0's evaluations in the profile: 819; the draft context's ubatches in the trace: 3126

| shift (profile index - trace index) | pairs | lower bound, ms | upper bound, ms | fits |
|---:|---:|---:|---:|---|
| -4 | 815 | 26827.228 (ubatch 4) | 920.388 (ubatch 202) | no |
| -3 | 816 | 26712.277 (ubatch 3) | 820.511 (ubatch 42) | no |
| -2 | 817 | 26468.051 (ubatch 2) | 721.231 (ubatch 41) | no |
| -1 | 818 | 25686.231 (ubatch 1) | 621.095 (ubatch 199) | no |
| 0 | 819 | 23504.312 (ubatch 0) | 522.381 (ubatch 199) | no |
| 1 | 818 | 423.273 (ubatch 427) | 426.395 (ubatch 190) | yes |
| 2 | 817 | 327.212 (ubatch 198) | -4381.730 (ubatch 347) | no |
| 3 | 816 | 228.471 (ubatch 197) | -6504.694 (ubatch 590) | no |
| 4 | 815 | 128.466 (ubatch 283) | -10681.481 (ubatch 338) | no |

- shift 1 (818 pairs): card 0's evaluations before the trace's first ubatch are the load's warm-up; a trace ubatch past the profile's last row is one the profile did not write before the exit
- card 0 bounds the offset (trace ms = profile ms + offset) to 423.273 … 426.395 ms, width 3.122 ms: from below by ubatch 427's inputs set, from above by ubatch 190's graphs queued
- the most evaluations inside a window, 10488, at offsets 423.488 … 423.618 ms (27 grid points of 0.005 ms, of 625); the offset: their middle, 423.553

| card | in a main ubatch | in a draft ubatch | in none |
|---|---:|---:|---:|
| 0 | 818 | 0 | 0 |
| 1 | 818 | 0 | 0 |
| 2 | 818 | 0 | 0 |
| 3 | 818 | 0 | 0 |
| 4 | 818 | 0 | 0 |
| 5 | 818 | 0 | 0 |
| 6 | 818 | 0 | 0 |
| 7 | 818 | 0 | 0 |
| 8 | 819 | 3125 | 0 |

t0 = 6233950.951
