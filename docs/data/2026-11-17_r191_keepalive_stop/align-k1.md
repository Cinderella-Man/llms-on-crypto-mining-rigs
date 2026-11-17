# <OUT>/st-k1.jsonl beside <OUT>/hp-k1.csv

- main context's ubatches in the trace: 185; card 0's evaluations in the profile: 185; the draft context's ubatches in the trace: 688

| shift (profile index - trace index) | pairs | lower bound, ms | upper bound, ms | fits |
|---:|---:|---:|---:|---|
| -4 | 181 | 28232.350 (ubatch 4) | 975.654 (ubatch 8) | no |
| -3 | 182 | 28125.428 (ubatch 3) | 873.879 (ubatch 8) | no |
| -2 | 183 | 28017.290 (ubatch 2) | 773.794 (ubatch 7) | no |
| -1 | 184 | 25545.600 (ubatch 1) | 672.780 (ubatch 6) | no |
| 0 | 185 | 23342.612 (ubatch 0) | 572.885 (ubatch 6) | no |
| 1 | 184 | 470.114 (ubatch 18) | 473.505 (ubatch 102) | yes |
| 2 | 183 | 383.187 (ubatch 17) | -4701.992 (ubatch 18) | no |
| 3 | 182 | 270.512 (ubatch 4) | -10002.046 (ubatch 26) | no |
| 4 | 181 | 169.509 (ubatch 4) | -11449.045 (ubatch 26) | no |

- shift 1 (184 pairs): card 0's evaluations before the trace's first ubatch are the load's warm-up; a trace ubatch past the profile's last row is one the profile did not write before the exit
- card 0 bounds the offset (trace ms = profile ms + offset) to 470.114 … 473.505 ms, width 3.391 ms: from below by ubatch 18's inputs set, from above by ubatch 102's graphs queued
- the most evaluations inside a window, 2344, at offsets 470.409 … 470.539 ms (27 grid points of 0.005 ms, of 679); the offset: their middle, 470.474

| card | in a main ubatch | in a draft ubatch | in none |
|---|---:|---:|---:|
| 0 | 184 | 0 | 0 |
| 1 | 184 | 0 | 0 |
| 2 | 184 | 0 | 0 |
| 3 | 184 | 0 | 0 |
| 4 | 184 | 0 | 0 |
| 5 | 184 | 0 | 0 |
| 6 | 184 | 0 | 0 |
| 7 | 184 | 0 | 0 |
| 8 | 185 | 687 | 0 |

t0 = 18115181.121
