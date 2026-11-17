# <OUT>/st-g1.jsonl beside <OUT>/hp-g1.csv

- main context's ubatches in the trace: 185; card 0's evaluations in the profile: 185; the draft context's ubatches in the trace: 688

| shift (profile index - trace index) | pairs | lower bound, ms | upper bound, ms | fits |
|---:|---:|---:|---:|---|
| -4 | 181 | 26835.229 (ubatch 4) | 969.099 (ubatch 15) | no |
| -3 | 182 | 26719.754 (ubatch 3) | 865.575 (ubatch 15) | no |
| -2 | 183 | 26475.013 (ubatch 2) | 764.946 (ubatch 15) | no |
| -1 | 184 | 25702.041 (ubatch 1) | 663.759 (ubatch 15) | no |
| 0 | 185 | 23505.586 (ubatch 0) | 563.147 (ubatch 14) | no |
| 1 | 184 | 459.264 (ubatch 18) | 462.683 (ubatch 82) | yes |
| 2 | 183 | 358.659 (ubatch 13) | -4358.959 (ubatch 27) | no |
| 3 | 182 | 257.905 (ubatch 13) | -10047.980 (ubatch 26) | no |
| 4 | 181 | 156.822 (ubatch 11) | -11234.546 (ubatch 26) | no |

- shift 1 (184 pairs): card 0's evaluations before the trace's first ubatch are the load's warm-up; a trace ubatch past the profile's last row is one the profile did not write before the exit
- card 0 bounds the offset (trace ms = profile ms + offset) to 459.264 … 462.683 ms, width 3.419 ms: from below by ubatch 18's inputs set, from above by ubatch 82's graphs queued
- the most evaluations inside a window, 2344, at offsets 459.554 … 459.689 ms (28 grid points of 0.005 ms, of 684); the offset: their middle, 459.621

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

t0 = 16465940.373
