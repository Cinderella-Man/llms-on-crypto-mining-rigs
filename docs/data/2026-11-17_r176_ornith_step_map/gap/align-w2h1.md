# <OUT>/st-w2h1.jsonl beside <OUT>/hp-w2h1.csv

- main context's ubatches in the trace: 601; card 0's evaluations in the profile: 601; the draft context's ubatches in the trace: 1158

| shift (profile index - trace index) | pairs | lower bound, ms | upper bound, ms | fits |
|---:|---:|---:|---:|---|
| -4 | 597 | 20151.856 (ubatch 330) | 1428.391 (ubatch 228) | no |
| -3 | 598 | 19416.621 (ubatch 330) | 1321.222 (ubatch 227) | no |
| -2 | 599 | 14395.652 (ubatch 330) | 1213.627 (ubatch 226) | no |
| -1 | 600 | 13632.720 (ubatch 330) | 1105.446 (ubatch 225) | no |
| 0 | 601 | 9751.433 (ubatch 330) | 998.731 (ubatch 254) | no |
| 1 | 600 | 889.878 (ubatch 168) | 892.347 (ubatch 69) | yes |
| 2 | 599 | 783.507 (ubatch 253) | -6278.983 (ubatch 329) | no |
| 3 | 598 | 676.824 (ubatch 223) | -8321.089 (ubatch 296) | no |
| 4 | 597 | 568.613 (ubatch 223) | -11928.298 (ubatch 327) | no |

- shift 1 (600 pairs): card 0's evaluations before the trace's first ubatch are the load's warm-up; a trace ubatch past the profile's last row is one the profile did not write before the exit
- card 0 bounds the offset (trace ms = profile ms + offset) to 889.878 … 892.347 ms, width 2.469 ms: from below by ubatch 168's inputs set, from above by ubatch 69's graphs queued
- the most evaluations inside a window, 6558, at offsets 889.928 … 890.148 ms (45 grid points of 0.005 ms, of 494); the offset: their middle, 890.038

| card | in a main ubatch | in a draft ubatch | in none |
|---|---:|---:|---:|
| 0 | 600 | 0 | 0 |
| 1 | 600 | 0 | 0 |
| 2 | 600 | 0 | 0 |
| 3 | 600 | 0 | 0 |
| 4 | 600 | 0 | 0 |
| 5 | 600 | 0 | 0 |
| 6 | 600 | 0 | 0 |
| 7 | 600 | 0 | 0 |
| 8 | 601 | 1157 | 0 |

t0 = 23274704.784
