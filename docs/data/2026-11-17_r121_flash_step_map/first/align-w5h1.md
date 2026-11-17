# <OUT>/st-w5h1.jsonl beside <OUT>/hp-w5h1.csv

- main context's ubatches in the trace: 698; card 0's evaluations in the profile: 698; the draft context's ubatches in the trace: 3307

| shift (profile index - trace index) | pairs | lower bound, ms | upper bound, ms | fits |
|---:|---:|---:|---:|---|
| -4 | 694 | 27033.194 (ubatch 4) | 1129.801 (ubatch 122) | no |
| -3 | 695 | 26841.316 (ubatch 3) | 992.123 (ubatch 122) | no |
| -2 | 696 | 26465.516 (ubatch 2) | 854.866 (ubatch 122) | no |
| -1 | 697 | 25794.797 (ubatch 1) | 717.979 (ubatch 121) | no |
| 0 | 698 | 23552.388 (ubatch 0) | 582.222 (ubatch 129) | no |
| 1 | 697 | 446.156 (ubatch 16) | 449.572 (ubatch 167) | yes |
| 2 | 696 | 323.711 (ubatch 108) | -4374.589 (ubatch 510) | no |
| 3 | 695 | 177.873 (ubatch 119) | -6576.733 (ubatch 500) | no |
| 4 | 694 | 40.825 (ubatch 119) | -10565.045 (ubatch 275) | no |

- shift 1 (697 pairs): card 0's evaluations before the trace's first ubatch are the load's warm-up; a trace ubatch past the profile's last row is one the profile did not write before the exit
- card 0 bounds the offset (trace ms = profile ms + offset) to 446.156 … 449.572 ms, width 3.416 ms: from below by ubatch 16's inputs set, from above by ubatch 167's graphs queued
- the most evaluations inside a window, 9580, at offsets 446.546 … 446.681 ms (28 grid points of 0.005 ms, of 684); the offset: their middle, 446.613

| card | in a main ubatch | in a draft ubatch | in none |
|---|---:|---:|---:|
| 0 | 697 | 0 | 0 |
| 1 | 697 | 0 | 0 |
| 2 | 697 | 0 | 0 |
| 3 | 697 | 0 | 0 |
| 4 | 697 | 0 | 0 |
| 5 | 697 | 0 | 0 |
| 6 | 697 | 0 | 0 |
| 7 | 697 | 0 | 0 |
| 8 | 698 | 3306 | 0 |

t0 = 7184470.906
