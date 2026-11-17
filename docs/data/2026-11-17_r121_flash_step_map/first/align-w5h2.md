# <OUT>/st-w5h2.jsonl beside <OUT>/hp-w5h2.csv

- main context's ubatches in the trace: 698; card 0's evaluations in the profile: 698; the draft context's ubatches in the trace: 3307

| shift (profile index - trace index) | pairs | lower bound, ms | upper bound, ms | fits |
|---:|---:|---:|---:|---|
| -4 | 694 | 26917.843 (ubatch 4) | 1125.950 (ubatch 121) | no |
| -3 | 695 | 26728.592 (ubatch 3) | 988.733 (ubatch 122) | no |
| -2 | 696 | 26376.356 (ubatch 2) | 851.367 (ubatch 121) | no |
| -1 | 697 | 25579.505 (ubatch 1) | 714.159 (ubatch 121) | no |
| 0 | 698 | 23400.951 (ubatch 0) | 578.410 (ubatch 129) | no |
| 1 | 697 | 442.455 (ubatch 421) | 445.729 (ubatch 77) | yes |
| 2 | 696 | 319.782 (ubatch 108) | -4372.119 (ubatch 284) | no |
| 3 | 695 | 173.849 (ubatch 119) | -6715.020 (ubatch 500) | no |
| 4 | 694 | 36.641 (ubatch 118) | -10864.478 (ubatch 275) | no |

- shift 1 (697 pairs): card 0's evaluations before the trace's first ubatch are the load's warm-up; a trace ubatch past the profile's last row is one the profile did not write before the exit
- card 0 bounds the offset (trace ms = profile ms + offset) to 442.455 … 445.729 ms, width 3.274 ms: from below by ubatch 421's inputs set, from above by ubatch 77's graphs queued
- the most evaluations inside a window, 9580, at offsets 442.715 … 442.855 ms (29 grid points of 0.005 ms, of 655); the offset: their middle, 442.785

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

t0 = 10118759.709
