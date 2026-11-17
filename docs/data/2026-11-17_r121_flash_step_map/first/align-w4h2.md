# <OUT>/st-w4h2.jsonl beside <OUT>/hp-w4h2.csv

- main context's ubatches in the trace: 819; card 0's evaluations in the profile: 819; the draft context's ubatches in the trace: 3126

| shift (profile index - trace index) | pairs | lower bound, ms | upper bound, ms | fits |
|---:|---:|---:|---:|---|
| -4 | 815 | 26707.423 (ubatch 4) | 939.880 (ubatch 152) | no |
| -3 | 816 | 26590.915 (ubatch 3) | 840.498 (ubatch 151) | no |
| -2 | 817 | 26340.899 (ubatch 2) | 741.342 (ubatch 150) | no |
| -1 | 818 | 25595.024 (ubatch 1) | 641.797 (ubatch 199) | no |
| 0 | 819 | 23412.754 (ubatch 0) | 542.875 (ubatch 199) | no |
| 1 | 818 | 443.792 (ubatch 506) | 446.933 (ubatch 98) | yes |
| 2 | 817 | 347.671 (ubatch 198) | -4362.668 (ubatch 600) | no |
| 3 | 816 | 248.729 (ubatch 197) | -6636.890 (ubatch 590) | no |
| 4 | 815 | 149.255 (ubatch 147) | -10412.465 (ubatch 345) | no |

- shift 1 (818 pairs): card 0's evaluations before the trace's first ubatch are the load's warm-up; a trace ubatch past the profile's last row is one the profile did not write before the exit
- card 0 bounds the offset (trace ms = profile ms + offset) to 443.792 … 446.933 ms, width 3.141 ms: from below by ubatch 506's inputs set, from above by ubatch 98's graphs queued
- the most evaluations inside a window, 10488, at offsets 444.012 … 444.142 ms (27 grid points of 0.005 ms, of 629); the offset: their middle, 444.077

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

t0 = 9176652.450
