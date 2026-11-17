# <OUT>/st-w1h2.jsonl beside <OUT>/hp-w1h2.csv

- main context's ubatches in the trace: 2366; card 0's evaluations in the profile: 2367; the draft context's ubatches in the trace: 0

| shift (profile index - trace index) | pairs | lower bound, ms | upper bound, ms | fits |
|---:|---:|---:|---:|---|
| -4 | 2362 | 21803.526 (ubatch 811) | 727.248 (ubatch 311) | no |
| -3 | 2363 | 19682.090 (ubatch 1588) | 676.241 (ubatch 356) | no |
| -2 | 2364 | 16078.301 (ubatch 811) | 625.263 (ubatch 355) | no |
| -1 | 2365 | 13337.347 (ubatch 1588) | 574.249 (ubatch 311) | no |
| 0 | 2366 | 10178.737 (ubatch 811) | 523.191 (ubatch 451) | no |
| 1 | 2366 | 6977.396 (ubatch 1588) | 471.910 (ubatch 229) | no |
| 2 | 2365 | 418.257 (ubatch 1248) | 421.302 (ubatch 387) | yes |
| 3 | 2364 | 367.356 (ubatch 412) | -4004.047 (ubatch 801) | no |
| 4 | 2363 | 316.377 (ubatch 449) | -6290.989 (ubatch 1586) | no |

- shift 2 (2365 pairs): card 0's evaluations before the trace's first ubatch are the load's warm-up; a trace ubatch past the profile's last row is one the profile did not write before the exit
- card 0 bounds the offset (trace ms = profile ms + offset) to 418.257 … 421.302 ms, width 3.045 ms: from below by ubatch 1248's inputs set, from above by ubatch 387's graphs queued
- the most evaluations inside a window, 21285, at offsets 418.257 … 418.587 ms (67 grid points of 0.005 ms, of 609); the offset: their middle, 418.422

| card | in a main ubatch | in a draft ubatch | in none |
|---|---:|---:|---:|
| 0 | 2365 | 0 | 0 |
| 1 | 2365 | 0 | 0 |
| 2 | 2365 | 0 | 0 |
| 3 | 2365 | 0 | 0 |
| 4 | 2365 | 0 | 0 |
| 5 | 2365 | 0 | 0 |
| 6 | 2365 | 0 | 0 |
| 7 | 2365 | 0 | 0 |
| 8 | 2365 | 0 | 0 |

t0 = 11094878.716
