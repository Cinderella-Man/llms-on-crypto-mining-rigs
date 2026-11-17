# <OUT>/st-w4h1.jsonl beside <OUT>/hp-w4h1.csv

- main context's ubatches in the trace: 408; card 0's evaluations in the profile: 408; the draft context's ubatches in the trace: 1500

| shift (profile index - trace index) | pairs | lower bound, ms | upper bound, ms | fits |
|---:|---:|---:|---:|---|
| -4 | 404 | 23202.908 (ubatch 240) | 1760.636 (ubatch 205) | no |
| -3 | 405 | 20258.794 (ubatch 239) | 1582.858 (ubatch 205) | no |
| -2 | 406 | 17425.539 (ubatch 240) | 1404.318 (ubatch 205) | no |
| -1 | 407 | 14481.425 (ubatch 239) | 1225.029 (ubatch 205) | no |
| 0 | 408 | 9805.793 (ubatch 239) | 1038.746 (ubatch 205) | no |
| 1 | 407 | 909.063 (ubatch 205) | 911.495 (ubatch 148) | yes |
| 2 | 406 | 791.482 (ubatch 204) | -7103.356 (ubatch 238) | no |
| 3 | 405 | 606.789 (ubatch 203) | -10032.675 (ubatch 238) | no |
| 4 | 404 | 427.537 (ubatch 202) | -12738.232 (ubatch 236) | no |

- shift 1 (407 pairs): card 0's evaluations before the trace's first ubatch are the load's warm-up; a trace ubatch past the profile's last row is one the profile did not write before the exit
- card 0 bounds the offset (trace ms = profile ms + offset) to 909.063 … 911.495 ms, width 2.432 ms: from below by ubatch 205's inputs set, from above by ubatch 148's graphs queued
- the most evaluations inside a window, 5163, at offsets 909.068 … 909.288 ms (45 grid points of 0.005 ms, of 487); the offset: their middle, 909.178

| card | in a main ubatch | in a draft ubatch | in none |
|---|---:|---:|---:|
| 0 | 407 | 0 | 0 |
| 1 | 407 | 0 | 0 |
| 2 | 407 | 0 | 0 |
| 3 | 407 | 0 | 0 |
| 4 | 407 | 0 | 0 |
| 5 | 407 | 0 | 0 |
| 6 | 407 | 0 | 0 |
| 7 | 407 | 0 | 0 |
| 8 | 408 | 1499 | 0 |

t0 = 26930528.474
