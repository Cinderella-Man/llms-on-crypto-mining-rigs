# <OUT>/st-w1h1.jsonl beside <OUT>/hp-w1h1.csv

- main context's ubatches in the trace: 2366; card 0's evaluations in the profile: 2367; the draft context's ubatches in the trace: 0

| shift (profile index - trace index) | pairs | lower bound, ms | upper bound, ms | fits |
|---:|---:|---:|---:|---|
| -4 | 2362 | 21648.278 (ubatch 811) | 773.566 (ubatch 423) | no |
| -3 | 2363 | 19491.223 (ubatch 1588) | 722.525 (ubatch 421) | no |
| -2 | 2364 | 15991.077 (ubatch 811) | 671.527 (ubatch 193) | no |
| -1 | 2365 | 13327.938 (ubatch 1588) | 620.470 (ubatch 299) | no |
| 0 | 2366 | 10205.962 (ubatch 811) | 569.432 (ubatch 294) | no |
| 1 | 2366 | 6986.275 (ubatch 1588) | 518.083 (ubatch 229) | no |
| 2 | 2365 | 464.486 (ubatch 264) | 467.499 (ubatch 315) | yes |
| 3 | 2364 | 413.646 (ubatch 297) | -3875.664 (ubatch 545) | no |
| 4 | 2363 | 362.628 (ubatch 296) | -6164.478 (ubatch 1586) | no |

- shift 2 (2365 pairs): card 0's evaluations before the trace's first ubatch are the load's warm-up; a trace ubatch past the profile's last row is one the profile did not write before the exit
- card 0 bounds the offset (trace ms = profile ms + offset) to 464.486 … 467.499 ms, width 3.013 ms: from below by ubatch 264's inputs set, from above by ubatch 315's graphs queued
- the most evaluations inside a window, 21285, at offsets 464.486 … 464.811 ms (66 grid points of 0.005 ms, of 603); the offset: their middle, 464.649

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

t0 = 8163671.328
