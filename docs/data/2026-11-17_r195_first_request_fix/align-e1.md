# <OUT>/st-e1.jsonl beside <OUT>/hp-e1.csv

- main context's ubatches in the trace: 495; card 0's evaluations in the profile: 495; the draft context's ubatches in the trace: 1814

| shift (profile index - trace index) | pairs | lower bound, ms | upper bound, ms | fits |
|---:|---:|---:|---:|---|
| -4 | 491 | 31863.318 (ubatch 4) | 1022.789 (ubatch 15) | no |
| -3 | 492 | 31747.682 (ubatch 3) | 919.402 (ubatch 15) | no |
| -2 | 493 | 31502.315 (ubatch 2) | 818.927 (ubatch 15) | no |
| -1 | 494 | 30714.116 (ubatch 1) | 717.213 (ubatch 8) | no |
| 0 | 495 | 28476.649 (ubatch 0) | 615.608 (ubatch 15) | no |
| 1 | 494 | 512.113 (ubatch 337) | 515.460 (ubatch 471) | yes |
| 2 | 493 | 411.654 (ubatch 14) | -4811.287 (ubatch 354) | no |
| 3 | 492 | 310.087 (ubatch 6) | -10313.219 (ubatch 26) | no |
| 4 | 491 | 208.361 (ubatch 12) | -11836.812 (ubatch 26) | no |

- shift 1 (494 pairs): card 0's evaluations before the trace's first ubatch are the load's warm-up; a trace ubatch past the profile's last row is one the profile did not write before the exit
- card 0 bounds the offset (trace ms = profile ms + offset) to 512.113 … 515.460 ms, width 3.347 ms: from below by ubatch 337's inputs set, from above by ubatch 471's graphs queued
- the most evaluations inside a window, 6260, at offsets 512.328 … 512.463 ms (28 grid points of 0.005 ms, of 670); the offset: their middle, 512.395

| card | in a main ubatch | in a draft ubatch | in none |
|---|---:|---:|---:|
| 0 | 494 | 0 | 0 |
| 1 | 494 | 0 | 0 |
| 2 | 494 | 0 | 0 |
| 3 | 494 | 0 | 0 |
| 4 | 494 | 0 | 0 |
| 5 | 494 | 0 | 0 |
| 6 | 494 | 0 | 0 |
| 7 | 494 | 0 | 0 |
| 8 | 495 | 1813 | 0 |

t0 = 19454154.837
