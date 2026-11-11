# rig/out/trace/step_trace.jsonl beside rig/out/trace/host_profile.csv

- main context's ubatches in the trace: 189; card 0's evaluations in the profile: 189; the draft context's ubatches in the trace: 742

| shift (profile index - trace index) | pairs | lower bound, ms | upper bound, ms | fits |
|---:|---:|---:|---:|---|
| -4 | 185 | 37337.861 (ubatch 4) | 1273.354 (ubatch 9) | no |
| -3 | 186 | 37200.545 (ubatch 3) | 1144.797 (ubatch 11) | no |
| -2 | 187 | 35470.693 (ubatch 2) | 1013.780 (ubatch 10) | no |
| -1 | 188 | 33541.957 (ubatch 1) | 885.267 (ubatch 9) | no |
| 0 | 189 | 29328.251 (ubatch 0) | 758.169 (ubatch 9) | no |
| 1 | 188 | 630.913 (ubatch 44) | 635.237 (ubatch 37) | yes |
| 2 | 187 | 504.621 (ubatch 8) | -5887.101 (ubatch 93) | no |
| 3 | 186 | 376.312 (ubatch 8) | -8419.845 (ubatch 88) | no |
| 4 | 185 | 245.158 (ubatch 8) | -14413.491 (ubatch 91) | no |

- shift 1 (188 pairs): card 0's evaluations before the trace's first ubatch are the load's warm-up; a trace ubatch past the profile's last row is one the profile did not write before the exit
- card 0 bounds the offset (trace ms = profile ms + offset) to 630.913 … 635.237 ms, width 4.324 ms: from below by ubatch 44's inputs set, from above by ubatch 37's graphs queued
- the most evaluations inside a window, 3176, at offsets 631.693 … 631.723 ms (7 grid points of 0.005 ms, of 865); the offset: their middle, 631.708

| card | in a main ubatch | in a draft ubatch | in none |
|---|---:|---:|---:|
| 0 | 188 | 0 | 0 |
| 1 | 188 | 0 | 0 |
| 2 | 188 | 0 | 0 |
| 3 | 188 | 0 | 0 |
| 4 | 189 | 741 | 0 |
| 5 | 188 | 0 | 0 |
| 6 | 188 | 0 | 0 |
| 7 | 188 | 0 | 0 |
| 8 | 189 | 741 | 0 |

t0 = 4755288.911
