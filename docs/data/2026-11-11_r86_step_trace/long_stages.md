# rig/out/trace/step_trace.jsonl: intervals over 20.0 ms

| request | step in the request | line | the line, ms | gap's host part, ms | v.ub.in → card 0's entry, ms | other intervals over 20.0 ms |
|---:|---:|---:|---:|---:|---:|---|
| 3 | 1 | 3 | 430.2 | 281.78 | 9.92 | v.sync.end → v.dec.out 269.7 |
| 7 | 1 | 92 | 1264.9 | 12.21 | 112.66 | — |
| 7 | 2 | 93 | 239.7 | 96.95 | 3.39 | v.ub.reuse → v.ub.in 96.0 |
| 7 | 3 | 94 | 243.4 | 1.75 | 111.00 | — |
| 7 | 4 | 95 | 246.8 | 1.59 | 114.26 | — |
| 7 | 5 | 96 | 247.7 | 1.73 | 115.06 | — |
| 7 | 6 | 97 | 248.9 | 1.81 | 117.30 | — |
| 7 | 7 | 98 | 292.9 | 1.74 | 161.01 | — |

| stage | lines |
|---|---:|
| v.ub.in → card's graph_compute entry | 6 |
| v.sync.end → v.dec.out | 1 |
| v.ub.reuse → v.ub.in | 1 |
