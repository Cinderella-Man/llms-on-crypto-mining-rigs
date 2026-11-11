| load | steps | yields a step (steps) | wakes a step (steps) | `s.drf → s.drf.in` | `s.drf.out → s.drf.end` | `s.dec.y → s.dec.in` | `s.dec.out → s.dec.end` | `s.mpd → s.cu.in` | `s.cu.out → s.cu.end` | the six, median of sums |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| off | 406 | 3: 406 | 3: 406 | 0.0020 | 0.0160 | 0.0020 | 0.0150 | 0.0020 | 0.0150 | 0.0520 |
| y1 | 406 | 1: 406 | 1: 406 | 0.0000 | 0.0000 | 0.0020 | 0.0140 | 0.0000 | 0.0000 | 0.0160 |
| y2 | 406 | 1: 406 | 0: 406 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0010 |
| boff | 406 | 3: 406 | 3: 406 | 0.0020 | 0.0140 | 0.0010 | 0.0130 | 0.0010 | 0.0130 | 0.0435 |
| by1 | 406 | 1: 406 | 1: 406 | 0.0000 | 0.0000 | 0.0020 | 0.0140 | 0.0000 | 0.0000 | 0.0160 |
| by2 | 406 | 1: 406 | 0: 406 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| base | 406 | — | — | 0.0010 | 0.0140 | 0.0010 | 0.0130 | 0.0010 | 0.0130 | 0.0420 |
| bbase | 406 | — | — | 0.0010 | 0.0140 | 0.0010 | 0.0140 | 0.0010 | 0.0130 | 0.0450 |

ms on the trace's steady clock; generation steps are the steps that drafted (n.drf > 0); a stage the build does not run (no yield there) still has both marks, back to back
