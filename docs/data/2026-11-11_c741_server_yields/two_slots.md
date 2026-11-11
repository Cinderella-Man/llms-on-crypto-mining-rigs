| load | B came during A | B posted in step | its prompt in step | steps between | reads by the client | failed | reads posted in a step (`q.read`) | of them answered in a yield of that step | left to the step's end | answered in a yield, posted before the step | step ends between a read's sending and its answer, client side (reads) | read ms, median | read ms, max | step ms, median | step ms, max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| p2base | yes | — | — | — | 228 | 0 | 0 | 0 | 0 | 0 | 0: 214, 1: 14 | 0.3655 | 25.7990 | 3.3720 | 75.4780 |
| p2off | yes | 19 | 20 | 1 | 234 | 0 | 229 | 225 | 4 | 0 | 0: 215, 1: 19 | 0.3280 | 20.0500 | 3.4415 | 83.4620 |
| p2y1 | yes | 19 | 20 | 1 | 185 | 0 | 181 | 168 | 13 | 0 | 0: 166, 1: 19 | 0.8370 | 28.8570 | 3.2115 | 75.4070 |
| p2y2 | yes | 19 | 20 | 1 | 191 | 0 | 184 | 174 | 10 | 0 | 0: 147, 1: 44 | 0.5920 | 28.4770 | 3.3870 | 80.8460 |

Steps are the trace's lines, counted from 0; a step ends at its s.upd.end mark. The server's columns come from its marks (a build without them shows 0); the loop takes every task left in the queue when a step ends. Reads are the client's /slots and /metrics GETs while A or B ran, their times on the client (the HTTP threads' and the client's own delays included); ms on the steady clock
