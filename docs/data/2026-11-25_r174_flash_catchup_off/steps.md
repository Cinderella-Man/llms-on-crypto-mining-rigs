| Pass type | Depth | `c` loads, ms (steps; accepted of drafted) | `off` loads | Arm − control | Control spread | Class (`ctl`) | Larger spread | Class (`lane`) |
|---|---:|---|---|---:|---:|---|---:|---|
| greedy, pass 1 | 0 | ca 105.37 (109; 147/321); cb 102.70 (109; 147/321) | offa 102.44 (109; 147/321); offb 102.42 (109; 147/321) | -1.60 | 2.67 | negligible | 2.67 | negligible |
| greedy, pass 1 | 4,096 | ca 114.35 (79; 177/231); cb 117.17 (79; 177/231) | offa 113.11 (79; 177/231); offb 113.02 (79; 177/231) | -2.69 | 2.82 | negligible | 2.82 | negligible |
| greedy, pass 1 | 8,192 | ca 120.47 (83; 173/245); cb 120.22 (83; 173/245) | offa 120.58 (83; 173/245); offb 120.71 (83; 173/245) | +0.30 | 0.25 | loss | 0.25 | loss |
| greedy, pass 1 | all | ca 112.61 (271; 497/797); cb 112.28 (271; 497/797) | offa 111.11 (271; 497/797); offb 111.11 (271; 497/797) | -1.34 | 0.33 | gain | 0.33 | gain |
| greedy, pass 2 | 0 | ca 98.08 (108; 148/318); cb 98.14 (108; 148/318) | offa 98.11 (108; 148/318); offb 98.12 (108; 148/318) | +0.00 | 0.07 | negligible | 0.07 | negligible |
| greedy, pass 2 | 4,096 | ca 108.70 (79; 177/231); cb 108.76 (79; 177/231) | offa 108.77 (79; 177/231); offb 108.91 (79; 177/231) | +0.11 | 0.06 | loss | 0.14 | negligible |
| greedy, pass 2 | 8,192 | ca 113.20 (78; 178/230); cb 113.04 (78; 178/230) | offa 113.27 (78; 178/230); offb 113.32 (78; 178/230) | +0.18 | 0.16 | loss | 0.16 | loss |
| greedy, pass 2 | all | ca 105.69 (265; 503/779); cb 105.69 (265; 503/779) | offa 105.75 (265; 503/779); offb 105.81 (265; 503/779) | +0.09 | 0.00 | loss | 0.06 | loss |
| sampled | 0 | ca 100.75 (97; 159/285); cb 100.66 (97; 159/285) | offa 100.91 (97; 159/285); offb 100.69 (97; 159/285) | +0.09 | 0.09 | loss | 0.22 | negligible |
| sampled | 4,096 | ca 111.84 (79; 177/232); cb 111.82 (79; 177/232) | offa 123.62 (79; 177/232); offb 111.90 (79; 177/232) | +5.93 | 0.02 | loss | 11.72 | negligible |
| sampled | 8,192 | ca 113.74 (75; 181/222); cb 114.14 (75; 181/222) | offa 124.97 (75; 181/222); offb 114.51 (75; 181/222) | +5.80 | 0.40 | loss | 10.46 | negligible |
| sampled | all | ca 108.12 (251; 517/739); cb 108.20 (251; 517/739) | offa 115.25 (251; 517/739); offb 108.35 (251; 517/739) | +3.63 | 0.08 | loss | 6.90 | negligible |

| Load | Arm | Full-head rounds after a miss (set-up line) | Drafts with the shortlist | With the full head | Target tokens outside it | Ids in it at the end |
|---|---|---:|---:|---:|---:|---:|
| ca | c | 2 | 736 | 52 | 26 | 33028 |
| cb | c | 2 | 736 | 52 | 26 | 33028 |
| offa | off | 2 | 736 | 52 | 26 | 33028 |
| offb | off | 2 | 736 | 52 | 26 | 33028 |
