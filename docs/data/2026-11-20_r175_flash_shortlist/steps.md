| Pass type | Depth | `c` loads, ms (steps; accepted of drafted) | `fb1` loads | Arm − control | Control spread | Class (`ctl`) | Larger spread | Class (`lane`) |
|---|---:|---|---|---:|---:|---|---:|---|
| greedy, pass 1 | 0 | ca 102.81 (109; 147/321); cb 94.53 (109; 147/321) | fb1a 94.17 (108; 148/318); fb1b 94.04 (108; 148/318) | -4.56 | 8.28 | negligible | 8.28 | negligible |
| greedy, pass 1 | 4,096 | ca 104.24 (79; 177/231); cb 103.81 (79; 177/231) | fb1a 103.61 (79; 177/231); fb1b 103.76 (79; 177/231) | -0.34 | 0.43 | negligible | 0.43 | negligible |
| greedy, pass 1 | 8,192 | ca 111.30 (83; 173/245); cb 111.20 (83; 173/245) | fb1a 109.82 (83; 173/245); fb1b 110.05 (83; 173/245) | -1.31 | 0.10 | gain | 0.22 | gain |
| greedy, pass 1 | all | ca 105.82 (271; 497/797); cb 102.34 (271; 497/797) | fb1a 101.74 (270; 498/794); fb1b 101.80 (270; 498/794) | -2.31 | 3.49 | negligible | 3.49 | negligible |
| greedy, pass 2 | 0 | ca 90.13 (108; 148/318); cb 90.17 (108; 148/318) | fb1a 90.26 (107; 149/315); fb1b 90.18 (107; 149/315) | +0.07 | 0.04 | loss | 0.09 | negligible |
| greedy, pass 2 | 4,096 | ca 99.52 (79; 177/231); cb 99.58 (79; 177/231) | fb1a 99.44 (79; 177/231); fb1b 99.67 (79; 177/231) | +0.01 | 0.05 | negligible | 0.23 | negligible |
| greedy, pass 2 | 8,192 | ca 103.82 (78; 178/230); cb 103.72 (78; 178/230) | fb1a 103.75 (78; 178/230); fb1b 103.69 (78; 178/230) | -0.05 | 0.10 | negligible | 0.10 | negligible |
| greedy, pass 2 | all | ca 96.96 (265; 503/779); cb 96.96 (265; 503/779) | fb1a 97.00 (264; 504/776); fb1b 97.01 (264; 504/776) | +0.04 | 0.00 | loss | 0.01 | loss |
| sampled | 0 | ca 92.92 (97; 159/285); cb 92.89 (97; 159/285) | fb1a 94.62 (111; 145/329); fb1b 94.69 (111; 145/329) | +1.75 | 0.02 | loss | 0.07 | loss |
| sampled | 4,096 | ca 114.12 (79; 177/232); cb 102.41 (79; 177/232) | fb1a 100.65 (77; 179/228); fb1b 100.86 (77; 179/228) | -7.50 | 11.72 | negligible | 11.72 | negligible |
| sampled | 8,192 | ca 112.93 (75; 181/222); cb 104.77 (75; 181/222) | fb1a 105.95 (75; 181/220); fb1b 105.75 (75; 181/220) | -3.00 | 8.15 | negligible | 8.15 | negligible |
| sampled | all | ca 105.57 (251; 517/739); cb 99.44 (251; 517/739) | fb1a 99.62 (263; 505/777); fb1b 99.65 (263; 505/777) | -2.87 | 6.13 | negligible | 6.13 | negligible |

| Load | Arm | Full-head rounds after a miss (set-up line) | Drafts with the shortlist | With the full head | Target tokens outside it | Ids in it at the end |
|---|---|---:|---:|---:|---:|---:|
| ca | c | 2 | 736 | 52 | 26 | 33028 |
| cb | c | 2 | 736 | 52 | 26 | 33028 |
| fb1a | fb1 | 1 | 768 | 31 | 31 | 33031 |
| fb1b | fb1 | 1 | 768 | 31 | 31 | 33031 |
