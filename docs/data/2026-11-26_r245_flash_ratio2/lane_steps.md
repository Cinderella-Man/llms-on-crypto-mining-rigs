# The step of every load, per depth and pass type, against the control (R245)

Made with R175's `run/steps.py` (commands in [README.md](README.md)); each arm against the control `c` (`_MIN_RATIO=0.5`, the v5 preset). Class (`ctl`): gain or loss beyond the control's own spread (its two loads); class (`lane`): beyond the larger of the two arms' spreads.

## `r42` (`_MIN_RATIO=0.42`: four-token verifies on the sparse kernel from a view of 3,584 cells) against the control `c`

| Pass type | Depth | `c` loads, ms (steps; accepted of drafted) | `r42` loads | Arm − control | Control spread | Class (`ctl`) | Larger spread | Class (`lane`) |
|---|---:|---|---|---:|---:|---|---:|---|
| greedy, pass 1 | 0 | ca 94.00 (109; 147/321); cb 95.10 (109; 147/321) | r42a 95.00 (109; 147/321); r42b 95.03 (109; 147/321) | +0.47 | 1.10 | negligible | 1.10 | negligible |
| greedy, pass 1 | 3,328 | ca 107.47 (76; 180/224); cb 103.60 (76; 180/224) | r42a 103.81 (76; 180/224); r42b 105.77 (76; 180/224) | -0.75 | 3.87 | negligible | 3.87 | negligible |
| greedy, pass 1 | 3,584 | ca 106.56 (88; 168/260); cb 105.47 (88; 168/260) | r42a 103.05 (76; 180/222); r42b 103.00 (76; 180/222) | -2.99 | 1.09 | gain | 1.09 | gain |
| greedy, pass 1 | 3,840 | ca 103.88 (72; 184/212); cb 103.73 (72; 184/212) | r42a 102.58 (72; 184/212); r42b 102.47 (72; 184/212) | -1.28 | 0.15 | gain | 0.15 | gain |
| greedy, pass 1 | all | ca 102.23 (345; 679/1017); cb 101.42 (345; 679/1017) | r42a 100.49 (333; 691/979); r42b 100.91 (333; 691/979) | -1.13 | 0.81 | gain | 0.81 | gain |
| greedy, pass 2 | 0 | ca 90.43 (108; 148/318); cb 90.60 (108; 148/318) | r42a 90.40 (108; 148/318); r42b 90.48 (108; 148/318) | -0.07 | 0.17 | negligible | 0.17 | negligible |
| greedy, pass 2 | 3,328 | ca 99.17 (75; 181/221); cb 99.02 (75; 181/221) | r42a 98.83 (75; 181/221); r42b 98.50 (75; 181/221) | -0.43 | 0.15 | gain | 0.33 | gain |
| greedy, pass 2 | 3,584 | ca 99.51 (87; 169/257); cb 99.50 (87; 169/257) | r42a 98.19 (74; 182/216); r42b 98.47 (74; 182/216) | -1.18 | 0.01 | gain | 0.28 | gain |
| greedy, pass 2 | 3,840 | ca 99.94 (72; 184/212); cb 100.02 (72; 184/212) | r42a 98.98 (72; 184/212); r42b 98.81 (72; 184/212) | -1.08 | 0.08 | gain | 0.17 | gain |
| greedy, pass 2 | all | ca 96.66 (342; 682/1008); cb 96.69 (342; 682/1008) | r42a 95.95 (329; 695/967); r42b 95.93 (329; 695/967) | -0.73 | 0.03 | gain | 0.03 | gain |
| sampled | 0 | ca 93.31 (97; 159/285); cb 93.24 (97; 159/285) | r42a 93.01 (97; 159/285); r42b 93.28 (97; 159/285) | -0.13 | 0.07 | gain | 0.27 | negligible |
| sampled | 3,328 | ca 100.30 (79; 177/233); cb 99.81 (79; 177/233) | r42a 101.03 (74; 182/216); r42b 114.38 (74; 182/216) | +7.65 | 0.49 | loss | 13.34 | negligible |
| sampled | 3,584 | ca 102.36 (80; 176/235); cb 101.94 (80; 176/235) | r42a 99.27 (71; 185/207); r42b 99.40 (71; 185/207) | -2.82 | 0.41 | gain | 0.41 | gain |
| sampled | 3,840 | ca 104.00 (76; 180/222); cb 102.35 (76; 180/222) | r42a 101.20 (77; 179/226); r42b 100.96 (77; 179/226) | -2.09 | 1.65 | gain | 1.65 | gain |
| sampled | all | ca 99.60 (332; 692/975); cb 98.99 (332; 692/975) | r42a 98.24 (319; 705/934); r42b 101.39 (319; 705/934) | +0.52 | 0.61 | negligible | 3.15 | negligible |

| Load | Arm | Full-head rounds after a miss (set-up line) | Drafts with the shortlist | With the full head | Target tokens outside it | Ids in it at the end |
|---|---|---:|---:|---:|---:|---:|
| ca | c | 2 | 972 | 46 | 23 | 32962 |
| cb | c | 2 | 972 | 46 | 23 | 32962 |
| r42a | r42 | 2 | 939 | 38 | 19 | 32961 |
| r42b | r42 | 2 | 939 | 38 | 19 | 32961 |

## `r45` (`_MIN_RATIO=0.45`: four-token verifies on the sparse kernel from a view of 3,840 cells) against the control `c`

| Pass type | Depth | `c` loads, ms (steps; accepted of drafted) | `r45` loads | Arm − control | Control spread | Class (`ctl`) | Larger spread | Class (`lane`) |
|---|---:|---|---|---:|---:|---|---:|---|
| greedy, pass 1 | 0 | ca 94.00 (109; 147/321); cb 95.10 (109; 147/321) | r45a 94.88 (109; 147/321); r45b 94.44 (109; 147/321) | +0.12 | 1.10 | negligible | 1.10 | negligible |
| greedy, pass 1 | 3,328 | ca 107.47 (76; 180/224); cb 103.60 (76; 180/224) | r45a 103.71 (76; 180/224); r45b 103.91 (76; 180/224) | -1.73 | 3.87 | negligible | 3.87 | negligible |
| greedy, pass 1 | 3,584 | ca 106.56 (88; 168/260); cb 105.47 (88; 168/260) | r45a 102.76 (82; 174/240); r45b 102.79 (82; 174/240) | -3.24 | 1.09 | gain | 1.09 | gain |
| greedy, pass 1 | 3,840 | ca 103.88 (72; 184/212); cb 103.73 (72; 184/212) | r45a 102.97 (72; 184/212); r45b 102.83 (72; 184/212) | -0.90 | 0.15 | gain | 0.15 | gain |
| greedy, pass 1 | all | ca 102.23 (345; 679/1017); cb 101.42 (345; 679/1017) | r45a 100.49 (339; 685/997); r45b 100.37 (339; 685/997) | -1.40 | 0.81 | gain | 0.81 | gain |
| greedy, pass 2 | 0 | ca 90.43 (108; 148/318); cb 90.60 (108; 148/318) | r45a 90.44 (108; 148/318); r45b 90.45 (108; 148/318) | -0.07 | 0.17 | negligible | 0.17 | negligible |
| greedy, pass 2 | 3,328 | ca 99.17 (75; 181/221); cb 99.02 (75; 181/221) | r45a 98.89 (75; 181/221); r45b 98.78 (75; 181/221) | -0.26 | 0.15 | gain | 0.15 | gain |
| greedy, pass 2 | 3,584 | ca 99.51 (87; 169/257); cb 99.50 (87; 169/257) | r45a 98.11 (81; 175/237); r45b 98.31 (81; 175/237) | -1.30 | 0.01 | gain | 0.20 | gain |
| greedy, pass 2 | 3,840 | ca 99.94 (72; 184/212); cb 100.02 (72; 184/212) | r45a 98.80 (72; 184/212); r45b 99.52 (72; 184/212) | -0.82 | 0.08 | gain | 0.72 | gain |
| greedy, pass 2 | all | ca 96.66 (342; 682/1008); cb 96.69 (342; 682/1008) | r45a 95.97 (336; 688/988); r45b 96.15 (336; 688/988) | -0.62 | 0.03 | gain | 0.18 | gain |
| sampled | 0 | ca 93.31 (97; 159/285); cb 93.24 (97; 159/285) | r45a 93.38 (97; 159/285); r45b 93.03 (97; 159/285) | -0.07 | 0.07 | gain | 0.35 | negligible |
| sampled | 3,328 | ca 100.30 (79; 177/233); cb 99.81 (79; 177/233) | r45a 99.65 (79; 177/233); r45b 99.78 (79; 177/233) | -0.34 | 0.49 | negligible | 0.49 | negligible |
| sampled | 3,584 | ca 102.36 (80; 176/235); cb 101.94 (80; 176/235) | r45a 99.12 (73; 183/213); r45b 99.83 (73; 183/213) | -2.68 | 0.41 | gain | 0.70 | gain |
| sampled | 3,840 | ca 104.00 (76; 180/222); cb 102.35 (76; 180/222) | r45a 101.33 (77; 179/226); r45b 101.51 (77; 179/226) | -1.76 | 1.65 | gain | 1.65 | gain |
| sampled | all | ca 99.60 (332; 692/975); cb 98.99 (332; 692/975) | r45a 98.06 (326; 698/957); r45b 98.19 (326; 698/957) | -1.17 | 0.61 | gain | 0.61 | gain |

| Load | Arm | Full-head rounds after a miss (set-up line) | Drafts with the shortlist | With the full head | Target tokens outside it | Ids in it at the end |
|---|---|---:|---:|---:|---:|---:|
| ca | c | 2 | 972 | 46 | 23 | 32962 |
| cb | c | 2 | 972 | 46 | 23 | 32962 |
| r45a | r45 | 2 | 962 | 36 | 18 | 32958 |
| r45b | r45 | 2 | 962 | 36 | 18 | 32958 |

## `r49` (`_MIN_RATIO=0.49`: four-token verifies on the sparse kernel from a view of 4,096 cells) against the control `c`

| Pass type | Depth | `c` loads, ms (steps; accepted of drafted) | `r49` loads | Arm − control | Control spread | Class (`ctl`) | Larger spread | Class (`lane`) |
|---|---:|---|---|---:|---:|---|---:|---|
| greedy, pass 1 | 0 | ca 94.00 (109; 147/321); cb 95.10 (109; 147/321) | r49a 95.46 (109; 147/321); r49b 95.03 (109; 147/321) | +0.70 | 1.10 | negligible | 1.10 | negligible |
| greedy, pass 1 | 3,328 | ca 107.47 (76; 180/224); cb 103.60 (76; 180/224) | r49a 103.56 (76; 180/224); r49b 103.79 (76; 180/224) | -1.87 | 3.87 | negligible | 3.87 | negligible |
| greedy, pass 1 | 3,584 | ca 106.56 (88; 168/260); cb 105.47 (88; 168/260) | r49a 105.39 (88; 168/260); r49b 105.73 (88; 168/260) | -0.45 | 1.09 | negligible | 1.09 | negligible |
| greedy, pass 1 | 3,840 | ca 103.88 (72; 184/212); cb 103.73 (72; 184/212) | r49a 103.00 (72; 184/212); r49b 102.33 (72; 184/212) | -1.14 | 0.15 | gain | 0.67 | gain |
| greedy, pass 1 | all | ca 102.23 (345; 679/1017); cb 101.42 (345; 679/1017) | r49a 101.35 (345; 679/1017); r49b 101.21 (345; 679/1017) | -0.54 | 0.81 | negligible | 0.81 | negligible |
| greedy, pass 2 | 0 | ca 90.43 (108; 148/318); cb 90.60 (108; 148/318) | r49a 90.62 (108; 148/318); r49b 90.42 (108; 148/318) | +0.01 | 0.17 | negligible | 0.20 | negligible |
| greedy, pass 2 | 3,328 | ca 99.17 (75; 181/221); cb 99.02 (75; 181/221) | r49a 99.14 (75; 181/221); r49b 98.92 (75; 181/221) | -0.06 | 0.15 | negligible | 0.21 | negligible |
| greedy, pass 2 | 3,584 | ca 99.51 (87; 169/257); cb 99.50 (87; 169/257) | r49a 100.65 (87; 169/257); r49b 99.54 (87; 169/257) | +0.59 | 0.01 | loss | 1.12 | negligible |
| greedy, pass 2 | 3,840 | ca 99.94 (72; 184/212); cb 100.02 (72; 184/212) | r49a 98.77 (72; 184/212); r49b 99.04 (72; 184/212) | -1.07 | 0.08 | gain | 0.27 | gain |
| greedy, pass 2 | all | ca 96.66 (342; 682/1008); cb 96.69 (342; 682/1008) | r49a 96.76 (342; 682/1008); r49b 96.42 (342; 682/1008) | -0.09 | 0.03 | gain | 0.33 | negligible |
| sampled | 0 | ca 93.31 (97; 159/285); cb 93.24 (97; 159/285) | r49a 93.25 (97; 159/285); r49b 102.49 (97; 159/285) | +4.59 | 0.07 | loss | 9.24 | negligible |
| sampled | 3,328 | ca 100.30 (79; 177/233); cb 99.81 (79; 177/233) | r49a 100.28 (81; 175/240); r49b 105.24 (79; 177/233) | +2.70 | 0.49 | loss | 4.96 | negligible |
| sampled | 3,584 | ca 102.36 (80; 176/235); cb 101.94 (80; 176/235) | r49a 101.95 (83; 173/244); r49b 101.91 (80; 176/235) | -0.22 | 0.41 | negligible | 0.41 | negligible |
| sampled | 3,840 | ca 104.00 (76; 180/222); cb 102.35 (76; 180/222) | r49a 101.24 (74; 182/216); r49b 101.18 (74; 182/216) | -1.97 | 1.65 | gain | 1.65 | gain |
| sampled | all | ca 99.60 (332; 692/975); cb 98.99 (332; 692/975) | r49a 98.87 (335; 689/985); r49b 102.72 (330; 694/969) | +1.50 | 0.61 | loss | 3.85 | negligible |

| Load | Arm | Full-head rounds after a miss (set-up line) | Drafts with the shortlist | With the full head | Target tokens outside it | Ids in it at the end |
|---|---|---:|---:|---:|---:|---:|
| ca | c | 2 | 972 | 46 | 23 | 32962 |
| cb | c | 2 | 972 | 46 | 23 | 32962 |
| r49a | r49 | 2 | 969 | 52 | 26 | 32966 |
| r49b | r49 | 2 | 968 | 48 | 24 | 32963 |
