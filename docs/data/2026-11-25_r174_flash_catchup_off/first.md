## R174: the first request against its repeat, arms `c` (control) and `off`

Timed loads: ca, cb, offa, offb.

### Per load and depth, ms

| Depth | Load | Pass 1 prompt (tokens) | Pass 2 prompt (tokens) | Pass 1 generation (steps) | Pass 2 generation (steps) | Excess: generation | Excess: prompt | Excess: both | Step pass 1 | Step pass 2 | Prime prompt (tokens) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | ca | 1222.735 (64) | 223.000 (4) | 11484.789 (109) | 10592.418 (108) | 892.371 | 999.735 | 1892.106 | 105.37 | 98.08 | — |
| 0 | cb | 1232.335 (64) | 107.434 (4) | 11193.776 (109) | 10599.499 (108) | 594.277 | 1124.901 | 1719.178 | 102.70 | 98.14 | — |
| 0 | offa | 1235.375 (64) | 107.696 (4) | 11165.816 (109) | 10595.690 (108) | 570.126 | 1127.679 | 1697.805 | 102.44 | 98.11 | — |
| 0 | offb | 1240.751 (64) | 113.807 (4) | 11163.465 (109) | 10596.770 (108) | 566.695 | 1126.944 | 1693.639 | 102.42 | 98.12 | — |
| 4,096 | ca | 1472.889 (64) | 120.943 (4) | 9033.677 (79) | 8587.279 (79) | 446.398 | 1351.946 | 1798.344 | 114.35 | 108.70 | 21651.211 (4032) |
| 4,096 | cb | 1484.295 (64) | 121.277 (4) | 9256.210 (79) | 8592.377 (79) | 663.833 | 1363.018 | 2026.851 | 117.17 | 108.76 | 20410.629 (4032) |
| 4,096 | offa | 1470.296 (64) | 123.715 (4) | 8936.002 (79) | 8592.752 (79) | 343.250 | 1346.581 | 1689.831 | 113.11 | 108.77 | 20493.578 (4032) |
| 4,096 | offb | 1450.561 (64) | 124.226 (4) | 8928.596 (79) | 8604.069 (79) | 324.527 | 1326.335 | 1650.862 | 113.02 | 108.91 | 20279.292 (4032) |
| 8,192 | ca | 1321.207 (64) | 127.991 (4) | 9999.159 (83) | 8829.364 (78) | 1169.795 | 1193.216 | 2363.011 | 120.47 | 113.20 | 20809.368 (4612) |
| 8,192 | cb | 1348.065 (64) | 128.750 (4) | 9978.594 (83) | 8816.835 (78) | 1161.759 | 1219.315 | 2381.074 | 120.22 | 113.04 | 21687.654 (4612) |
| 8,192 | offa | 1327.354 (64) | 124.343 (4) | 10008.403 (83) | 8835.195 (78) | 1173.208 | 1203.011 | 2376.219 | 120.58 | 113.27 | 21357.590 (4612) |
| 8,192 | offb | 1347.504 (64) | 125.268 (4) | 10018.642 (83) | 8838.779 (78) | 1179.863 | 1222.236 | 2402.099 | 120.71 | 113.32 | 20886.250 (4612) |

### Classes of `off` against `c` (time: lower is a gain)

| Measure | Depth | `c` loads | `off` loads | Spread | `off` minus `c` | Class |
|---|---:|---|---|---:|---:|---|
| first request's excess in generation (pass 1 minus pass 2), ms | 0 | 892.371, 594.277 | 570.126, 566.695 | 298.094 | -174.914 | negligible |
| first request's excess in generation (pass 1 minus pass 2), ms | 4,096 | 446.398, 663.833 | 343.250, 324.527 | 217.435 | -221.227 | gain |
| first request's excess in generation (pass 1 minus pass 2), ms | 8,192 | 1169.795, 1161.759 | 1173.208, 1179.863 | 8.036 | +10.758 | loss |
| first request's excess in generation (pass 1 minus pass 2), ms | all | 2508.564, 2419.869 | 2086.584, 2071.085 | 88.695 | -385.382 | gain |
| first request's excess in prompt time (pass 1 minus pass 2), ms | 0 | 999.735, 1124.901 | 1127.679, 1126.944 | 125.166 | +64.994 | negligible |
| first request's excess in prompt time (pass 1 minus pass 2), ms | 4,096 | 1351.946, 1363.018 | 1346.581, 1326.335 | 20.246 | -21.024 | gain |
| first request's excess in prompt time (pass 1 minus pass 2), ms | 8,192 | 1193.216, 1219.315 | 1203.011, 1222.236 | 26.099 | +6.358 | negligible |
| first request's excess in prompt time (pass 1 minus pass 2), ms | all | 3544.897, 3707.234 | 3677.271, 3675.515 | 162.337 | +50.327 | negligible |
| first request's excess, generation and prompt together, ms | 0 | 1892.106, 1719.178 | 1697.805, 1693.639 | 172.928 | -109.920 | negligible |
| first request's excess, generation and prompt together, ms | 4,096 | 1798.344, 2026.851 | 1689.831, 1650.862 | 228.507 | -242.251 | gain |
| first request's excess, generation and prompt together, ms | 8,192 | 2363.011, 2381.074 | 2376.219, 2402.099 | 25.880 | +17.116 | negligible |
| first request's excess, generation and prompt together, ms | all | 6053.461, 6127.103 | 5763.855, 5746.600 | 73.642 | -335.054 | gain |
| first request's excess a step (pass 1 step minus pass 2 step), ms | 0 | 7.287, 4.552 | 4.330, 4.299 | 2.735 | -1.605 | negligible |
| first request's excess a step (pass 1 step minus pass 2 step), ms | 4,096 | 5.651, 8.403 | 4.345, 4.108 | 2.752 | -2.800 | gain |
| first request's excess a step (pass 1 step minus pass 2 step), ms | 8,192 | 7.275, 7.188 | 7.311, 7.389 | 0.087 | +0.119 | loss |
| first request's excess a step (pass 1 step minus pass 2 step), ms | all | 6.917, 6.589 | 5.358, 5.300 | 0.327 | -1.424 | gain |
| pass 1 generation, ms | 0 | 11484.789, 11193.776 | 11165.816, 11163.465 | 291.013 | -174.642 | negligible |
| pass 1 generation, ms | 4,096 | 9033.677, 9256.210 | 8936.002, 8928.596 | 222.533 | -212.645 | negligible |
| pass 1 generation, ms | 8,192 | 9999.159, 9978.594 | 10008.403, 10018.642 | 20.565 | +24.646 | loss |
| pass 1 generation, ms | all | 30517.625, 30428.580 | 30110.221, 30110.703 | 89.045 | -362.641 | gain |
| pass 2 generation, ms | 0 | 10592.418, 10599.499 | 10595.690, 10596.770 | 7.081 | +0.271 | negligible |
| pass 2 generation, ms | 4,096 | 8587.279, 8592.377 | 8592.752, 8604.069 | 11.317 | +8.582 | negligible |
| pass 2 generation, ms | 8,192 | 8829.364, 8816.835 | 8835.195, 8838.779 | 12.529 | +13.888 | loss |
| pass 2 generation, ms | all | 28009.061, 28008.711 | 28023.637, 28039.618 | 15.981 | +22.742 | loss |
| pass 1 prompt time, ms | 0 | 1222.735, 1232.335 | 1235.375, 1240.751 | 9.600 | +10.528 | loss |
| pass 1 prompt time, ms | 4,096 | 1472.889, 1484.295 | 1470.296, 1450.561 | 19.735 | -18.164 | negligible |
| pass 1 prompt time, ms | 8,192 | 1321.207, 1348.065 | 1327.354, 1347.504 | 26.858 | +2.793 | negligible |
| pass 1 prompt time, ms | all | 4016.831, 4064.695 | 4033.025, 4038.816 | 47.864 | -4.842 | negligible |
| pass 2 prompt time, ms | 0 | 223.000, 107.434 | 107.696, 113.807 | 115.566 | -54.465 | negligible |
| pass 2 prompt time, ms | 4,096 | 120.943, 121.277 | 123.715, 124.226 | 0.511 | +2.861 | loss |
| pass 2 prompt time, ms | 8,192 | 127.991, 128.750 | 124.343, 125.268 | 0.925 | -3.565 | gain |
| pass 2 prompt time, ms | all | 471.934, 357.461 | 355.754, 363.301 | 114.473 | -55.170 | negligible |
| pass 1 step, ms | 0 | 105.365, 102.695 | 102.439, 102.417 | 2.670 | -1.602 | negligible |
| pass 1 step, ms | 4,096 | 114.350, 117.167 | 113.114, 113.020 | 2.817 | -2.692 | negligible |
| pass 1 step, ms | 8,192 | 120.472, 120.224 | 120.583, 120.707 | 0.248 | +0.297 | loss |
| pass 1 step, ms | all | 112.611, 112.283 | 111.108, 111.110 | 0.329 | -1.338 | gain |
| pass 2 step, ms | 0 | 98.078, 98.144 | 98.108, 98.118 | 0.066 | +0.003 | negligible |
| pass 2 step, ms | 4,096 | 108.700, 108.764 | 108.769, 108.912 | 0.143 | +0.109 | negligible |
| pass 2 step, ms | 8,192 | 113.197, 113.036 | 113.272, 113.318 | 0.161 | +0.178 | loss |
| pass 2 step, ms | all | 105.695, 105.693 | 105.750, 105.810 | 0.060 | +0.086 | loss |
| prime's prompt time (the chunk on top of the previous depth's cache), ms | 4,096 | 21651.211, 20410.629 | 20493.578, 20279.292 | 1240.582 | -644.485 | negligible |
| prime's prompt time (the chunk on top of the previous depth's cache), ms | 8,192 | 20809.368, 21687.654 | 21357.590, 20886.250 | 878.286 | -126.591 | negligible |
| prime's prompt time (the chunk on top of the previous depth's cache), ms | all | 42460.579, 42098.283 | 41851.168, 41165.542 | 685.626 | -771.076 | gain |

### The greedy replies (token ids' md5 of each decode pass)

| Depth | ca p1 | ca p2 | cb p1 | cb p2 | offa p1 | offa p2 | offb p1 | offb p2 | All the same |
|---:|---|---|---|---|---|---|---|---|---|
| 0 | 2aadc9cf | 2aadc9cf | 2aadc9cf | 2aadc9cf | 2aadc9cf | 2aadc9cf | 2aadc9cf | 2aadc9cf | yes |
| 4,096 | 765babe2 | 765babe2 | 765babe2 | 765babe2 | 765babe2 | 765babe2 | 765babe2 | 765babe2 | yes |
| 8,192 | 8a589d39 | 8a589d39 | 8a589d39 | 8a589d39 | 8a589d39 | 8a589d39 | 8a589d39 | 8a589d39 | yes |

Every greedy reply the same on both passes of every timed load of both arms: yes.
