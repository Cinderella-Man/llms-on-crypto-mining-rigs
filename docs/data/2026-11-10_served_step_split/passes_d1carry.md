3725 evaluations, 221 main passes on devices 0,1,2,3,4,5,6,7,8, 214 steps after skipping 3
draft passes a step (median): 4; evaluations a pass (median): 2

| pass | part | median, ms | p10 | p90 |
|---|---|---:|---:|---:|
| 0 | lead_ms | 1.44 | 1.43 | 1.51 |
| 0 | first_ms | 0.03 | 0.03 | 0.04 |
| 0 | gpu_ms | 3.68 | 3.64 | 4.19 |
| 0 | wait_ms | 3.83 | 3.78 | 4.42 |
| 0 | idle_ms | 0.28 | 0.28 | 0.35 |
| 1 | lead_ms | 0.57 | 0.57 | 0.61 |
| 1 | first_ms | 0.03 | 0.02 | 0.03 |
| 1 | gpu_ms | 2.69 | 2.62 | 6.79 |
| 1 | wait_ms | 3.46 | 3.38 | 6.94 |
| 1 | idle_ms | 0.92 | 0.90 | 0.98 |
| 2 | lead_ms | 0.35 | 0.34 | 0.38 |
| 2 | first_ms | 0.02 | 0.02 | 0.03 |
| 2 | gpu_ms | 2.70 | 2.66 | 6.34 |
| 2 | wait_ms | 2.85 | 2.78 | 6.33 |
| 2 | idle_ms | 0.29 | 0.28 | 0.33 |
| 3 | lead_ms | 0.35 | 0.34 | 0.39 |
| 3 | first_ms | 0.02 | 0.02 | 0.02 |
| 3 | gpu_ms | 2.67 | 2.62 | 6.33 |
| 3 | wait_ms | 9.73 | 4.83 | 13.56 |
| 3 | idle_ms | 6.97 | 2.19 | 8.11 |

| per step | median, ms | p10 | p90 | total over the steps |
|---|---:|---:|---:|---:|
| step_ms | 130.30 | 126.61 | 143.33 | 51474 |
| draft_host_ms | 13.40 | 13.23 | 22.13 | 4705 |
| draft_gpu_ms | 11.73 | 11.58 | 23.98 | 4263 |
| lead_sum_ms | 2.73 | 2.70 | 2.88 | 933 |
| lead_sum_drafting_ms | 1.28 | 1.26 | 1.35 | 567 |
| idle_sum_drafting_ms | 1.22 | 1.19 | 1.28 | 159 |
