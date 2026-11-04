3214 evaluations, 190 main passes on devices 0,1,2,3,4,5,6,7,8, 186 steps after skipping 3
draft passes a step (median): 4; evaluations a pass (median): 2

| pass | part | median, ms | p10 | p90 |
|---|---|---:|---:|---:|
| 0 | lead_ms | 1.56 | 1.55 | 1.58 |
| 0 | first_ms | 0.03 | 0.03 | 0.03 |
| 0 | gpu_ms | 3.63 | 3.54 | 4.66 |
| 0 | wait_ms | 1.77 | 1.60 | 1.91 |
| 0 | idle_ms | -1.85 | -2.78 | -1.54 |
| 1 | lead_ms | 2.42 | 2.11 | 3.37 |
| 1 | first_ms | 0.02 | 0.02 | 0.03 |
| 1 | gpu_ms | 7.06 | 7.03 | 9.25 |
| 1 | wait_ms | 7.48 | 7.41 | 9.67 |
| 1 | idle_ms | 0.55 | 0.51 | 0.58 |
| 2 | lead_ms | 0.60 | 0.59 | 0.62 |
| 2 | first_ms | 0.02 | 0.01 | 0.02 |
| 2 | gpu_ms | 7.06 | 7.01 | 9.22 |
| 2 | wait_ms | 7.49 | 7.42 | 9.65 |
| 2 | idle_ms | 0.51 | 0.48 | 0.55 |
| 3 | lead_ms | 0.59 | 0.58 | 0.61 |
| 3 | first_ms | 0.02 | 0.01 | 0.02 |
| 3 | gpu_ms | 7.06 | 7.01 | 9.22 |
| 3 | wait_ms | 14.50 | 11.61 | 16.59 |
| 3 | idle_ms | 7.03 | 2.51 | 8.51 |

| per step | median, ms | p10 | p90 | total over the steps |
|---|---:|---:|---:|---:|
| step_ms | 179.24 | 175.68 | 190.17 | 34259 |
| draft_host_ms | 22.23 | 22.02 | 27.72 | 4535 |
| draft_gpu_ms | 24.77 | 24.62 | 32.35 | 5155 |
| lead_sum_ms | 5.18 | 4.87 | 6.15 | 1007 |
| lead_sum_drafting_ms | 3.61 | 3.31 | 4.58 | 717 |
| idle_sum_drafting_ms | 1.06 | 1.00 | 1.12 | 203 |
