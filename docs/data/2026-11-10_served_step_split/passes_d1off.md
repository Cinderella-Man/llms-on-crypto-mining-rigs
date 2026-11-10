5232 evaluations, 310 main passes on devices 0,1,2,3,4,5,6,7,8, 303 steps after skipping 3
draft passes a step (median): 4; evaluations a pass (median): 2

| pass | part | median, ms | p10 | p90 |
|---|---|---:|---:|---:|
| 0 | lead_ms | 1.45 | 1.43 | 1.54 |
| 0 | first_ms | 0.03 | 0.02 | 0.04 |
| 0 | gpu_ms | 3.72 | 3.63 | 4.33 |
| 0 | wait_ms | 3.88 | 3.78 | 4.49 |
| 0 | idle_ms | 0.29 | 0.27 | 0.30 |
| 1 | lead_ms | 0.56 | 0.55 | 0.64 |
| 1 | first_ms | 0.03 | 0.02 | 0.03 |
| 1 | gpu_ms | 2.69 | 2.64 | 6.31 |
| 1 | wait_ms | 3.06 | 3.00 | 6.55 |
| 1 | idle_ms | 0.51 | 0.50 | 0.54 |
| 2 | lead_ms | 0.56 | 0.55 | 0.63 |
| 2 | first_ms | 0.02 | 0.02 | 0.02 |
| 2 | gpu_ms | 2.65 | 2.59 | 6.28 |
| 2 | wait_ms | 3.07 | 2.98 | 6.67 |
| 2 | idle_ms | 0.50 | 0.47 | 0.55 |
| 3 | lead_ms | 0.57 | 0.55 | 0.64 |
| 3 | first_ms | 0.02 | 0.01 | 0.02 |
| 3 | gpu_ms | 2.65 | 2.59 | 6.27 |
| 3 | wait_ms | 10.34 | 7.94 | 13.60 |
| 3 | idle_ms | 7.55 | 5.21 | 8.57 |

| per step | median, ms | p10 | p90 | total over the steps |
|---|---:|---:|---:|---:|
| step_ms | 132.07 | 128.55 | 144.08 | 66260 |
| draft_host_ms | 13.58 | 13.34 | 20.76 | 5985 |
| draft_gpu_ms | 11.69 | 11.46 | 22.61 | 5453 |
| lead_sum_ms | 3.14 | 3.10 | 3.45 | 1319 |
| lead_sum_drafting_ms | 1.69 | 1.66 | 1.91 | 818 |
| idle_sum_drafting_ms | 1.02 | 0.97 | 1.09 | 204 |
