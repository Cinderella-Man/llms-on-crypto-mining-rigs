2756 evaluations, 164 main passes on devices 0,1,2,3,4,5,6,7,8, 157 steps after skipping 3
draft passes a step (median): 4; evaluations a pass (median): 2

| pass | part | median, ms | p10 | p90 |
|---|---|---:|---:|---:|
| 0 | lead_ms | 1.44 | 1.43 | 1.53 |
| 0 | first_ms | 0.03 | 0.03 | 0.03 |
| 0 | gpu_ms | 3.65 | 3.60 | 4.26 |
| 0 | wait_ms | 3.82 | 3.77 | 4.43 |
| 0 | idle_ms | 0.30 | 0.30 | 0.33 |
| 1 | lead_ms | 0.56 | 0.55 | 0.63 |
| 1 | first_ms | 0.03 | 0.02 | 0.03 |
| 1 | gpu_ms | 2.65 | 2.62 | 6.31 |
| 1 | wait_ms | 3.01 | 2.96 | 6.50 |
| 1 | idle_ms | 0.51 | 0.49 | 0.53 |
| 2 | lead_ms | 0.55 | 0.55 | 0.62 |
| 2 | first_ms | 0.02 | 0.02 | 0.02 |
| 2 | gpu_ms | 2.62 | 2.56 | 6.24 |
| 2 | wait_ms | 3.01 | 2.95 | 6.63 |
| 2 | idle_ms | 0.48 | 0.46 | 0.54 |
| 3 | lead_ms | 0.56 | 0.55 | 0.63 |
| 3 | first_ms | 0.02 | 0.01 | 0.02 |
| 3 | gpu_ms | 2.61 | 2.56 | 6.24 |
| 3 | wait_ms | 10.09 | 5.40 | 13.72 |
| 3 | idle_ms | 7.42 | 2.45 | 8.34 |

| per step | median, ms | p10 | p90 | total over the steps |
|---|---:|---:|---:|---:|
| step_ms | 130.62 | 127.39 | 142.96 | 43713 |
| draft_host_ms | 13.42 | 13.27 | 20.69 | 3816 |
| draft_gpu_ms | 11.52 | 11.38 | 22.61 | 3479 |
| lead_sum_ms | 3.12 | 3.09 | 3.41 | 837 |
| lead_sum_drafting_ms | 1.68 | 1.65 | 1.88 | 553 |
| idle_sum_drafting_ms | 0.99 | 0.96 | 1.06 | 77 |
