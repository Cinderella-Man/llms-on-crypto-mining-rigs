| group | arm | verify steps | draft micro-batches | rebuilt | of up to 8 tokens: rebuilt / reused | rebuilt, no output / one output | rebuilt per step (up to 8 tokens) | why rebuilt |
|---|---|---:|---:|---:|---|---|---:|---|
| plain | base-plain | 202 | 819 | 415 | 411 / 404 | 208 / 203 | 2.035 | large:not_allocated 3, large:params 1, small:not_allocated 411 |
| plain | off-plain | 202 | 819 | 415 | 411 / 404 | 208 / 203 | 2.035 | large:not_allocated 3, large:params 1, small:not_allocated 411 |
| plain | on-plain | 202 | 819 | 9 | 6 / 809 | 4 / 2 | 0.030 | large:not_allocated 1, large:params 2, small:not_allocated 6 |
| short | base-short | 202 | 819 | 415 | 411 / 404 | 208 / 203 | 2.035 | large:not_allocated 3, large:params 1, small:not_allocated 411 |
| short | off-short | 202 | 819 | 415 | 411 / 404 | 208 / 203 | 2.035 | large:not_allocated 3, large:params 1, small:not_allocated 411 |
| short | on-short | 202 | 819 | 10 | 7 / 808 | 4 / 3 | 0.035 | large:not_allocated 1, large:params 2, small:not_allocated 7 |
| pmin | base-pmin | 186 | 776 | 415 | 411 / 361 | 208 / 203 | 2.210 | large:not_allocated 3, large:params 1, small:not_allocated 411 |
| pmin | off-pmin | 186 | 776 | 415 | 411 / 361 | 208 / 203 | 2.210 | large:not_allocated 3, large:params 1, small:not_allocated 411 |
| pmin | on-pmin | 186 | 776 | 9 | 6 / 766 | 4 / 2 | 0.032 | large:not_allocated 1, large:params 2, small:not_allocated 6 |

