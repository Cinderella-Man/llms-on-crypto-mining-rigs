## The draft by depth: arm `draft`, tails tail1, tail2, tail3

Per tail: draft tokens accepted of drafted (the share), then tokens a step. Pooled: the counters summed over the tails, the acceptance with a 95 % Wilson interval, tokens a step, and the acceptance minus the acceptance at depth 65,536 with a 95 % interval. The intervals count every drafted token as an independent trial.

### Greedy, pass 1

| Depth (tokens) | tail1 | tail2 | tail3 | Pooled | Acceptance | 95 % interval | Tokens a step | Against 65,536 | 95 % interval |
|---:|---|---|---|---|---:|---|---:|---:|---|
| 65,536 | 176 of 235 (0.749), 3.20 | 181 of 219 (0.826), 3.41 | 189 of 197 (0.959), 3.82 | 546 of 651 | 0.839 | 0.808 to 0.865 | 3.46 | — | — |
| 120,000 | 188 of 198 (0.949), 3.76 | 161 of 280 (0.575), 2.69 | 190 of 193 (0.984), 3.88 | 539 of 671 | 0.803 | 0.772 to 0.832 | 3.35 | -0.035 | -0.077 to +0.006 |

### Greedy, pass 2

| Depth (tokens) | tail1 | tail2 | tail3 | Pooled | Acceptance | 95 % interval | Tokens a step | Against 65,536 | 95 % interval |
|---:|---|---|---|---|---:|---|---:|---:|---|
| 65,536 | 175 of 238 (0.735), 3.16 | 182 of 216 (0.843), 3.46 | 189 of 197 (0.959), 3.82 | 546 of 651 | 0.839 | 0.808 to 0.865 | 3.46 | — | — |
| 120,000 | 188 of 198 (0.949), 3.76 | 161 of 280 (0.575), 2.69 | 190 of 193 (0.984), 3.88 | 539 of 671 | 0.803 | 0.772 to 0.832 | 3.35 | -0.035 | -0.077 to +0.006 |

### Sampler

| Depth (tokens) | tail1 | tail2 | tail3 | Pooled | Acceptance | 95 % interval | Tokens a step | Against 65,536 | 95 % interval |
|---:|---|---|---|---|---:|---|---:|---:|---|
| 65,536 | 169 of 256 (0.660), 2.94 | 175 of 237 (0.738), 3.16 | 162 of 276 (0.587), 2.72 | 506 of 769 | 0.658 | 0.624 to 0.691 | 2.93 | — | — |
| 120,000 | 157 of 293 (0.536), 2.59 | 162 of 278 (0.583), 2.72 | 164 of 270 (0.607), 2.78 | 483 of 841 | 0.574 | 0.541 to 0.607 | 2.69 | -0.084 | -0.131 to -0.036 |
