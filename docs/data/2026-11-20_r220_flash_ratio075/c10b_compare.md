12 scored attempts, arms v4r; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | v4r: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 1.9, 5, 5 / 0 / 0 |
| bugfix-javascript-space-age | pass, 0.8, 5, 4 / 0 / 0 |
| bugfix-python-phone-number | pass, 1.2, 6, 6 / 0 / 0 |
| bugfix-javascript-triangle | pass, 0.6, 5, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 1.2, 5, 5 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 0.8, 5, 4 / 0 / 0 |
| bugfix-python-list-ops | pass, 1.1, 5, 5 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 0.6, 5, 4 / 0 / 0 |
| bugfix-python-scale-generator | pass, 2.1, 8, 8 / 0 / 0 |
| bugfix-javascript-complex-numbers | runaway (time), 12.0, 39, 38 / 3 / 0 |
| bugfix-python-pig-latin | pass, 1.8, 6, 5 / 1 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 5.2, 5, 6 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| v4r | 11 / 12 | 29.3 | 1.2 | 2.27, 8.94 | 142 | 29.5 | 94 / 4 / 0 | 1 |

Graph reallocations of the target in each arm's server logs (C804): an arm is marked `reallocates` where another arm's logs have none.

- v4r: graph reallocations 0 of 98 requests (0 same size, 0 changed)

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | v4r: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 6.49, 809 |
| 2 | 12, 7.89, 1648 |
| 3 | 12, 2.22, 318 |
| 4 | 12, 0.97, 31 |
| 5 | 12, 1.66, 115 |
| 6 | 4, 2.01, 241 |
| 7 | 2, 1.53, 170 |
| 8 | 2, 2.05, 228 |
| 9 | 1, 1.60, 88 |
| 10 | 1, 2.03, 216 |
| 11 | 1, 1.38, 56 |
| 12 | 1, 1.72, 111 |
| 13 | 1, 3.14, 468 |
| 14 | 1, 17.96, 3150 |
| 15 | 1, 8.69, 740 |
| 16 | 1, 3.29, 477 |
| 17 | 1, 3.60, 541 |
| 18 | 1, 2.25, 291 |
| 19 | 1, 1.03, 19 |
| 20 | 1, 2.35, 342 |
| 21 | 1, 2.30, 313 |
| 22 | 1, 2.27, 295 |
| 23 | 1, 12.37, 1531 |
| 24 | 1, 2.38, 315 |
| 25 | 1, 0.87, 19 |
| 26 | 1, 1.57, 66 |
| 27 | 1, 12.73, 1305 |
| 28 | 1, 3.19, 397 |
| 29 | 1, 1.37, 56 |
| 30 | 1, 0.85, 19 |
| 31 | 1, 1.30, 46 |
| 32 | 1, 1.12, 26 |
| 33 | 1, 1.42, 56 |
| 34 | 1, 3.26, 431 |
| 35 | 1, 13.39, 1308 |
| 36 | 1, 2.08, 218 |
| 37 | 1, 1.13, 32 |
| 38 | 1, 11.18, 645 |
| 39 | 1, —, — |
