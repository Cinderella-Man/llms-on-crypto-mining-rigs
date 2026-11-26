12 scored attempts, arms v4r; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | v4r: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 2.0, 5, 4 / 0 / 0 |
| bugfix-javascript-space-age | pass, 0.7, 5, 4 / 0 / 0 |
| bugfix-python-phone-number | pass, 0.9, 5, 5 / 0 / 0 |
| bugfix-javascript-triangle | pass, 0.7, 5, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 3.2, 6, 6 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 0.7, 4, 3 / 0 / 0 |
| bugfix-python-list-ops | pass, 1.8, 5, 5 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 0.7, 5, 4 / 0 / 0 |
| bugfix-python-scale-generator | pass, 2.6, 5, 5 / 0 / 0 |
| bugfix-javascript-complex-numbers | runaway (time), 12.0, 30, 29 / 0 / 0 |
| bugfix-python-pig-latin | pass, 1.8, 5, 5 / 1 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 1.2, 6, 6 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| v4r | 11 / 12 | 28.2 | 1.5 | 3.08, 8.33 | 149 | 27.0 | 80 / 1 / 0 | 1 |

Graph reallocations of the target in each arm's server logs (C804): an arm is marked `reallocates` where another arm's logs have none.

- v4r: graph reallocations 0 of 85 requests (0 same size, 0 changed)

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | v4r: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 5.31, 809 |
| 2 | 12, 8.31, 1573 |
| 3 | 12, 3.41, 322 |
| 4 | 12, 1.07, 30 |
| 5 | 11, 1.77, 87 |
| 6 | 3, 1.89, 170 |
| 7 | 1, 1.54, 61 |
| 8 | 1, 1.69, 90 |
| 9 | 1, 1.11, 32 |
| 10 | 1, 1.37, 51 |
| 11 | 1, 1.08, 29 |
| 12 | 1, 2.18, 210 |
| 13 | 1, 3.38, 327 |
| 14 | 1, 4.87, 658 |
| 15 | 1, 7.33, 1195 |
| 16 | 1, 4.15, 594 |
| 17 | 1, 6.57, 990 |
| 18 | 1, 4.10, 521 |
| 19 | 1, 5.24, 740 |
| 20 | 1, 3.77, 432 |
| 21 | 1, 2.17, 173 |
| 22 | 1, 2.12, 169 |
| 23 | 1, 7.48, 1112 |
| 24 | 1, 3.44, 364 |
| 25 | 1, 1.39, 40 |
| 26 | 1, 3.28, 315 |
| 27 | 1, 2.14, 147 |
| 28 | 1, 1.87, 76 |
| 29 | 1, 1.87, 73 |
| 30 | 1, —, — |
