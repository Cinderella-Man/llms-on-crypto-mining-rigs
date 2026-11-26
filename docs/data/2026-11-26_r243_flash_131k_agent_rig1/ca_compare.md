12 scored attempts, arms v4r; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | v4r: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 3.4, 6, 6 / 0 / 0 |
| bugfix-javascript-space-age | pass, 0.7, 5, 4 / 0 / 0 |
| bugfix-python-phone-number | pass, 1.0, 5, 5 / 0 / 0 |
| bugfix-javascript-triangle | pass, 0.7, 5, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 3.7, 5, 5 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 0.8, 5, 4 / 0 / 0 |
| bugfix-python-list-ops | pass, 1.4, 5, 5 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 1.2, 10, 9 / 0 / 0 |
| bugfix-python-scale-generator | pass, 1.4, 6, 6 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 0.8, 5, 5 / 0 / 0 |
| bugfix-python-pig-latin | pass, 1.6, 5, 5 / 0 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 2.7, 11, 11 / 1 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| v4r | 12 / 12 | 19.2 | 1.3 | 2.43, 8.41 | 149 | 29.1 | 69 / 1 / 0 | 0 |

Graph reallocations of the target in each arm's server logs (C804): an arm is marked `reallocates` where another arm's logs have none.

- v4r: graph reallocations 0 of 73 requests (0 same size, 0 changed)

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | v4r: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 5.07, 789 |
| 2 | 12, 8.61, 1720 |
| 3 | 12, 2.43, 303 |
| 4 | 12, 1.07, 31 |
| 5 | 12, 1.76, 106 |
| 6 | 4, 3.31, 410 |
| 7 | 2, 1.59, 130 |
| 8 | 2, 1.39, 61 |
| 9 | 2, 2.50, 288 |
| 10 | 2, 2.28, 220 |
| 11 | 1, 3.17, 375 |
