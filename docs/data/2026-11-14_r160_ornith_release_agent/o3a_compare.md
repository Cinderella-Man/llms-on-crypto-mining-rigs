12 scored attempts, arms o3a; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | o3a: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 4.7, 13, 13 / 1 / 0 |
| bugfix-javascript-space-age | pass, 1.3, 4, 4 / 0 / 0 |
| bugfix-python-phone-number | pass, 2.5, 5, 5 / 0 / 0 |
| bugfix-javascript-triangle | pass, 1.2, 4, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 3.2, 7, 7 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 1.5, 4, 4 / 0 / 0 |
| bugfix-python-list-ops | pass, 2.5, 5, 5 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 2.0, 7, 7 / 1 / 0 |
| bugfix-python-scale-generator | pass, 9.2, 7, 7 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 1.5, 4, 4 / 0 / 0 |
| bugfix-python-pig-latin | pass, 3.6, 5, 5 / 0 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 2.6, 6, 6 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| o3a | 12 / 12 | 35.8 | 2.5 | 7.26, 19.56 | 61 | 16.2 | 71 / 2 / 0 | 0 |

Graph reallocations of the target in each arm's server logs (C804): an arm is marked `reallocates` where another arm's logs have none.

- o3a: graph reallocations 0 of 71 requests (0 same size, 0 changed)

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | o3a: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 16.34, 789 |
| 2 | 12, 18.51, 1452 |
| 3 | 12, 7.83, 316 |
| 4 | 12, 5.01, 128 |
| 5 | 8, 3.25, 36 |
| 6 | 5, 3.90, 61 |
| 7 | 4, 4.19, 76 |
| 8 | 1, 3.24, 32 |
| 9 | 1, 3.29, 35 |
| 10 | 1, 3.34, 32 |
| 11 | 1, 3.27, 32 |
| 12 | 1, 3.29, 35 |
| 13 | 1, 3.36, 36 |
