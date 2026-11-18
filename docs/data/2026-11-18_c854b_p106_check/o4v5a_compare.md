12 scored attempts, arms o4v5a; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | o4v5a: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 3.5, 9, 9 / 2 / 0 |
| bugfix-javascript-space-age | pass, 0.9, 4, 4 / 0 / 0 |
| bugfix-python-phone-number | pass, 2.1, 5, 5 / 0 / 0 |
| bugfix-javascript-triangle | pass, 1.1, 4, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 2.1, 5, 5 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 1.2, 4, 4 / 0 / 0 |
| bugfix-python-list-ops | pass, 3.0, 5, 5 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 1.0, 4, 4 / 0 / 0 |
| bugfix-python-scale-generator | pass, 5.0, 5, 5 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 1.1, 4, 4 / 0 / 0 |
| bugfix-python-pig-latin | pass, 2.0, 5, 5 / 0 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 2.1, 7, 7 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| o4v5a | 12 / 12 | 24.9 | 2.0 | 6.56, 11.10 | 99 | 16.5 | 61 / 2 / 0 | 0 |

Graph reallocations of the target in each arm's server logs (C804): an arm is marked `reallocates` where another arm's logs have none.

- o4v5a: graph reallocations 0 of 61 requests (0 same size, 0 changed)

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | o4v5a: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 8.63, 789 |
| 2 | 12, 10.31, 1452 |
| 3 | 12, 6.35, 316 |
| 4 | 12, 1.64, 39 |
| 5 | 7, 1.41, 36 |
| 6 | 2, 1.50, 34 |
| 7 | 2, 2.60, 88 |
| 8 | 1, 1.38, 32 |
| 9 | 1, 1.40, 35 |
