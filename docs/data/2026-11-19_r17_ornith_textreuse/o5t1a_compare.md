12 scored attempts, arms o5t1a; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | o5t1a: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 2.5, 11, 11 / 2 / 0 |
| bugfix-javascript-space-age | pass, 0.8, 4, 4 / 0 / 0 |
| bugfix-python-phone-number | pass, 2.0, 5, 5 / 0 / 0 |
| bugfix-javascript-triangle | pass, 0.9, 4, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 3.4, 8, 8 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 1.3, 5, 5 / 0 / 0 |
| bugfix-python-list-ops | pass, 1.2, 4, 4 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 0.9, 4, 4 / 0 / 0 |
| bugfix-python-scale-generator | pass, 3.3, 5, 5 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 1.0, 4, 4 / 0 / 0 |
| bugfix-python-pig-latin | pass, 2.9, 5, 5 / 0 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 1.4, 5, 5 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| o5t1a | 12 / 12 | 21.4 | 1.3 | 6.13, 9.58 | 103 | 19.4 | 64 / 2 / 0 | 0 |

Graph reallocations of the target in each arm's server logs (C804): an arm is marked `reallocates` where another arm's logs have none.

- o5t1a: graph reallocations 0 of 64 requests (0 same size, 0 changed)

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | o5t1a: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 8.00, 789 |
| 2 | 12, 9.58, 1452 |
| 3 | 12, 5.34, 294 |
| 4 | 12, 1.27, 34 |
| 5 | 7, 1.70, 53 |
| 6 | 2, 1.62, 48 |
| 7 | 2, 1.53, 43 |
| 8 | 2, 1.29, 34 |
| 9 | 1, 1.28, 35 |
| 10 | 1, 2.15, 72 |
| 11 | 1, 4.39, 183 |
