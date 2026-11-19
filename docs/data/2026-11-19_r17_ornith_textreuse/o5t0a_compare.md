12 scored attempts, arms o5t0a; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | o5t0a: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 2.2, 7, 7 / 1 / 0 |
| bugfix-javascript-space-age | pass, 0.8, 4, 4 / 0 / 0 |
| bugfix-python-phone-number | pass, 1.7, 5, 5 / 0 / 0 |
| bugfix-javascript-triangle | pass, 0.9, 4, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 1.6, 5, 5 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 1.4, 5, 5 / 0 / 0 |
| bugfix-python-list-ops | pass, 1.1, 4, 4 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 0.9, 4, 4 / 0 / 0 |
| bugfix-python-scale-generator | pass, 6.7, 6, 6 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 1.0, 4, 4 / 0 / 0 |
| bugfix-python-pig-latin | pass, 2.2, 5, 5 / 0 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 4.1, 5, 5 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| o5t0a | 12 / 12 | 24.5 | 1.5 | 7.03, 10.60 | 110 | 19.1 | 58 / 1 / 0 | 0 |

Graph reallocations of the target in each arm's server logs (C804): an arm is marked `reallocates` where another arm's logs have none.

- o5t0a: graph reallocations 0 of 58 requests (0 same size, 0 changed)

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | o5t0a: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 7.95, 789 |
| 2 | 12, 9.57, 1452 |
| 3 | 12, 5.91, 316 |
| 4 | 12, 2.71, 104 |
| 5 | 7, 2.58, 96 |
| 6 | 2, 1.30, 34 |
| 7 | 1, 1.71, 52 |
