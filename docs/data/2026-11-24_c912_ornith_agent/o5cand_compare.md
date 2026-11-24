12 scored attempts, arms o5cand; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | o5cand: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 2.3, 8, 8 / 1 / 0 |
| bugfix-javascript-space-age | pass, 0.7, 4, 4 / 0 / 0 |
| bugfix-python-phone-number | pass, 1.7, 5, 5 / 0 / 0 |
| bugfix-javascript-triangle | pass, 0.9, 4, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 2.2, 6, 6 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 1.2, 5, 5 / 0 / 0 |
| bugfix-python-list-ops | pass, 1.7, 5, 5 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 0.9, 4, 4 / 0 / 0 |
| bugfix-python-scale-generator | pass, 3.4, 6, 6 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 1.0, 4, 4 / 0 / 0 |
| bugfix-python-pig-latin | pass, 2.6, 5, 5 / 0 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 1.6, 6, 6 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| o5cand | 12 / 12 | 20.0 | 1.6 | 6.06, 9.38 | 106 | 19.5 | 62 / 1 / 0 | 0 |

Graph reallocations of the target in each arm's server logs (C804): an arm is marked `reallocates` where another arm's logs have none.

- o5cand: graph reallocations 0 of 62 requests (0 same size, 0 changed)

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | o5cand: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 7.71, 789 |
| 2 | 12, 9.33, 1452 |
| 3 | 12, 6.02, 340 |
| 4 | 12, 2.04, 74 |
| 5 | 8, 1.28, 36 |
| 6 | 4, 1.33, 36 |
| 7 | 1, 1.23, 32 |
| 8 | 1, 1.27, 35 |
