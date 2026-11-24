12 scored attempts, arms o5f16a; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | o5f16a: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 2.3, 6, 6 / 1 / 0 |
| bugfix-javascript-space-age | pass, 0.8, 4, 4 / 0 / 0 |
| bugfix-python-phone-number | pass, 1.8, 5, 5 / 0 / 0 |
| bugfix-javascript-triangle | pass, 0.9, 4, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 3.0, 8, 8 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 1.1, 4, 4 / 0 / 0 |
| bugfix-python-list-ops | pass, 1.6, 5, 5 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 0.9, 4, 4 / 0 / 0 |
| bugfix-python-scale-generator | pass, 3.6, 5, 5 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 1.1, 4, 4 / 0 / 0 |
| bugfix-python-pig-latin | pass, 2.2, 5, 5 / 0 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 2.6, 5, 5 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| o5f16a | 12 / 12 | 21.9 | 1.7 | 6.59, 10.27 | 104 | 19.4 | 59 / 1 / 0 | 0 |

Graph reallocations of the target in each arm's server logs (C804): an arm is marked `reallocates` where another arm's logs have none.

- o5f16a: graph reallocations 0 of 59 requests (0 same size, 0 changed)

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | o5f16a: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 8.01, 789 |
| 2 | 12, 9.54, 1452 |
| 3 | 12, 5.60, 292 |
| 4 | 12, 1.57, 39 |
| 5 | 7, 1.28, 36 |
| 6 | 2, 1.08, 28 |
| 7 | 1, 1.29, 32 |
| 8 | 1, 1.31, 36 |
