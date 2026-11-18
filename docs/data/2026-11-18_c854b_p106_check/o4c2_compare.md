12 scored attempts, arms o4c2; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | o4c2: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 2.0, 6, 6 / 0 / 0 |
| bugfix-javascript-space-age | pass, 1.0, 4, 4 / 0 / 0 |
| bugfix-python-phone-number | pass, 1.6, 5, 5 / 0 / 0 |
| bugfix-javascript-triangle | pass, 1.1, 4, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 2.8, 7, 7 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 1.4, 5, 5 / 0 / 0 |
| bugfix-python-list-ops | pass, 3.3, 5, 5 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 1.0, 4, 4 / 0 / 0 |
| bugfix-python-scale-generator | pass, 7.8, 8, 8 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 1.1, 4, 4 / 0 / 0 |
| bugfix-python-pig-latin | pass, 2.5, 5, 5 / 0 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 2.2, 6, 6 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| o4c2 | 12 / 12 | 28.0 | 1.8 | 7.56, 11.11 | 96 | 16.3 | 63 / 0 / 0 | 0 |

Graph reallocations of the target in each arm's server logs (C804): an arm is marked `reallocates` where another arm's logs have none.

- o4c2: graph reallocations 0 of 63 requests (0 same size, 0 changed)

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | o4c2: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 8.58, 789 |
| 2 | 12, 10.36, 1452 |
| 3 | 12, 7.99, 449 |
| 4 | 12, 1.38, 31 |
| 5 | 8, 1.39, 36 |
| 6 | 4, 1.40, 34 |
| 7 | 2, 5.81, 270 |
| 8 | 1, 8.92, 449 |
