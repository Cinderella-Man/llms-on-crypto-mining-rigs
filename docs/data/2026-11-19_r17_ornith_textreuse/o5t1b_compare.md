12 scored attempts, arms o5t1b; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | o5t1b: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 1.9, 7, 7 / 1 / 0 |
| bugfix-javascript-space-age | pass, 0.7, 4, 4 / 0 / 0 |
| bugfix-python-phone-number | pass, 1.8, 5, 5 / 0 / 0 |
| bugfix-javascript-triangle | pass, 0.9, 4, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 1.5, 5, 5 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 1.1, 4, 4 / 0 / 0 |
| bugfix-python-list-ops | pass, 8.6, 5, 5 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 0.8, 4, 4 / 0 / 0 |
| bugfix-python-scale-generator | runaway (time), 12.0, 3, 3 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 1.0, 4, 4 / 0 / 0 |
| bugfix-python-pig-latin | pass, 3.0, 5, 5 / 0 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 1.5, 5, 5 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| o5t1b | 11 / 12 | 34.8 | 1.5 | 6.94, 10.31 | 106 | 19.0 | 55 / 1 / 0 | 1 |

Graph reallocations of the target in each arm's server logs (C804): an arm is marked `reallocates` where another arm's logs have none.

- o5t1b: graph reallocations 0 of 54 requests (0 same size, 0 changed)

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | o5t1b: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 7.98, 789 |
| 2 | 12, 9.56, 1452 |
| 3 | 12, 5.77, 223 |
| 4 | 11, 1.56, 44 |
| 5 | 6, 1.38, 36 |
| 6 | 1, 1.25, 32 |
| 7 | 1, 1.34, 35 |
