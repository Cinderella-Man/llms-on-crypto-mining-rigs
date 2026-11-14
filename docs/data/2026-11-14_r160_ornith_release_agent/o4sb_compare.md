12 scored attempts, arms o4sb; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | o4sb: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 3.4, 11, 11 / 1 / 0 |
| bugfix-javascript-space-age | pass, 0.8, 4, 4 / 0 / 0 |
| bugfix-python-phone-number | pass, 1.9, 5, 5 / 0 / 0 |
| bugfix-javascript-triangle | pass, 0.9, 4, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 2.4, 5, 5 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 1.1, 4, 4 / 0 / 0 |
| bugfix-python-list-ops | pass, 1.1, 4, 4 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 0.9, 4, 4 / 0 / 0 |
| bugfix-python-scale-generator | pass, 5.0, 5, 5 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 1.1, 4, 4 / 0 / 0 |
| bugfix-python-pig-latin | pass, 4.7, 9, 9 / 0 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 2.6, 6, 6 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| o4sb | 12 / 12 | 25.9 | 1.5 | 3.27, 11.28 | 114 | 16.5 | 65 / 1 / 0 | 0 |

Graph reallocations of the target in each arm's server logs (C804): an arm is marked `reallocates` where another arm's logs have none.

- o4sb: graph reallocations 0 of 65 requests (0 same size, 0 changed)

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | o4sb: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 8.44, 789 |
| 2 | 12, 10.15, 1452 |
| 3 | 12, 2.08, 127 |
| 4 | 12, 2.87, 246 |
| 5 | 6, 1.43, 36 |
| 6 | 3, 2.67, 191 |
| 7 | 2, 1.83, 61 |
| 8 | 2, 1.41, 34 |
| 9 | 2, 2.05, 110 |
| 10 | 1, 1.42, 32 |
| 11 | 1, 1.88, 52 |
