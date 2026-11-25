12 scored attempts, arms o5kdpa; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | o5kdpa: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 3.6, 15, 15 / 2 / 0 |
| bugfix-javascript-space-age | pass, 0.8, 4, 4 / 0 / 0 |
| bugfix-python-phone-number | pass, 1.4, 5, 5 / 0 / 0 |
| bugfix-javascript-triangle | pass, 1.0, 4, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 1.7, 5, 5 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 1.3, 5, 5 / 0 / 0 |
| bugfix-python-list-ops | pass, 1.3, 4, 4 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 0.9, 4, 4 / 0 / 0 |
| bugfix-python-scale-generator | pass, 3.9, 7, 7 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 1.1, 4, 4 / 0 / 0 |
| bugfix-python-pig-latin | pass, 1.8, 5, 5 / 0 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 5.2, 6, 6 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| o5kdpa | 12 / 12 | 23.9 | 1.4 | 5.37, 9.09 | 108 | 20.5 | 68 / 2 / 0 | 0 |

Graph reallocations of the target in each arm's server logs (C804): an arm is marked `reallocates` where another arm's logs have none.

- o5kdpa: graph reallocations 0 of 68 requests (0 same size, 0 changed)

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | o5kdpa: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 7.59, 789 |
| 2 | 12, 9.07, 1452 |
| 3 | 12, 2.76, 128 |
| 4 | 12, 1.80, 40 |
| 5 | 7, 1.33, 36 |
| 6 | 3, 1.52, 44 |
| 7 | 2, 1.27, 34 |
| 8 | 1, 0.81, 18 |
| 9 | 1, 1.28, 31 |
| 10 | 1, 1.26, 32 |
| 11 | 1, 1.28, 35 |
| 12 | 1, 6.31, 331 |
| 13 | 1, 1.29, 32 |
| 14 | 1, 1.32, 35 |
| 15 | 1, 1.67, 48 |
