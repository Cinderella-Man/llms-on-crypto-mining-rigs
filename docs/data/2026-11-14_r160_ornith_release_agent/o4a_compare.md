12 scored attempts, arms o4a; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | o4a: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 3.3, 7, 7 / 0 / 0 |
| bugfix-javascript-space-age | pass, 1.1, 4, 4 / 0 / 0 |
| bugfix-python-phone-number | pass, 2.5, 5, 5 / 0 / 0 |
| bugfix-javascript-triangle | pass, 1.2, 4, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 2.8, 6, 6 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 1.7, 5, 5 / 0 / 0 |
| bugfix-python-list-ops | pass, 1.6, 4, 4 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 1.2, 4, 4 / 0 / 0 |
| bugfix-python-scale-generator | pass, 5.6, 5, 5 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 1.5, 4, 4 / 0 / 0 |
| bugfix-python-pig-latin | pass, 4.5, 7, 7 / 0 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 2.7, 7, 7 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| o4a | 12 / 12 | 29.7 | 2.1 | 8.65, 19.22 | 57 | 16.5 | 62 / 0 / 0 | 0 |

Graph reallocations of the target in each arm's server logs (C804): an arm is marked `reallocates` where another arm's logs have none.

- o4a: graph reallocations 0 of 62 requests (0 same size, 0 changed)

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | o4a: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 16.32, 789 |
| 2 | 12, 18.85, 1452 |
| 3 | 12, 7.76, 280 |
| 4 | 12, 3.37, 34 |
| 5 | 7, 3.44, 36 |
| 6 | 4, 4.25, 78 |
| 7 | 3, 3.35, 36 |
