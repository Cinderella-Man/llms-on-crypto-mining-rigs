12 scored attempts, arms o4b; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | o4b: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 2.5, 6, 6 / 0 / 0 |
| bugfix-javascript-space-age | pass, 1.1, 4, 4 / 0 / 0 |
| bugfix-python-phone-number | pass, 2.8, 5, 5 / 0 / 0 |
| bugfix-javascript-triangle | pass, 1.2, 4, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 2.2, 5, 5 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 1.5, 4, 4 / 0 / 0 |
| bugfix-python-list-ops | pass, 1.7, 4, 4 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 1.3, 4, 4 / 0 / 0 |
| bugfix-python-scale-generator | pass, 8.4, 5, 5 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 1.5, 4, 4 / 0 / 0 |
| bugfix-python-pig-latin | pass, 4.1, 7, 7 / 0 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 2.4, 5, 5 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| o4b | 12 / 12 | 31.0 | 2.0 | 9.58, 20.29 | 65 | 16.4 | 57 / 0 / 0 | 0 |

Graph reallocations of the target in each arm's server logs (C804): an arm is marked `reallocates` where another arm's logs have none.

- o4b: graph reallocations 0 of 57 requests (0 same size, 0 changed)

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | o4b: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 16.26, 789 |
| 2 | 12, 18.94, 1452 |
| 3 | 12, 5.10, 127 |
| 4 | 12, 7.03, 216 |
| 5 | 6, 3.39, 36 |
| 6 | 2, 3.31, 36 |
| 7 | 1, 6.60, 188 |
