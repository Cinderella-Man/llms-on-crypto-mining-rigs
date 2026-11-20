12 scored attempts, arms v4r; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | v4r: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 2.2, 5, 5 / 0 / 0 |
| bugfix-javascript-space-age | pass, 0.8, 5, 4 / 0 / 0 |
| bugfix-python-phone-number | pass, 1.2, 6, 6 / 0 / 0 |
| bugfix-javascript-triangle | pass, 0.7, 5, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 1.1, 5, 5 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 0.8, 5, 4 / 0 / 0 |
| bugfix-python-list-ops | pass, 1.8, 5, 5 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 0.6, 5, 4 / 0 / 0 |
| bugfix-python-scale-generator | pass, 1.8, 5, 5 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 1.0, 5, 5 / 0 / 0 |
| bugfix-python-pig-latin | pass, 3.5, 8, 8 / 2 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 4.4, 5, 5 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| v4r | 12 / 12 | 19.9 | 1.1 | 2.64, 7.94 | 151 | 30.4 | 60 / 2 / 0 | 0 |

Graph reallocations of the target in each arm's server logs (C804): an arm is marked `reallocates` where another arm's logs have none.

- v4r: graph reallocations 0 of 64 requests (0 same size, 0 changed)

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | v4r: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 6.35, 789 |
| 2 | 12, 7.83, 1648 |
| 3 | 12, 2.96, 356 |
| 4 | 12, 0.98, 31 |
| 5 | 12, 1.61, 104 |
| 6 | 2, 1.50, 132 |
| 7 | 1, 3.04, 390 |
| 8 | 1, 1.95, 190 |
