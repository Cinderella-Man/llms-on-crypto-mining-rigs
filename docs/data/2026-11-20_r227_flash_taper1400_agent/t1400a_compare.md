12 scored attempts, arms v4r; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | v4r: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 2.0, 7, 7 / 1 / 0 |
| bugfix-javascript-space-age | pass, 0.6, 5, 4 / 0 / 0 |
| bugfix-python-phone-number | pass, 0.8, 6, 5 / 0 / 0 |
| bugfix-javascript-triangle | pass, 0.7, 5, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 2.1, 6, 5 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 1.0, 5, 4 / 0 / 0 |
| bugfix-python-list-ops | pass, 1.8, 5, 5 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 0.7, 5, 5 / 0 / 0 |
| bugfix-python-scale-generator | pass, 3.0, 5, 5 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 1.2, 6, 5 / 0 / 0 |
| bugfix-python-pig-latin | pass, 1.9, 6, 6 / 0 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 5.7, 5, 5 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| v4r | 12 / 12 | 21.5 | 1.5 | 2.32, 7.88 | 158 | 30.7 | 60 / 1 / 0 | 0 |

Graph reallocations of the target in each arm's server logs (C804): an arm is marked `reallocates` where another arm's logs have none.

- v4r: graph reallocations 0 of 66 requests (0 same size, 0 changed)

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | v4r: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 4.67, 789 |
| 2 | 12, 7.80, 1442 |
| 3 | 12, 2.20, 318 |
| 4 | 12, 0.99, 31 |
| 5 | 12, 1.42, 68 |
| 6 | 5, 1.96, 214 |
| 7 | 1, 1.07, 32 |
