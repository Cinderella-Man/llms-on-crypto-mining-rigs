12 scored attempts, arms v4r; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | v4r: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 1.6, 5, 5 / 0 / 0 |
| bugfix-javascript-space-age | pass, 0.7, 5, 6 / 0 / 0 |
| bugfix-python-phone-number | pass, 0.9, 5, 5 / 0 / 0 |
| bugfix-javascript-triangle | pass, 0.7, 5, 5 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 3.4, 4, 4 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 0.8, 5, 4 / 0 / 0 |
| bugfix-python-list-ops | pass, 1.3, 5, 5 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 0.8, 6, 5 / 0 / 0 |
| bugfix-python-scale-generator | pass, 1.4, 5, 5 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 1.0, 8, 8 / 2 / 0 |
| bugfix-python-pig-latin | pass, 1.6, 5, 4 / 0 / 0 |
| bugfix-javascript-variable-length-quantity | runaway (time), 12.0, 2, 2 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| v4r | 11 / 12 | 26.3 | 1.1 | 2.48, 8.18 | 145 | 30.3 | 58 / 2 / 0 | 1 |

Graph reallocations of the target in each arm's server logs (C804): an arm is marked `reallocates` where another arm's logs have none.

- v4r: graph reallocations 0 of 59 requests (0 same size, 0 changed)

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | v4r: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 5.06, 789 |
| 2 | 12, 8.11, 1557 |
| 3 | 11, 2.49, 334 |
| 4 | 11, 1.06, 30 |
| 5 | 10, 1.67, 90 |
| 6 | 2, 2.02, 182 |
| 7 | 1, 1.07, 32 |
| 8 | 1, 1.71, 113 |
