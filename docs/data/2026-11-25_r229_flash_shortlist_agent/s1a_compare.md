12 scored attempts, arms v4r; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | v4r: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 3.3, 8, 8 / 0 / 0 |
| bugfix-javascript-space-age | pass, 0.6, 4, 3 / 0 / 0 |
| bugfix-python-phone-number | pass, 0.8, 5, 5 / 0 / 0 |
| bugfix-javascript-triangle | pass, 0.6, 5, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 2.4, 6, 6 / 1 / 0 |
| bugfix-javascript-phone-number | pass, 0.7, 5, 4 / 0 / 0 |
| bugfix-python-list-ops | pass, 1.4, 5, 5 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 0.7, 5, 4 / 0 / 0 |
| bugfix-python-scale-generator | pass, 2.6, 5, 5 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 0.8, 5, 5 / 0 / 0 |
| bugfix-python-pig-latin | pass, 2.7, 7, 6 / 1 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 3.2, 6, 6 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| v4r | 12 / 12 | 19.9 | 1.1 | 2.19, 7.88 | 163 | 30.1 | 61 / 2 / 0 | 0 |

Graph reallocations of the target in each arm's server logs (C804): an arm is marked `reallocates` where another arm's logs have none.

- v4r: graph reallocations 0 of 66 requests (0 same size, 0 changed)

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | v4r: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 4.78, 789 |
| 2 | 12, 7.83, 1618 |
| 3 | 12, 2.29, 318 |
| 4 | 12, 0.97, 31 |
| 5 | 11, 1.61, 87 |
| 6 | 4, 1.60, 112 |
| 7 | 2, 1.95, 251 |
| 8 | 1, 2.49, 344 |
