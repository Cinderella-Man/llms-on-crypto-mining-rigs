12 scored attempts, arms q5f4t; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | q5f4t: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 1.2, 4, 4 / 0 / 0 |
| bugfix-javascript-space-age | pass, 1.0, 10, 9 / 0 / 0 |
| bugfix-python-phone-number | pass, 1.3, 5, 4 / 0 / 0 |
| bugfix-javascript-triangle | pass, 0.3, 4, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 1.0, 6, 5 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 1.0, 5, 5 / 0 / 0 |
| bugfix-python-list-ops | pass, 2.4, 5, 5 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 0.6, 5, 5 / 0 / 0 |
| bugfix-python-scale-generator | pass, 9.3, 8, 7 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 0.8, 5, 5 / 0 / 0 |
| bugfix-python-pig-latin | pass, 1.8, 6, 6 / 1 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 2.5, 11, 11 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| q5f4t | 12 / 12 | 23.0 | 1.1 | 1.82, 7.13 | 196 | 52.0 | 70 / 1 / 0 | 0 |

Graph reallocations of the target in each arm's server logs (C804): an arm is marked `reallocates` where another arm's logs have none.

- q5f4t: graph reallocations 73 of 74 requests (295 same size, 0 changed)

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | q5f4t: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 3.39, 693 |
| 2 | 12, 5.30, 1186 |
| 3 | 12, 1.95, 418 |
| 4 | 12, 0.87, 74 |
| 5 | 10, 1.90, 260 |
| 6 | 5, 1.19, 52 |
| 7 | 3, 0.91, 53 |
| 8 | 3, 0.99, 24 |
| 9 | 2, 0.88, 71 |
| 10 | 2, 1.13, 78 |
| 11 | 1, 1.39, 102 |
