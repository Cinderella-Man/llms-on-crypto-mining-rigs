12 scored attempts, arms v4r; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | v4r: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 2.1, 9, 9 / 1 / 0 |
| bugfix-javascript-space-age | pass, 0.6, 4, 3 / 0 / 0 |
| bugfix-python-phone-number | pass, 1.1, 6, 6 / 0 / 0 |
| bugfix-javascript-triangle | pass, 0.6, 5, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 7.9, 9, 9 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 1.1, 6, 6 / 0 / 0 |
| bugfix-python-list-ops | pass, 1.2, 5, 5 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 0.7, 5, 4 / 0 / 0 |
| bugfix-python-scale-generator | pass, 3.8, 5, 5 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 1.1, 7, 6 / 0 / 0 |
| bugfix-python-pig-latin | pass, 1.5, 5, 5 / 0 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 4.4, 7, 7 / 1 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| v4r | 12 / 12 | 26.1 | 1.2 | 2.58, 7.91 | 159 | 32.0 | 69 / 2 / 0 | 0 |

Graph reallocations of the target in each arm's server logs (C804): an arm is marked `reallocates` where another arm's logs have none.

- v4r: graph reallocations 0 of 73 requests (0 same size, 0 changed)

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | v4r: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 4.81, 809 |
| 2 | 12, 7.90, 1648 |
| 3 | 12, 2.61, 370 |
| 4 | 12, 0.99, 32 |
| 5 | 11, 1.16, 40 |
| 6 | 6, 1.93, 186 |
| 7 | 4, 1.87, 168 |
| 8 | 2, 1.25, 48 |
| 9 | 2, 2.45, 332 |
