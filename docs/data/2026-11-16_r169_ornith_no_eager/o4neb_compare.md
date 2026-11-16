12 scored attempts, arms o4neb; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | o4neb: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 2.5, 6, 6 / 1 / 0 |
| bugfix-javascript-space-age | pass, 0.9, 4, 4 / 0 / 0 |
| bugfix-python-phone-number | pass, 1.8, 5, 5 / 0 / 0 |
| bugfix-javascript-triangle | pass, 1.0, 4, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 1.7, 5, 5 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 1.5, 4, 4 / 0 / 0 |
| bugfix-python-list-ops | pass, 3.2, 4, 4 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 1.0, 4, 4 / 0 / 0 |
| bugfix-python-scale-generator | pass, 10.1, 9, 9 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 1.1, 4, 4 / 0 / 0 |
| bugfix-python-pig-latin | pass, 2.6, 5, 5 / 0 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 3.3, 9, 9 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| o4neb | 12 / 12 | 30.5 | 1.7 | 6.43, 10.08 | 97 | 16.2 | 63 / 1 / 0 | 0 |

Graph reallocations of the target in each arm's server logs (C804): an arm is marked `reallocates` where another arm's logs have none.

- o4neb: graph reallocations 0 of 63 requests (0 same size, 0 changed)

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | o4neb: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 8.35, 789 |
| 2 | 12, 10.03, 1452 |
| 3 | 12, 3.05, 126 |
| 4 | 12, 3.98, 172 |
| 5 | 6, 1.43, 34 |
| 6 | 3, 3.17, 117 |
| 7 | 2, 1.46, 32 |
| 8 | 2, 2.31, 72 |
| 9 | 2, 5.61, 243 |
