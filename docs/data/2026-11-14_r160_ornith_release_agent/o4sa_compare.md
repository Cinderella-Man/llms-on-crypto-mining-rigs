12 scored attempts, arms o4sa; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | o4sa: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 3.7, 9, 9 / 0 / 0 |
| bugfix-javascript-space-age | pass, 0.8, 4, 4 / 0 / 0 |
| bugfix-python-phone-number | pass, 2.0, 5, 5 / 0 / 0 |
| bugfix-javascript-triangle | pass, 0.9, 4, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 2.0, 5, 5 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 1.1, 4, 4 / 0 / 0 |
| bugfix-python-list-ops | pass, 1.9, 4, 4 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 1.3, 6, 7 / 0 / 0 |
| bugfix-python-scale-generator | pass, 8.6, 7, 7 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 1.1, 4, 4 / 0 / 0 |
| bugfix-python-pig-latin | pass, 2.2, 5, 5 / 0 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 2.0, 6, 6 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| o4sa | 12 / 12 | 27.6 | 1.9 | 3.05, 10.97 | 115 | 16.3 | 64 / 0 / 0 | 0 |

Graph reallocations of the target in each arm's server logs (C804): an arm is marked `reallocates` where another arm's logs have none.

- o4sa: graph reallocations 0 of 63 requests (0 same size, 0 changed)

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | o4sa: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 8.45, 789 |
| 2 | 12, 10.19, 1452 |
| 3 | 12, 2.07, 128 |
| 4 | 12, 1.97, 83 |
| 5 | 7, 1.40, 36 |
| 6 | 4, 2.39, 128 |
| 7 | 2, 1.30, 30 |
| 8 | 1, 1.40, 32 |
| 9 | 1, 1.42, 35 |
