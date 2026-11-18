12 scored attempts, arms o4oc1; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | o4oc1: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 2.9, 11, 11 / 2 / 0 |
| bugfix-javascript-space-age | pass, 0.8, 4, 4 / 0 / 0 |
| bugfix-python-phone-number | pass, 1.7, 5, 5 / 0 / 0 |
| bugfix-javascript-triangle | pass, 1.1, 4, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 1.2, 4, 4 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 1.4, 5, 5 / 0 / 0 |
| bugfix-python-list-ops | pass, 1.1, 4, 4 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 1.0, 4, 4 / 0 / 0 |
| bugfix-python-scale-generator | pass, 6.4, 5, 5 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 1.1, 4, 4 / 0 / 0 |
| bugfix-python-pig-latin | pass, 3.6, 5, 5 / 0 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 2.0, 7, 7 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| o4oc1 | 12 / 12 | 24.1 | 1.3 | 6.60, 10.82 | 102 | 17.3 | 62 / 2 / 0 | 0 |

Graph reallocations of the target in each arm's server logs (C804): an arm is marked `reallocates` where another arm's logs have none.

- o4oc1: graph reallocations 0 of 62 requests (0 same size, 0 changed)

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | o4oc1: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 8.30, 789 |
| 2 | 12, 10.06, 1452 |
| 3 | 12, 3.03, 128 |
| 4 | 12, 1.51, 36 |
| 5 | 6, 3.14, 120 |
| 6 | 2, 1.50, 38 |
| 7 | 2, 2.48, 84 |
| 8 | 1, 1.34, 32 |
| 9 | 1, 1.37, 32 |
| 10 | 1, 1.33, 32 |
| 11 | 1, 1.37, 35 |
