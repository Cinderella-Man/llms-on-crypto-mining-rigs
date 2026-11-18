12 scored attempts, arms o4oc2; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | o4oc2: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 2.2, 6, 6 / 0 / 0 |
| bugfix-javascript-space-age | pass, 0.8, 4, 4 / 0 / 0 |
| bugfix-python-phone-number | pass, 2.3, 5, 5 / 0 / 0 |
| bugfix-javascript-triangle | pass, 0.9, 4, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 3.7, 9, 9 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 1.3, 5, 5 / 0 / 0 |
| bugfix-python-list-ops | pass, 1.3, 4, 4 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 0.9, 4, 4 / 0 / 0 |
| bugfix-python-scale-generator | pass, 8.0, 8, 8 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 1.1, 4, 4 / 0 / 0 |
| bugfix-python-pig-latin | pass, 3.0, 5, 5 / 0 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 1.6, 5, 5 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| o4oc2 | 12 / 12 | 27.0 | 1.5 | 7.39, 10.75 | 110 | 17.9 | 63 / 0 / 0 | 0 |

Graph reallocations of the target in each arm's server logs (C804): an arm is marked `reallocates` where another arm's logs have none.

- o4oc2: graph reallocations 0 of 63 requests (0 same size, 0 changed)

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | o4oc2: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 8.01, 789 |
| 2 | 12, 9.58, 1452 |
| 3 | 12, 7.46, 442 |
| 4 | 12, 4.56, 210 |
| 5 | 7, 1.37, 36 |
| 6 | 3, 1.30, 36 |
| 7 | 2, 7.48, 1186 |
| 8 | 2, 1.35, 34 |
| 9 | 1, 1.35, 36 |
