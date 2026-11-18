12 scored attempts, arms o4oc0; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | o4oc0: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 2.5, 6, 6 / 1 / 0 |
| bugfix-javascript-space-age | pass, 1.0, 4, 4 / 0 / 0 |
| bugfix-python-phone-number | pass, 2.7, 6, 6 / 0 / 0 |
| bugfix-javascript-triangle | pass, 1.0, 4, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 2.5, 6, 6 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 2.4, 9, 9 / 1 / 0 |
| bugfix-python-list-ops | pass, 4.1, 4, 4 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 0.9, 4, 4 / 0 / 0 |
| bugfix-python-scale-generator | pass, 6.7, 8, 8 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 1.1, 4, 4 / 0 / 0 |
| bugfix-python-pig-latin | pass, 3.1, 6, 6 / 0 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 3.8, 6, 6 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| o4oc0 | 12 / 12 | 31.9 | 2.5 | 7.07, 11.51 | 102 | 16.4 | 67 / 2 / 0 | 0 |

Graph reallocations of the target in each arm's server logs (C804): an arm is marked `reallocates` where another arm's logs have none.

- o4oc0: graph reallocations 0 of 67 requests (0 same size, 0 changed)

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | o4oc0: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 8.48, 789 |
| 2 | 12, 10.19, 1452 |
| 3 | 12, 7.71, 428 |
| 4 | 12, 5.07, 214 |
| 5 | 7, 1.39, 32 |
| 6 | 7, 2.94, 104 |
| 7 | 2, 10.09, 1342 |
| 8 | 2, 1.36, 32 |
| 9 | 1, 5.59, 271 |
