12 scored attempts, arms o4sna; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | o4sna: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 2.5, 9, 9 / 2 / 0 |
| bugfix-javascript-space-age | pass, 0.9, 4, 4 / 0 / 0 |
| bugfix-python-phone-number | pass, 1.8, 5, 5 / 0 / 0 |
| bugfix-javascript-triangle | pass, 1.0, 4, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 3.2, 6, 6 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 1.4, 5, 5 / 0 / 0 |
| bugfix-python-list-ops | pass, 3.6, 5, 5 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 1.0, 4, 4 / 0 / 0 |
| bugfix-python-scale-generator | pass, 4.5, 5, 5 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 1.2, 4, 4 / 0 / 0 |
| bugfix-python-pig-latin | pass, 3.8, 7, 7 / 0 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 4.2, 5, 5 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| o4sna | 12 / 12 | 29.1 | 2.2 | 6.69, 11.29 | 103 | 16.5 | 63 / 2 / 0 | 0 |

Graph reallocations of the target in each arm's server logs (C804): an arm is marked `reallocates` where another arm's logs have none.

- o4sna: graph reallocations 0 of 63 requests (0 same size, 0 changed)

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | o4sna: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 8.43, 789 |
| 2 | 12, 10.17, 1452 |
| 3 | 12, 6.91, 367 |
| 4 | 12, 3.25, 128 |
| 5 | 8, 1.44, 36 |
| 6 | 3, 1.41, 36 |
| 7 | 2, 1.39, 34 |
| 8 | 1, 1.41, 32 |
| 9 | 1, 1.84, 52 |
