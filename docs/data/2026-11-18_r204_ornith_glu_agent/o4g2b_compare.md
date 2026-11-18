12 scored attempts, arms o4g2b; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | o4g2b: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 2.1, 6, 6 / 0 / 0 |
| bugfix-javascript-space-age | pass, 0.8, 4, 4 / 0 / 0 |
| bugfix-python-phone-number | pass, 2.4, 5, 5 / 0 / 0 |
| bugfix-javascript-triangle | pass, 0.9, 4, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 1.2, 4, 4 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 1.4, 5, 5 / 0 / 0 |
| bugfix-python-list-ops | pass, 1.2, 4, 4 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 1.0, 4, 4 / 0 / 0 |
| bugfix-python-scale-generator | pass, 8.0, 8, 8 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 1.1, 4, 4 / 0 / 0 |
| bugfix-python-pig-latin | pass, 2.7, 5, 5 / 0 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 3.7, 6, 6 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| o4g2b | 12 / 12 | 26.6 | 1.3 | 6.97, 11.74 | 103 | 17.3 | 59 / 0 / 0 | 0 |

Graph reallocations of the target in each arm's server logs (C804): an arm is marked `reallocates` where another arm's logs have none.

- o4g2b: graph reallocations 0 of 59 requests (0 same size, 0 changed)

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | o4g2b: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 8.51, 789 |
| 2 | 12, 10.39, 1452 |
| 3 | 12, 3.10, 128 |
| 4 | 12, 3.17, 126 |
| 5 | 6, 1.46, 36 |
| 6 | 3, 1.34, 35 |
| 7 | 1, 1.94, 52 |
| 8 | 1, 2.69, 78 |
