12 scored attempts, arms o4g2a; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | o4g2a: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 2.7, 9, 9 / 1 / 0 |
| bugfix-javascript-space-age | pass, 0.8, 4, 4 / 0 / 0 |
| bugfix-python-phone-number | pass, 1.8, 5, 5 / 0 / 0 |
| bugfix-javascript-triangle | pass, 1.0, 4, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 2.5, 6, 6 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 1.2, 4, 4 / 0 / 0 |
| bugfix-python-list-ops | pass, 1.6, 5, 5 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 0.9, 4, 4 / 0 / 0 |
| bugfix-python-scale-generator | pass, 4.0, 6, 6 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 1.1, 4, 4 / 0 / 0 |
| bugfix-python-pig-latin | pass, 3.5, 6, 6 / 0 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 1.4, 5, 5 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| o4g2a | 12 / 12 | 22.6 | 1.5 | 6.99, 11.18 | 98 | 17.7 | 62 / 1 / 0 | 0 |

Graph reallocations of the target in each arm's server logs (C804): an arm is marked `reallocates` where another arm's logs have none.

- o4g2a: graph reallocations 0 of 62 requests (0 same size, 0 changed)

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | o4g2a: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 8.53, 789 |
| 2 | 12, 10.34, 1452 |
| 3 | 12, 5.15, 228 |
| 4 | 12, 4.99, 212 |
| 5 | 7, 1.38, 36 |
| 6 | 4, 1.55, 40 |
| 7 | 1, 2.43, 81 |
| 8 | 1, 1.37, 32 |
| 9 | 1, 1.39, 35 |
