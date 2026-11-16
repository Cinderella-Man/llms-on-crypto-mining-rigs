12 scored attempts, arms f4ta; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | f4ta: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 2.9, 5, 5 / 0 / 0 |
| bugfix-javascript-space-age | pass, 0.8, 7, 6 / 0 / 0 |
| bugfix-python-phone-number | pass, 1.0, 5, 5 / 0 / 0 |
| bugfix-javascript-triangle | pass, 0.3, 4, 3 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 0.6, 5, 5 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 0.8, 10, 9 / 2 / 0 |
| bugfix-python-list-ops | pass, 1.8, 6, 5 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 0.6, 5, 5 / 0 / 0 |
| bugfix-python-scale-generator | runaway (time), 12.0, 5, 4 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 1.2, 5, 5 / 0 / 0 |
| bugfix-python-pig-latin | pass, 3.1, 8, 8 / 0 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 3.7, 8, 8 / 1 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| f4ta | 11 / 12 | 28.8 | 1.1 | 1.74, 6.61 | 220 | 49.3 | 68 / 3 / 0 | 1 |

Graph reallocations of the target in each arm's server logs (C804): an arm is marked `reallocates` where another arm's logs have none.

- f4ta: graph reallocations 71 of 72 requests (250 same size, 0 changed)

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | f4ta: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 3.24, 693 |
| 2 | 12, 5.23, 1230 |
| 3 | 12, 2.38, 508 |
| 4 | 12, 0.77, 29 |
| 5 | 11, 1.30, 100 |
| 6 | 5, 0.66, 29 |
| 7 | 4, 0.62, 17 |
| 8 | 3, 0.65, 25 |
| 9 | 1, 0.58, 18 |
| 10 | 1, 0.93, 84 |
