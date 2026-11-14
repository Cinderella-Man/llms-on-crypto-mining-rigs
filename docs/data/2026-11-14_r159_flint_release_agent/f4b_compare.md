12 scored attempts, arms f4b; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | f4b: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 1.8, 5, 4 / 0 / 0 |
| bugfix-javascript-space-age | pass, 0.5, 5, 5 / 0 / 0 |
| bugfix-python-phone-number | pass, 1.0, 6, 5 / 0 / 0 |
| bugfix-javascript-triangle | pass, 0.3, 4, 3 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 0.6, 6, 5 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 0.9, 6, 5 / 0 / 0 |
| bugfix-python-list-ops | pass, 2.1, 5, 5 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 0.6, 6, 6 / 0 / 0 |
| bugfix-python-scale-generator | pass, 7.8, 9, 8 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 0.9, 5, 5 / 0 / 0 |
| bugfix-python-pig-latin | pass, 4.0, 10, 9 / 0 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 8.7, 13, 12 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| f4b | 12 / 12 | 29.2 | 1.0 | 2.56, 6.76 | 179 | 48.2 | 72 / 0 / 0 | 0 |

Graph reallocations of the target in each arm's server logs (C804): an arm is marked `reallocates` where another arm's logs have none.

- f4b: graph reallocations 79 of 80 requests (308 same size, 0 changed)

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | f4b: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 4.16, 693 |
| 2 | 12, 2.48, 372 |
| 3 | 12, 3.71, 774 |
| 4 | 12, 2.08, 290 |
| 5 | 11, 1.04, 20 |
| 6 | 7, 1.56, 218 |
| 7 | 3, 1.92, 157 |
| 8 | 3, 3.79, 379 |
| 9 | 3, 1.28, 20 |
| 10 | 2, 2.16, 130 |
| 11 | 1, 1.19, 14 |
| 12 | 1, 1.33, 21 |
| 13 | 1, 2.55, 99 |
