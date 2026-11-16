12 scored attempts, arms f4tb; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | f4tb: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 1.7, 6, 5 / 0 / 0 |
| bugfix-javascript-space-age | pass, 1.0, 8, 8 / 1 / 0 |
| bugfix-python-phone-number | pass, 1.6, 8, 8 / 0 / 0 |
| bugfix-javascript-triangle | pass, 0.3, 4, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 0.9, 6, 6 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 0.8, 5, 5 / 0 / 0 |
| bugfix-python-list-ops | pass, 1.2, 6, 5 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 0.6, 5, 5 / 0 / 0 |
| bugfix-python-scale-generator | pass, 10.2, 6, 5 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 1.0, 6, 6 / 0 / 0 |
| bugfix-python-pig-latin | pass, 1.7, 5, 5 / 0 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 2.2, 6, 6 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| f4tb | 12 / 12 | 23.3 | 1.1 | 1.82, 6.35 | 204 | 47.0 | 68 / 1 / 0 | 0 |

Graph reallocations of the target in each arm's server logs (C804): an arm is marked `reallocates` where another arm's logs have none.

- f4tb: graph reallocations 70 of 71 requests (290 same size, 0 changed)

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | f4tb: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 3.18, 693 |
| 2 | 12, 4.23, 1013 |
| 3 | 12, 3.29, 676 |
| 4 | 12, 1.82, 394 |
| 5 | 11, 0.90, 20 |
| 6 | 8, 1.10, 58 |
| 7 | 2, 0.56, 16 |
| 8 | 2, 0.82, 76 |
