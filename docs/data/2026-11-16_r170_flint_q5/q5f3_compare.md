12 scored attempts, arms q5f3; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | q5f3: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 1.8, 5, 4 / 0 / 0 |
| bugfix-javascript-space-age | pass, 0.7, 8, 7 / 0 / 0 |
| bugfix-python-phone-number | pass, 1.0, 5, 5 / 0 / 0 |
| bugfix-javascript-triangle | pass, 0.3, 4, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 0.6, 5, 4 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 0.6, 5, 5 / 0 / 0 |
| bugfix-python-list-ops | pass, 2.0, 6, 5 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 0.6, 5, 5 / 0 / 0 |
| bugfix-python-scale-generator | pass, 7.3, 8, 8 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 0.9, 6, 5 / 0 / 0 |
| bugfix-python-pig-latin | pass, 1.5, 6, 6 / 0 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 2.8, 7, 6 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| q5f3 | 12 / 12 | 20.1 | 0.9 | 2.81, 8.84 | 169 | 51.8 | 64 / 0 / 0 | 0 |

Graph reallocations of the target in each arm's server logs (C804): an arm is marked `reallocates` where another arm's logs have none.

- q5f3: graph reallocations 69 of 70 requests (242 same size, 0 changed)

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | q5f3: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 4.27, 693 |
| 2 | 12, 5.56, 1126 |
| 3 | 12, 3.26, 528 |
| 4 | 12, 1.30, 104 |
| 5 | 11, 1.12, 20 |
| 6 | 6, 1.10, 22 |
| 7 | 3, 1.27, 19 |
| 8 | 2, 1.61, 78 |
