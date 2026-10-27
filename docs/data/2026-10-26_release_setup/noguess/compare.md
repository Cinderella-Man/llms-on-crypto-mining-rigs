10 scored attempts, arms rel; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | rel: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 2.2, 6, 6 / 0 / 0 |
| bugfix-javascript-space-age | pass, 0.8, 6, 5 / 0 / 0 |
| bugfix-python-phone-number | pass, 2.4, 6, 6 / 0 / 0 |
| bugfix-javascript-triangle | pass, 0.4, 4, 3 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 0.8, 5, 4 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 1.2, 5, 5 / 0 / 0 |
| bugfix-python-list-ops | pass, 2.4, 5, 5 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 1.0, 8, 8 / 0 / 0 |
| bugfix-python-scale-generator | runaway (time), 12.0, 6, 6 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 2.2, 5, 6 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| rel | 9 / 10 | 25.5 | 1.7 | 2.33, 6.63 | 184 | 29.3 | 54 / 0 / 0 | 1 |

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | rel: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 10, 4.58, 693 |
| 2 | 10, 5.42, 1220 |
| 3 | 10, 2.24, 336 |
| 4 | 10, 1.31, 101 |
| 5 | 9, 1.04, 20 |
| 6 | 5, 1.25, 77 |
| 7 | 1, 0.84, 19 |
| 8 | 1, 1.63, 302 |
