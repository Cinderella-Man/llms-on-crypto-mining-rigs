10 scored attempts, arms rel; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | rel: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 2.3, 7, 7 / 1 / 0 |
| bugfix-javascript-space-age | pass, 1.1, 7, 6 / 0 / 0 |
| bugfix-python-phone-number | pass, 1.5, 5, 5 / 0 / 0 |
| bugfix-javascript-triangle | pass, 0.3, 4, 3 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 0.7, 6, 6 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 1.0, 5, 4 / 0 / 0 |
| bugfix-python-list-ops | pass, 3.6, 6, 5 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 1.2, 6, 6 / 0 / 0 |
| bugfix-python-scale-generator | runaway (time), 12.0, 6, 5 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 2.2, 11, 12 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| rel | 9 / 10 | 25.8 | 1.3 | 2.58, 7.60 | 191 | 44.6 | 59 / 1 / 0 | 1 |

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | rel: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 10, 4.30, 693 |
| 2 | 10, 5.44, 1206 |
| 3 | 10, 2.65, 438 |
| 4 | 10, 2.13, 260 |
| 5 | 9, 1.57, 143 |
| 6 | 7, 1.04, 20 |
| 7 | 3, 1.54, 132 |
| 8 | 1, 0.91, 12 |
| 9 | 1, 2.10, 194 |
| 10 | 1, 1.02, 25 |
| 11 | 1, 2.05, 137 |
