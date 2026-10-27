12 scored attempts, arms rel; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | rel: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 3.4, 7, 7 / 0 / 0 |
| bugfix-javascript-space-age | pass, 0.8, 6, 6 / 0 / 0 |
| bugfix-python-phone-number | pass, 2.1, 7, 6 / 0 / 0 |
| bugfix-javascript-triangle | pass, 0.3, 4, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 0.7, 5, 5 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 0.9, 6, 6 / 0 / 0 |
| bugfix-python-list-ops | pass, 1.1, 6, 5 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 0.9, 6, 6 / 0 / 0 |
| bugfix-python-scale-generator | pass, 8.9, 12, 11 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 1.0, 5, 4 / 0 / 0 |
| bugfix-python-pig-latin | pass, 3.0, 11, 10 / 1 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 3.5, 6, 6 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| rel | 12 / 12 | 26.6 | 1.1 | 2.61, 6.18 | 176 | 45.9 | 76 / 1 / 0 | 0 |

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | rel: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 4.23, 693 |
| 2 | 12, 3.89, 774 |
| 3 | 12, 3.26, 596 |
| 4 | 12, 2.21, 328 |
| 5 | 11, 0.98, 19 |
| 6 | 9, 1.69, 134 |
| 7 | 4, 1.53, 94 |
| 8 | 2, 2.39, 185 |
| 9 | 2, 30.32, 4618 |
| 10 | 2, 1.73, 124 |
| 11 | 2, 2.04, 202 |
| 12 | 1, 1.00, 20 |
