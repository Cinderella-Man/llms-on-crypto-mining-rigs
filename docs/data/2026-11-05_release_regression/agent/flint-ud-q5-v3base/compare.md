12 scored attempts, arms rel; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | rel: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 0.9, 7, 6 / 0 / 0 |
| bugfix-javascript-space-age | pass, 0.9, 7, 6 / 0 / 0 |
| bugfix-python-phone-number | pass, 3.2, 6, 6 / 0 / 0 |
| bugfix-javascript-triangle | pass, 0.3, 4, 3 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 0.8, 5, 4 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 1.1, 5, 5 / 0 / 0 |
| bugfix-python-list-ops | pass, 1.1, 5, 4 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 0.9, 5, 5 / 0 / 0 |
| bugfix-python-scale-generator | pass, 6.0, 9, 9 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 0.9, 5, 5 / 0 / 0 |
| bugfix-python-pig-latin | pass, 4.3, 5, 5 / 0 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 4.4, 7, 6 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| rel | 12 / 12 | 24.5 | 1.0 | 2.68, 8.32 | 182 | 46.4 | 64 / 0 / 0 | 0 |

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | rel: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 4.60, 693 |
| 2 | 12, 5.08, 1014 |
| 3 | 12, 2.88, 522 |
| 4 | 12, 1.05, 20 |
| 5 | 11, 1.68, 149 |
| 6 | 5, 1.10, 21 |
| 7 | 4, 2.09, 221 |
| 8 | 1, 1.17, 19 |
| 9 | 1, 1.20, 20 |
