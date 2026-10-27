12 scored attempts, arms rel; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | rel: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 1.3, 6, 5 / 0 / 0 |
| bugfix-javascript-space-age | pass, 0.8, 7, 7 / 0 / 0 |
| bugfix-python-phone-number | pass, 1.6, 6, 5 / 0 / 0 |
| bugfix-javascript-triangle | pass, 0.4, 4, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 0.9, 5, 5 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 0.8, 5, 4 / 0 / 0 |
| bugfix-python-list-ops | pass, 2.0, 5, 5 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 0.8, 6, 6 / 0 / 0 |
| bugfix-python-scale-generator | pass, 7.3, 7, 6 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 1.0, 6, 6 / 0 / 0 |
| bugfix-python-pig-latin | pass, 1.1, 6, 5 / 0 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 5.6, 7, 7 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| rel | 12 / 12 | 23.7 | 1.1 | 2.77, 6.13 | 183 | 46.5 | 65 / 0 / 0 | 0 |

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | rel: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 4.08, 693 |
| 2 | 12, 5.46, 1222 |
| 3 | 12, 2.84, 546 |
| 4 | 12, 2.76, 412 |
| 5 | 11, 0.95, 20 |
| 6 | 8, 1.06, 20 |
| 7 | 3, 1.52, 102 |
