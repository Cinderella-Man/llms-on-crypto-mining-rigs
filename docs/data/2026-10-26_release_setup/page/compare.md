11 scored attempts, arms rel; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | rel: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 0.8, 5, 4 / 0 / 0 |
| bugfix-javascript-space-age | pass, 1.9, 11, 10 / 1 / 0 |
| bugfix-python-phone-number | pass, 3.4, 6, 6 / 0 / 0 |
| bugfix-javascript-triangle | pass, 0.4, 4, 3 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 0.7, 5, 4 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 1.0, 5, 4 / 0 / 0 |
| bugfix-python-list-ops | pass, 0.4, 6, 5 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 0.9, 6, 6 / 0 / 0 |
| bugfix-python-scale-generator | pass, 9.7, 5, 5 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 3.6, 19, 18 / 0 / 0 |
| bugfix-python-pig-latin | pass, 3.2, 5, 5 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| rel | 11 / 11 | 25.9 | 1.0 | 2.43, 7.59 | 184 | 44.6 | 70 / 1 / 0 | 0 |

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | rel: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 11, 4.18, 702 |
| 2 | 11, 5.46, 1257 |
| 3 | 11, 2.79, 513 |
| 4 | 11, 1.70, 192 |
| 5 | 10, 0.93, 20 |
| 6 | 5, 2.89, 320 |
| 7 | 2, 4.49, 850 |
| 8 | 2, 1.83, 223 |
| 9 | 2, 1.41, 85 |
| 10 | 2, 1.46, 189 |
| 11 | 2, 1.60, 124 |
| 12 | 1, 3.43, 277 |
| 13 | 1, 0.99, 19 |
| 14 | 1, 2.17, 261 |
| 15 | 1, 1.98, 199 |
| 16 | 1, 2.33, 164 |
| 17 | 1, 1.04, 19 |
| 18 | 1, 2.08, 235 |
| 19 | 1, 2.13, 283 |
