11 scored attempts, arms stock; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | stock: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 4.4, 6, 5 / 0 / 0 |
| bugfix-javascript-space-age | pass, 2.4, 8, 7 / 0 / 0 |
| bugfix-python-phone-number | pass, 6.9, 5, 5 / 0 / 0 |
| bugfix-javascript-triangle | pass, 0.6, 4, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 1.5, 5, 4 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 1.7, 5, 5 / 0 / 0 |
| bugfix-python-list-ops | pass, 1.3, 5, 5 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 2.1, 9, 9 / 0 / 0 |
| bugfix-python-scale-generator | runaway (time), 12.0, 5, 5 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 0.7, 4, 4 / 0 / 0 |
| bugfix-python-pig-latin | pass, 5.1, 6, 5 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| stock | 10 / 11 | 38.7 | 2.1 | 3.85, 8.38 | 143 | 15.6 | 58 / 0 / 0 | 1 |

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | stock: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 11, 6.06, 704 |
| 2 | 11, 7.58, 1187 |
| 3 | 11, 4.09, 336 |
| 4 | 11, 3.58, 371 |
| 5 | 9, 1.22, 20 |
| 6 | 4, 2.09, 96 |
| 7 | 2, 2.22, 162 |
| 8 | 2, 1.77, 77 |
| 9 | 1, 3.43, 323 |
