12 scored attempts, arms rel; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | rel: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 3.3, 5, 4 / 0 / 0 |
| bugfix-javascript-space-age | pass, 0.8, 8, 8 / 0 / 0 |
| bugfix-python-phone-number | pass, 1.9, 9, 8 / 0 / 0 |
| bugfix-javascript-triangle | pass, 0.4, 5, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 0.6, 6, 5 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 0.8, 5, 5 / 0 / 0 |
| bugfix-python-list-ops | pass, 4.1, 5, 4 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 0.7, 7, 6 / 0 / 0 |
| bugfix-python-scale-generator | pass, 5.6, 6, 5 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 8.9, 31, 31 / 1 / 0 |
| bugfix-python-pig-latin | pass, 1.4, 5, 4 / 0 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 3.2, 7, 6 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| rel | 12 / 12 | 31.7 | 1.7 | 3.02, 7.94 | 168 | 49.4 | 90 / 1 / 0 | 0 |

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | rel: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 4.74, 693 |
| 2 | 12, 4.78, 1016 |
| 3 | 12, 2.82, 380 |
| 4 | 12, 2.69, 414 |
| 5 | 12, 1.48, 88 |
| 6 | 7, 1.06, 21 |
| 7 | 5, 1.70, 99 |
| 8 | 3, 1.84, 135 |
| 9 | 2, 2.63, 282 |
| 10 | 1, 3.33, 336 |
| 11 | 1, 18.05, 2957 |
| 12 | 1, 5.46, 541 |
| 13 | 1, 2.38, 250 |
| 14 | 1, 3.95, 398 |
| 15 | 1, 2.72, 237 |
| 16 | 1, 2.44, 184 |
| 17 | 1, 1.13, 15 |
| 18 | 1, 4.74, 639 |
| 19 | 1, 11.49, 1428 |
| 20 | 1, 2.44, 120 |
| 21 | 1, 1.53, 36 |
| 22 | 1, 10.14, 1060 |
| 23 | 1, 2.67, 125 |
| 24 | 1, 6.35, 623 |
| 25 | 1, 3.31, 161 |
| 26 | 1, 4.42, 290 |
| 27 | 1, 6.97, 721 |
| 28 | 1, 5.19, 449 |
| 29 | 1, 1.44, 19 |
| 30 | 1, 4.62, 288 |
| 31 | 1, 5.09, 350 |
