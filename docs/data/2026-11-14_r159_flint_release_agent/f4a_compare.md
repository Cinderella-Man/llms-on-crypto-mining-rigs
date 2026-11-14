12 scored attempts, arms f4a; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | f4a: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 1.4, 5, 5 / 0 / 0 |
| bugfix-javascript-space-age | pass, 0.7, 6, 6 / 0 / 0 |
| bugfix-python-phone-number | pass, 1.6, 7, 7 / 0 / 0 |
| bugfix-javascript-triangle | pass, 0.4, 5, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 0.7, 5, 4 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 1.0, 5, 5 / 0 / 0 |
| bugfix-python-list-ops | pass, 3.5, 5, 5 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 0.7, 5, 5 / 0 / 0 |
| bugfix-python-scale-generator | pass, 11.5, 9, 9 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 0.9, 5, 5 / 0 / 0 |
| bugfix-python-pig-latin | pass, 4.4, 11, 10 / 1 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 4.5, 5, 5 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| f4a | 12 / 12 | 31.3 | 1.2 | 2.85, 7.84 | 177 | 47.4 | 70 / 1 / 0 | 0 |

Graph reallocations of the target in each arm's server logs (C804): an arm is marked `reallocates` where another arm's logs have none.

- f4a: graph reallocations 71 of 73 requests (323 same size, 0 changed)

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | f4a: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 4.29, 693 |
| 2 | 12, 6.24, 1334 |
| 3 | 12, 2.93, 496 |
| 4 | 12, 1.15, 44 |
| 5 | 12, 1.70, 142 |
| 6 | 4, 1.87, 250 |
| 7 | 3, 1.21, 34 |
| 8 | 2, 1.52, 63 |
| 9 | 2, 14.66, 2662 |
| 10 | 1, 1.76, 106 |
| 11 | 1, 1.02, 20 |
