12 scored attempts, arms f3a; 0 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | f3a: result, min, turns, tool calls |
|---|---|
| bugfix-python-affine-cipher | pass, 1.7, 7, 6 / 0 / 0 |
| bugfix-javascript-space-age | pass, 0.4, 4, 4 / 0 / 0 |
| bugfix-python-phone-number | pass, 1.2, 6, 5 / 0 / 0 |
| bugfix-javascript-triangle | pass, 0.4, 4, 4 / 0 / 0 |
| bugfix-python-variable-length-quantity | pass, 0.7, 4, 3 / 0 / 0 |
| bugfix-javascript-phone-number | pass, 0.8, 5, 4 / 0 / 0 |
| bugfix-python-list-ops | pass, 1.7, 6, 5 / 0 / 0 |
| bugfix-javascript-grade-school | pass, 1.1, 10, 9 / 0 / 0 |
| bugfix-python-scale-generator | runaway (time), 12.0, 4, 3 / 0 / 0 |
| bugfix-javascript-complex-numbers | pass, 1.1, 5, 4 / 0 / 0 |
| bugfix-python-pig-latin | pass, 2.3, 6, 5 / 0 / 0 |
| bugfix-javascript-variable-length-quantity | pass, 1.3, 5, 5 / 0 / 0 |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| f3a | 11 / 12 | 24.7 | 1.2 | 2.65, 7.20 | 179 | 48.5 | 57 / 0 / 0 | 1 |

Graph reallocations of the target in each arm's server logs (C804): an arm is marked `reallocates` where another arm's logs have none.

- f3a: graph reallocations 64 of 65 requests (188 same size, 0 changed)

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | f3a: tasks, TTFT s, new prompt tokens |
|---|---|
| 1 | 12, 4.17, 693 |
| 2 | 12, 5.58, 1196 |
| 3 | 12, 2.76, 463 |
| 4 | 12, 2.22, 193 |
| 5 | 8, 1.85, 190 |
| 6 | 5, 1.04, 20 |
| 7 | 2, 1.43, 59 |
| 8 | 1, 3.13, 473 |
| 9 | 1, 0.91, 19 |
| 10 | 1, 1.73, 245 |
