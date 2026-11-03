17 scored attempts, arms stock, rel; 1 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | stock: result, min, turns, tool calls | rel: result, min, turns, tool calls | time stock / rel |
|---|---|---|---|
| bugfix-python-affine-cipher | pass, 5.9, 5, 4 / 0 / 0 | pass, 5.8, 6, 6 / 0 / 0 | 1.02× |
| bugfix-javascript-space-age | pass, 3.0, 5, 4 / 0 / 0 | pass, 2.4, 5, 4 / 0 / 0 | 1.25× |
| bugfix-python-phone-number | pass, 3.2, 4, 4 / 0 / 0 | pass, 2.5, 6, 5 / 0 / 0 | 1.24× |
| bugfix-javascript-triangle | pass, 2.7, 5, 4 / 0 / 0 | pass, 1.9, 5, 4 / 0 / 0 | 1.47× |
| bugfix-python-variable-length-quantity | runaway (time), 12.2, 3, 4 / 0 / 0 | runaway (time), 15.6, 5, 5 / 0 / 0 | 0.78× |
| bugfix-javascript-phone-number | pass, 3.4, 5, 4 / 0 / 0 | pass, 2.8, 5, 6 / 0 / 0 | 1.21× |
| bugfix-python-list-ops | pass, 7.0, 5, 5 / 0 / 0 | pass, 4.5, 5, 5 / 0 / 0 | 1.55× |
| bugfix-javascript-grade-school | pass, 2.2, 5, 4 / 0 / 0 | pass, 3.0, 7, 6 / 0 / 0 | 0.75× |
| bugfix-python-scale-generator | — | pass, 5.6, 6, 6 / 0 / 0 | — |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| stock | 7 / 8 | 39.6 | 3.3 | 9.17, 23.22 | 53 | 6.4 | 33 / 0 / 0 | 1 |
| rel | 9 / 9 | 44.2 | 3.0 | 7.65, 19.80 | 61 | 7.8 | 47 / 0 / 0 | 1 |

Tasks scored on every arm: 8.
- rel against stock: 38.6 against 39.6 minutes in all (1.03× faster), per task 1.12× (geometric mean, range 0.75–1.55); passed 8 against 7

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | stock: tasks, TTFT s, new prompt tokens | rel: tasks, TTFT s, new prompt tokens |
|---|---|---|
| 1 | 8, 16.78, 706 | 9, 15.75, 803 |
| 2 | 8, 22.08, 1591 | 9, 17.73, 1533 |
| 3 | 8, 8.41, 290 | 9, 9.70, 411 |
| 4 | 7, 2.97, 30 | 9, 3.94, 31 |
| 5 | 6, 5.11, 100 | 9, 5.07, 84 |
| 6 | 0, —, — | 4, 5.19, 84 |
| 7 | 0, —, — | 1, 5.28, 186 |
