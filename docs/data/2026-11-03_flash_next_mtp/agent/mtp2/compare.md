20 scored attempts, arms nodraft, mtp2; 2 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | nodraft: result, min, turns, tool calls | mtp2: result, min, turns, tool calls | time nodraft / mtp2 |
|---|---|---|---|
| bugfix-python-affine-cipher | pass, 8.2, 8, 8 / 1 / 0 | pass, 4.4, 5, 5 / 0 / 0 | 1.89× |
| bugfix-javascript-space-age | pass, 4.1, 6, 5 / 0 / 0 | pass, 2.0, 5, 4 / 0 / 0 | 2.06× |
| bugfix-python-phone-number | pass, 3.0, 5, 5 / 0 / 0 | pass, 2.7, 5, 5 / 0 / 0 | 1.13× |
| bugfix-javascript-triangle | pass, 2.4, 5, 4 / 0 / 0 | pass, 1.7, 5, 4 / 0 / 0 | 1.42× |
| bugfix-python-variable-length-quantity | pass, 6.9, 7, 7 / 1 / 0 | pass, 3.1, 6, 6 / 0 / 0 | 2.28× |
| bugfix-javascript-phone-number | pass, 2.5, 5, 4 / 0 / 0 | pass, 2.6, 5, 5 / 0 / 0 | 0.97× |
| bugfix-python-list-ops | pass, 4.3, 5, 5 / 0 / 0 | pass, 4.6, 5, 5 / 0 / 0 | 0.94× |
| bugfix-javascript-grade-school | pass, 2.8, 5, 4 / 0 / 0 | pass, 1.9, 5, 4 / 0 / 0 | 1.49× |
| bugfix-python-scale-generator | pass, 8.7, 5, 5 / 0 / 0 | pass, 6.2, 5, 5 / 0 / 0 | 1.40× |
| bugfix-javascript-complex-numbers | — | pass, 2.7, 5, 5 / 0 / 0 | — |
| bugfix-python-pig-latin | — | runaway (time), 12.0, 14, 13 / 1 / 0 | — |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| nodraft | 9 / 9 | 43.0 | 4.1 | 8.70, 19.01 | 59 | 7.8 | 47 / 2 / 0 | 0 |
| mtp2 | 10 / 11 | 43.7 | 2.7 | 7.85, 21.09 | 55 | 8.8 | 61 / 1 / 0 | 1 |

Tasks scored on every arm: 9.
- mtp2 against nodraft: 29.0 against 43.0 minutes in all (1.48× faster), per task 1.44× (geometric mean, range 0.94–2.28); passed 9 against 9

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | nodraft: tasks, TTFT s, new prompt tokens | mtp2: tasks, TTFT s, new prompt tokens |
|---|---|---|
| 1 | 9, 15.76, 803 | 11, 16.99, 803 |
| 2 | 9, 18.44, 1566 | 11, 20.39, 1649 |
| 3 | 9, 10.15, 437 | 11, 8.22, 304 |
| 4 | 9, 3.60, 31 | 11, 3.19, 31 |
| 5 | 9, 4.34, 91 | 11, 4.20, 84 |
| 6 | 3, 4.59, 104 | 2, 12.48, 634 |
| 7 | 2, 9.10, 340 | 1, 7.61, 276 |
| 8 | 1, 3.51, 42 | 1, 6.41, 185 |
| 9 | 0, —, — | 1, 3.30, 50 |
| 10 | 0, —, — | 1, 6.72, 217 |
| 11 | 0, —, — | 1, 4.08, 87 |
| 12 | 0, —, — | 1, 18.74, 810 |
| 13 | 0, —, — | 1, 5.73, 32 |
