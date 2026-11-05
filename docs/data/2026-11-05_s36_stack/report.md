36 scored attempts, arms s35stack, s36adapt, s36ref; 1 infrastructure tries re-run, 0 attempts given up; per-request rows in res/turns.csv

### Per task

Minutes from pi's start to its exit; tool calls: all / failed (the tool returned an error) / malformed (unknown tool, arguments that fail the schema, or call markup in the text).

| task | s35stack: result, min, turns, tool calls | s36adapt: result, min, turns, tool calls | s36ref: result, min, turns, tool calls | time s35stack / s36adapt | time s35stack / s36ref |
|---|---|---|---|---|---|
| bugfix-python-affine-cipher | pass, 3.8, 5, 5 / 0 / 0 | pass, 2.8, 6, 6 / 1 / 0 | pass, 2.7, 5, 5 / 0 / 0 | 1.37× | 1.38× |
| bugfix-javascript-space-age | pass, 1.9, 5, 6 / 0 / 0 | pass, 1.8, 5, 4 / 0 / 0 | pass, 1.3, 5, 6 / 0 / 0 | 1.10× | 1.51× |
| bugfix-python-phone-number | pass, 2.6, 6, 6 / 0 / 0 | pass, 1.7, 6, 6 / 0 / 0 | pass, 1.7, 6, 6 / 1 / 0 | 1.57× | 1.53× |
| bugfix-javascript-triangle | pass, 1.4, 5, 4 / 0 / 0 | pass, 1.0, 5, 4 / 0 / 0 | pass, 1.1, 5, 4 / 0 / 0 | 1.39× | 1.27× |
| bugfix-python-variable-length-quantity | pass, 7.5, 6, 6 / 0 / 0 | pass, 2.6, 5, 5 / 0 / 0 | pass, 2.9, 6, 6 / 1 / 0 | 2.91× | 2.57× |
| bugfix-javascript-phone-number | pass, 1.7, 5, 4 / 0 / 0 | pass, 1.4, 5, 4 / 0 / 0 | pass, 1.4, 5, 5 / 0 / 0 | 1.24× | 1.24× |
| bugfix-python-list-ops | pass, 2.0, 5, 5 / 0 / 0 | pass, 2.1, 5, 5 / 0 / 0 | pass, 2.3, 5, 5 / 0 / 0 | 0.94× | 0.84× |
| bugfix-javascript-grade-school | pass, 1.2, 5, 4 / 0 / 0 | pass, 1.1, 4, 3 / 0 / 0 | pass, 1.2, 5, 4 / 0 / 0 | 1.12× | 1.02× |
| bugfix-python-scale-generator | pass, 4.1, 5, 5 / 0 / 0 | pass, 3.0, 5, 5 / 0 / 0 | pass, 5.7, 7, 7 / 0 / 0 | 1.34× | 0.72× |
| bugfix-javascript-complex-numbers | pass, 1.4, 6, 6 / 1 / 0 | pass, 1.4, 5, 5 / 0 / 0 | pass, 1.7, 5, 5 / 0 / 0 | 1.01× | 0.81× |
| bugfix-python-pig-latin | pass, 4.7, 6, 6 / 0 / 0 | runaway (time), 12.0, 11, 11 / 3 / 0 | pass, 3.3, 7, 6 / 0 / 0 | 0.39× | 1.41× |
| bugfix-javascript-variable-length-quantity | pass, 2.5, 5, 5 / 0 / 0 | pass, 4.3, 6, 6 / 0 / 0 | pass, 4.1, 5, 6 / 0 / 0 | 0.57× | 0.60× |

### Per arm

| arm | tasks passed | minutes, all tasks | minutes per task (median) | TTFT per turn s (median, p90) | prompt tok/s | generation tok/s | tool calls / failed / malformed | runaways |
|---|---|---|---|---|---|---|---|---|
| s35stack | 12 / 12 | 34.8 | 2.2 | 7.42, 17.34 | 68 | 18.8 | 62 / 1 / 0 | 0 |
| s36adapt | 12 / 12 | 35.1 | 1.9 | 6.93, 15.81 | 70 | 18.1 | 64 / 4 / 0 | 1 |
| s36ref | 12 / 12 | 29.5 | 2.0 | 6.54, 15.73 | 69 | 19.0 | 65 / 2 / 0 | 0 |

Tasks scored on every arm: 12.
- s36adapt against s35stack: 35.1 against 34.8 minutes in all (0.99× faster), per task 1.12× (geometric mean, range 0.39–2.91); passed 12 against 12
- s36ref against s35stack: 29.5 against 34.8 minutes in all (1.18× faster), per task 1.15× (geometric mean, range 0.60–2.57); passed 12 against 12

### Time to first token per turn

Median over tasks of the seconds from the request to its first token as the client sees it (network included), and of the prompt tokens the server processed for it (the rest came from its cache).

| turn | s35stack: tasks, TTFT s, new prompt tokens | s36adapt: tasks, TTFT s, new prompt tokens | s36ref: tasks, TTFT s, new prompt tokens |
|---|---|---|---|
| 1 | 12, 13.73, 789 | 12, 13.58, 789 | 12, 13.66, 789 |
| 2 | 12, 15.61, 1578 | 12, 15.69, 1550 | 12, 15.65, 1608 |
| 3 | 12, 7.12, 319 | 12, 7.35, 404 | 12, 7.73, 381 |
| 4 | 12, 2.71, 30 | 12, 2.69, 30 | 12, 2.63, 32 |
| 5 | 12, 3.43, 90 | 11, 4.02, 109 | 12, 3.57, 90 |
| 6 | 4, 4.89, 180 | 4, 3.49, 82 | 4, 3.58, 96 |
| 7 | 0, —, — | 1, 5.46, 204 | 2, 4.58, 144 |
| 8 | 0, —, — | 1, 2.93, 32 | 0, —, — |
| 9 | 0, —, — | 1, 8.11, 334 | 0, —, — |
| 10 | 0, —, — | 1, 3.01, 32 | 0, —, — |
| 11 | 0, —, — | 1, —, — | 0, —, — |

## Per attempt (table)

36 scored attempts -> res/attempts.csv; 1 infrastructure tries re-run, 0 attempts given up

| arm | attempts | pass@1 | minutes per task (median, p90) | prefill / decode / fixed / other (s, mean) | turns (median) | prompt tokens processed per task | cache hit | TTFT p50 s | malformed calls | runaways |
|---|---|---|---|---|---|---|---|---|---|---|
| s35stack | 12 | 100.0 % | 2.2, 4.7 | 47 / 101 / 2 / 25 | 5.0 | 3154 | 87.3 % | 6.97 | 0/62 | 0 |
| s36adapt | 12 | 100.0 % | 1.9, 4.3 | 49 / 120 / 1 / 6 | 5.0 | 3393 | 87.5 % | 6.73 | 0/64 | 1 |
| s36ref | 12 | 100.0 % | 2.0, 4.1 | 47 / 96 / 0 / 4 | 5.0 | 3236 | 87.3 % | 6.11 | 0/65 | 0 |

## Go / no-go checks

- **harness: infrastructure: GO** — 35 of 36 attempts finished without an infrastructure failure; 1 tries re-run (s35stack/bugfix-python-list-ops/a1-t1-infra), 0 attempts given up
- **harness: proxy rows: NO-GO** — 198 requests logged; attempts with a row count different from pi's replies: s36adapt/bugfix-python-pig-latin/a1/t1 (11 rows / 10 replies); rows without timings: 1
- **harness: first token vs prefill: GO** — proxy first token minus server prefill: median 0.07 s, p90 0.13 s (limit 0.5 s + round trip 0.30 s; includes the per-request fixed cost before prefill)
- **tool calls: GO** — malformed 0 of 191 calls (0.0 %: unknown tool 0, invalid arguments 0, markup in text 0); calls inside the reasoning 0
- **cache: GO** — turns ≥ 2: median cache hit 96.9 %, min 39.4 %; cache paths exact 154, older checkpoint 0, reply re-read 7, unknown 1
- **memory: NO-GO** — lowest MemAvailable 88 MB; swap used 34 → 1091 MB
- server log server-s35stack.log: backend sampling off 0, checkpoint restored 14, checkpoint created 0, checkpoint erased 0, full re-processing 0, screened head 0, NaN / inf 0, watchdog kill 0
- server log server-s36adapt.log: backend sampling off 0, checkpoint restored 13, checkpoint created 0, checkpoint erased 0, full re-processing 0, screened head 0, NaN / inf 0, watchdog kill 0
- server log server-s36ref.log: backend sampling off 0, checkpoint restored 14, checkpoint created 0, checkpoint erased 0, full re-processing 0, screened head 0, NaN / inf 0, watchdog kill 0

**Overall: NO-GO**
