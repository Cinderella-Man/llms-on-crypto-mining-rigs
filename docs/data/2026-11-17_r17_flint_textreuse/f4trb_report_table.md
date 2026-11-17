12 scored attempts -> res/attempts.csv; 0 infrastructure tries re-run, 0 attempts given up

| arm | attempts | pass@1 | minutes per task (median, p90) | prefill / decode / fixed / other (s, mean) | turns (median) | prompt tokens processed per task | cache hit | TTFT p50 s | malformed calls | runaways | prompt tokens read | reply tokens re-read (turns; their prompt s of all prompt s) | longest first-token wait s (task, turn) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| f4tr | 12 | 100.0 % | 1.1, 2.9 | 14 / 78 / 1 / 2 | 6.0 | 2942 | 89.4 % | 1.75 | 0/64 | 0 | 35309 | 0 (0; 0.0 of 170.1) | 12.57 (bugfix-javascript-complex-numbers, 2) |
