24 scored attempts -> res/attempts.csv; 1 infrastructure tries re-run, 0 attempts given up

| arm | attempts | pass@1 | minutes per task (median, p90) | prefill / decode / fixed / other (s, mean) | turns (median) | prompt tokens processed per task | cache hit | TTFT p50 s | malformed calls | runaways |
|---|---|---|---|---|---|---|---|---|---|---|
| gains | 12 | 100.0 % | 2.3, 3.9 | 47 / 74 / 2 / 20 | 5.0 | 3491 | 86.9 % | 5.08 | 0/63 | 0 |
| v3 | 12 | 91.7 % | 2.5, 4.1 | 44 / 74 / 59 / 8 | 5.5 | 2887 | 86.8 % | 6.97 | 0/62 | 1 |
