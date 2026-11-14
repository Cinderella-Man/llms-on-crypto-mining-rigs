24 scored attempts -> res/attempts.csv; 0 infrastructure tries re-run, 0 attempts given up

| arm | attempts | pass@1 | minutes per task (median, p90) | prefill / decode / fixed / other (s, mean) | turns (median) | prompt tokens processed per task | cache hit | TTFT p50 s | malformed calls | runaways |
|---|---|---|---|---|---|---|---|---|---|---|
| f3a | 12 | 91.7 % | 1.2, 2.3 | 17 / 49 / 55 / 2 | 5.0 | 3111 | 85.6 % | 2.81 | 0/57 | 1 |
| f4a | 12 | 100.0 % | 1.2, 4.5 | 34 / 120 / 1 / 2 | 5.0 | 6007 | 85.4 % | 2.76 | 0/70 | 0 |
