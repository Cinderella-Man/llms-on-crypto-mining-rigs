16 scored attempts -> res/attempts.csv; 1 infrastructure tries re-run, 0 attempts given up

| arm | attempts | pass@1 | minutes per task (median, p90) | prefill / decode / fixed / other (s, mean) | turns (median) | prompt tokens processed per task | cache hit | TTFT p50 s | malformed calls | runaways |
|---|---|---|---|---|---|---|---|---|---|---|
| pipe0 | 8 | 87.5 % | 3.2, 12.0 | 48 / 161 / 87 / 21 | 5.0 | 2943 | 86.5 % | 8.38 | 0/36 | 1 |
| pipe3b | 8 | 100.0 % | 3.7, 11.5 | 52 / 249 / 1 / 7 | 5.0 | 3139 | 85.7 % | 8.25 | 0/37 | 0 |
