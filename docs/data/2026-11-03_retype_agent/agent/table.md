29 scored attempts -> res/attempts.csv; 1 infrastructure tries re-run, 0 attempts given up

| arm | attempts | pass@1 | minutes per task (median, p90) | prefill / decode / fixed / other (s, mean) | turns (median) | prompt tokens processed per task | cache hit | TTFT p50 s | malformed calls | runaways |
|---|---|---|---|---|---|---|---|---|---|---|
| hcq8 | 12 | 100.0 % | 3.7, 5.4 | 49 / 155 / 0 / 9 | 5.0 | 3148 | 86.3 % | 7.95 | 0/57 | 0 |
| pipe0 | 8 | 87.5 % | 3.2, 12.0 | 48 / 161 / 87 / 21 | 5.0 | 2943 | 86.5 % | 8.38 | 0/36 | 1 |
| rel | 9 | 100.0 % | 3.0, 15.6 | 54 / 202 / 8 / 31 | 5 | 3287 | 87.2 % | 8.69 | 0/47 | 1 |
