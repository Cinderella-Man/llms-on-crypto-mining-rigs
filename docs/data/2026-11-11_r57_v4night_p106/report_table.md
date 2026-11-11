24 scored attempts -> res/attempts.csv; 0 infrastructure tries re-run, 0 attempts given up

| arm | attempts | pass@1 | minutes per task (median, p90) | prefill / decode / fixed / other (s, mean) | turns (median) | prompt tokens processed per task | cache hit | TTFT p50 s | malformed calls | runaways |
|---|---|---|---|---|---|---|---|---|---|---|
| v3rel | 12 | 91.7 % | 2.0, 3.5 | 43 / 61 / 59 / 2 | 5.0 | 2950 | 85.2 % | 7.68 | 0/53 | 1 |
| v4night | 12 | 100.0 % | 2.1, 4.1 | 45 / 128 / 3 / 2 | 5.5 | 3153 | 87.8 % | 6.94 | 0/61 | 1 |
