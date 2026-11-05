93 scored attempts -> res/attempts.csv; 0 infrastructure tries re-run, 0 attempts given up

| arm | attempts | pass@1 | minutes per task (median, p90) | prefill / decode / fixed / other (s, mean) | turns (median) | prompt tokens processed per task | cache hit | TTFT p50 s | malformed calls | runaways |
|---|---|---|---|---|---|---|---|---|---|---|
| s35ka | 12 | 100.0 % | 1.8, 3.9 | 49 / 92 / 1 / 6 | 5.0 | 3811 | 85.9 % | 6.46 | 0/59 | 0 |
| s35kaidle | 12 | 83.3 % | 1.6, 12.0 | 80 / 68 / 54 / 2 | 5.0 | 2868 | 81.0 % | 6.89 | 0/56 | 2 |
| s35kaoff | 12 | 100.0 % | 2.3, 4.6 | 46 / 98 / 1 / 5 | 5.0 | 3200 | 87.1 % | 7.03 | 0/63 | 0 |
| s35kareq | 12 | 100.0 % | 1.8, 2.9 | 48 / 79 / 1 / 3 | 5.0 | 3462 | 86.8 % | 6.81 | 0/64 | 0 |
| stockq2 | 9 | 100.0 % | 2.8, 10.5 | 72 / 240 / 0 / 2 | 5 | 3874 | 85.2 % | 10.22 | 0/43 | 0 |
| t1base | 12 | 100.0 % | 1.9, 3.1 | 46 / 96 / 1 / 3 | 5.0 | 3203 | 87.2 % | 7.69 | 0/62 | 0 |
| t1neg | 12 | 100.0 % | 1.9, 3.1 | 47 / 121 / 0 / 4 | 5.0 | 3271 | 86.5 % | 6.89 | 0/63 | 1 |
| t1skip | 12 | 91.7 % | 1.7, 6.6 | 41 / 85 / 59 / 3 | 5.0 | 2957 | 84.7 % | 6.27 | 0/55 | 1 |
