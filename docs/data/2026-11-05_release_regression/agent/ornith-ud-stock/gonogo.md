- **harness: infrastructure: GO** — 12 of 12 attempts finished without an infrastructure failure; 0 tries re-run (none), 0 attempts given up
- **harness: proxy rows: NO-GO** — 56 requests logged; attempts with a row count different from pi's replies: stock/bugfix-python-pig-latin/a1/t1 (6 rows / 5 replies), stock/bugfix-python-scale-generator/a1/t1 (5 rows / 4 replies); rows without timings: 2
- **harness: first token vs prefill: GO** — proxy first token minus server prefill: median 0.08 s, p90 0.15 s (limit 0.5 s + round trip 0.30 s; includes the per-request fixed cost before prefill)
- **tool calls: GO** — malformed 0 of 56 calls (0.0 %: unknown tool 0, invalid arguments 0, markup in text 0); calls inside the reasoning 0
- **cache: GO** — turns ≥ 2: median cache hit 95.2 %, min 44.6 %; cache paths exact 40, older checkpoint 0, reply re-read 2, unknown 2
- **memory: NO-GO** — lowest MemAvailable 978 MB; swap used 32 → 328 MB
- server log server-stock.log: backend sampling off 56, checkpoint restored 0, checkpoint created 0, checkpoint erased 0, full re-processing 0, screened head 0, NaN / inf 0, watchdog kill 0

**Overall: NO-GO**
