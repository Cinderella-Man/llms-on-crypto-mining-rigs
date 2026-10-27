- **harness: infrastructure: GO** — 7 of 7 attempts finished without an infrastructure failure; 0 tries re-run (none), 0 attempts given up
- **harness: proxy rows: NO-GO** — 31 requests logged; attempts with a row count different from pi's replies: stock/bugfix-python-list-ops/a1/t1 (2 rows / 1 replies); rows without timings: 1
- **harness: first token vs prefill: GO** — proxy first token minus server prefill: median 0.08 s, p90 0.16 s (limit 0.5 s + round trip 0.30 s; includes the per-request fixed cost before prefill)
- **tool calls: GO** — malformed 0 of 31 calls (0.0 %: unknown tool 0, invalid arguments 0, markup in text 0); calls inside the reasoning 0
- **cache: GO** — turns ≥ 2: median cache hit 94.7 %, min 55.9 %; cache paths exact 23, older checkpoint 0, unknown 1
- **memory: NO-GO** — lowest MemAvailable 507 MB; swap used 349 → 653 MB
- server log server-stock.log: backend sampling off 31, checkpoint restored 0, checkpoint created 0, checkpoint erased 0, full re-processing 0, screened head 0, NaN / inf 0, watchdog kill 0

**Overall: NO-GO**
