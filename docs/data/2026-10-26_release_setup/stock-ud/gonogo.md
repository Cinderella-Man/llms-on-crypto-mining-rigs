- **harness: infrastructure: GO** — 9 of 9 attempts finished without an infrastructure failure; 0 tries re-run (none), 0 attempts given up
- **harness: proxy rows: NO-GO** — 52 requests logged; attempts with a row count different from pi's replies: stock/bugfix-python-scale-generator/a1/t1 (3 rows / 2 replies); rows without timings: 1
- **harness: first token vs prefill: GO** — proxy first token minus server prefill: median 0.21 s, p90 0.28 s (limit 0.5 s + round trip 0.30 s; includes the per-request fixed cost before prefill)
- **tool calls: GO** — malformed 0 of 48 calls (0.0 %: unknown tool 0, invalid arguments 0, markup in text 0); calls inside the reasoning 0
- **cache: GO** — turns ≥ 2: median cache hit 95.0 %, min 42.4 %; cache paths exact 40, older checkpoint 0, reply re-read 2, unknown 1
- **memory: NO-GO** — lowest MemAvailable 947 MB; swap used 367 → 624 MB
- server log server-stock.log: backend sampling off 52, checkpoint restored 0, checkpoint created 0, checkpoint erased 0, full re-processing 0, screened head 0, NaN / inf 0, watchdog kill 0

**Overall: NO-GO**
