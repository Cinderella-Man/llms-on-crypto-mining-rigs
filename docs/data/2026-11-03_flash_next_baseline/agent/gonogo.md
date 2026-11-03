- **harness: infrastructure: GO** — 16 of 17 attempts finished without an infrastructure failure; 1 tries re-run (rel/bugfix-python-scale-generator/a1-t1-infra), 0 attempts given up
- **harness: proxy rows: NO-GO** — 87 requests logged; attempts with a row count different from pi's replies: stock/bugfix-python-variable-length-quantity/a1/t1 (3 rows / 2 replies); rows without timings: 1
- **harness: first token vs prefill: NO-GO** — proxy first token minus server prefill: median 0.11 s, p90 1.88 s (limit 0.5 s + round trip 0.30 s; includes the per-request fixed cost before prefill)
- **tool calls: GO** — malformed 0 of 80 calls (0.0 %: unknown tool 0, invalid arguments 0, markup in text 0); calls inside the reasoning 0
- **cache: GO** — turns ≥ 2: median cache hit 96.6 %, min 49.5 %; cache paths exact 67, older checkpoint 0, reply re-read 2, unknown 1
- **memory: NO-GO** — lowest MemAvailable 228 MB; swap used 356 → 974 MB
- server log server-rel.log: backend sampling off 0, checkpoint restored 11, checkpoint created 0, checkpoint erased 0, full re-processing 0, screened head 0, NaN / inf 0, watchdog kill 0
- server log server-stock.log: backend sampling off 37, checkpoint restored 0, checkpoint created 0, checkpoint erased 0, full re-processing 0, screened head 0, NaN / inf 0, watchdog kill 0

**Overall: NO-GO**
