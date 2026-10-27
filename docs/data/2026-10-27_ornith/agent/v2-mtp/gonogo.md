- **harness: infrastructure: GO** — 9 of 9 attempts finished without an infrastructure failure; 0 tries re-run (none), 0 attempts given up
- **harness: proxy rows: NO-GO** — 40 requests logged; attempts with a row count different from pi's replies: rel/bugfix-python-scale-generator/a1/t1 (3 rows / 2 replies); rows without timings: 1
- **harness: first token vs prefill: GO** — proxy first token minus server prefill: median 0.07 s, p90 0.15 s (limit 0.5 s + round trip 0.30 s; includes the per-request fixed cost before prefill)
- **tool calls: GO** — malformed 0 of 40 calls (0.0 %: unknown tool 0, invalid arguments 0, markup in text 0); calls inside the reasoning 0
- **cache: GO** — turns ≥ 2: median cache hit 93.3 %, min 53.1 %; cache paths exact 30, older checkpoint 0, unknown 1
- **memory: NO-GO** — lowest MemAvailable 421 MB; swap used 354 → 652 MB
- server log server-rel.log: backend sampling off 0, checkpoint restored 8, checkpoint created 0, checkpoint erased 0, full re-processing 0, screened head 0, NaN / inf 0, watchdog kill 0

**Overall: NO-GO**
