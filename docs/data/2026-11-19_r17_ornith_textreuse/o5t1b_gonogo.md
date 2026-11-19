- **harness: infrastructure: GO** — 12 of 12 attempts finished without an infrastructure failure; 0 tries re-run (none), 0 attempts given up
- **harness: proxy rows: NO-GO** — 55 requests logged; attempts with a row count different from pi's replies: o5t1b/bugfix-python-scale-generator/a1/t1 (3 rows / 2 replies); rows without timings: 1
- **harness: first token vs prefill: GO** — proxy first token minus server prefill: median 0.07 s, p90 0.16 s (limit 0.5 s + round trip 0.30 s; includes the per-request fixed cost before prefill)
- **tool calls: GO** — malformed 0 of 55 calls (0.0 %: unknown tool 0, invalid arguments 0, markup in text 0); calls inside the reasoning 0
- **cache: GO** — turns ≥ 2: median cache hit 94.8 %, min 44.6 %; cache paths exact 42, older checkpoint 0, unknown 1
- **memory: NO-GO** — lowest MemAvailable 1227 MB; swap used 54 → 363 MB
- server log server-o5t1b.log: backend sampling off 0, checkpoint restored 11, checkpoint created 0, checkpoint erased 0, full re-processing 0, screened head 0, NaN / inf 0, watchdog kill 0, graph reallocations 0 of 54 requests (0 same size, 0 changed)

**Overall: NO-GO**
