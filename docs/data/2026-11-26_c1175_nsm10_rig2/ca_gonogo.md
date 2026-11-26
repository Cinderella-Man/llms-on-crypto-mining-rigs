- **harness: infrastructure: GO** — 12 of 12 attempts finished without an infrastructure failure; 0 tries re-run (none), 0 attempts given up
- **harness: proxy rows: NO-GO** — 85 requests logged; attempts with a row count different from pi's replies: v4r/bugfix-javascript-complex-numbers/a1/t1 (25 rows / 24 replies); rows without timings: 1
- **harness: first token vs prefill: GO** — proxy first token minus server prefill: median 0.07 s, p90 0.15 s (limit 0.5 s + round trip 0.30 s; includes the per-request fixed cost before prefill)
- **tool calls: GO** — malformed 0 of 80 calls (0.0 %: unknown tool 0, invalid arguments 0, markup in text 0); calls inside the reasoning 0
- **cache: GO** — turns ≥ 2: median cache hit 97.4 %, min 48.4 %; cache paths exact 72, older checkpoint 0, unknown 1
- **memory: NO-GO** — lowest MemAvailable 295 MB; swap used 52 → 998 MB
- server log server-v4r.log: backend sampling off 0, checkpoint restored 10, checkpoint created 0, checkpoint erased 2, full re-processing 1, screened head 0, NaN / inf 0, watchdog kill 0, graph reallocations 0 of 84 requests (0 same size, 0 changed)

**Overall: NO-GO**
