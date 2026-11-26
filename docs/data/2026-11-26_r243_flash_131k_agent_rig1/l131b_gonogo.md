- **harness: infrastructure: GO** — 12 of 12 attempts finished without an infrastructure failure; 0 tries re-run (none), 0 attempts given up
- **harness: proxy rows: NO-GO** — 86 requests logged; attempts with a row count different from pi's replies: v4r/bugfix-javascript-complex-numbers/a1/t1 (30 rows / 29 replies); rows without timings: 1
- **harness: first token vs prefill: GO** — proxy first token minus server prefill: median 0.06 s, p90 0.15 s (limit 0.5 s + round trip 0.30 s; includes the per-request fixed cost before prefill)
- **tool calls: GO** — malformed 0 of 80 calls (0.0 %: unknown tool 0, invalid arguments 0, markup in text 0); calls inside the reasoning 0
- **cache: GO** — turns ≥ 2: median cache hit 97.5 %, min 48.3 %; cache paths exact 73, older checkpoint 0, unknown 1
- **memory: NO-GO** — lowest MemAvailable 232 MB; swap used 22 → 22 MB
- server log server-v4r.log: backend sampling off 0, checkpoint restored 10, checkpoint created 0, checkpoint erased 1, full re-processing 1, screened head 0, NaN / inf 0, watchdog kill 0, graph reallocations 0 of 85 requests (0 same size, 0 changed)

**Overall: NO-GO**
