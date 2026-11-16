- **harness: infrastructure: GO** — 12 of 12 attempts finished without an infrastructure failure; 0 tries re-run (none), 0 attempts given up
- **harness: proxy rows: GO** — 74 requests logged; attempts with a row count different from pi's replies: none; rows without timings: 0
- **harness: first token vs prefill: GO** — proxy first token minus server prefill: median 0.17 s, p90 0.27 s (limit 0.5 s + round trip 0.30 s; includes the per-request fixed cost before prefill)
- **tool calls: GO** — malformed 0 of 70 calls (0.0 %: unknown tool 0, invalid arguments 0, markup in text 0); calls inside the reasoning 0
- **cache: GO** — turns ≥ 2: median cache hit 94.0 %, min 20.7 %; cache paths exact 58, older checkpoint 0, reply re-read 4
- **memory: NO-GO** — lowest MemAvailable 1104 MB; swap used 17 → 290 MB
- server log server-q5f4t.log: backend sampling off 0, checkpoint restored 14, checkpoint created 0, checkpoint erased 0, full re-processing 0, screened head 0, NaN / inf 0, watchdog kill 0, graph reallocations 73 of 74 requests (295 same size, 0 changed)

**Overall: NO-GO**
