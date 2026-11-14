- **harness: infrastructure: GO** — 36 of 36 attempts finished without an infrastructure failure; 0 tries re-run (none), 0 attempts given up
- **harness: proxy rows: GO** — 196 requests logged; attempts with a row count different from pi's replies: none; rows without timings: 0
- **harness: first token vs prefill: GO** — proxy first token minus server prefill: median 0.07 s, p90 0.13 s (limit 0.5 s + round trip 0.30 s; includes the per-request fixed cost before prefill)
- **tool calls: GO** — malformed 0 of 197 calls (0.0 %: unknown tool 0, invalid arguments 0, markup in text 0); calls inside the reasoning 0
- **cache: GO** — turns ≥ 2: median cache hit 97.3 %, min 44.6 %; cache paths exact 153, older checkpoint 0, reply re-read 7
- **memory: NO-GO** — lowest MemAvailable 732 MB; swap used 29 → 341 MB
- server log server-o3a.log: backend sampling off 0, checkpoint restored 15, checkpoint created 0, checkpoint erased 0, full re-processing 0, screened head 0, NaN / inf 0, watchdog kill 0, graph reallocations 0 of 71 requests (0 same size, 0 changed)
- server log server-o4a.log: backend sampling off 0, checkpoint restored 13, checkpoint created 0, checkpoint erased 0, full re-processing 0, screened head 0, NaN / inf 0, watchdog kill 0, graph reallocations 0 of 62 requests (0 same size, 0 changed)
- server log server-o4sa.log: backend sampling off 0, checkpoint restored 12, checkpoint created 0, checkpoint erased 0, full re-processing 0, screened head 0, NaN / inf 0, watchdog kill 0, graph reallocations 0 of 63 requests (0 same size, 0 changed)

**Overall: NO-GO**
