- **harness: infrastructure: GO** — 12 of 12 attempts finished without an infrastructure failure; 0 tries re-run (none), 0 attempts given up
- **harness: proxy rows: GO** — 71 requests logged; attempts with a row count different from pi's replies: none; rows without timings: 0
- **harness: first token vs prefill: GO** — proxy first token minus server prefill: median 0.05 s, p90 0.07 s (limit 0.5 s + round trip 0.30 s; includes the per-request fixed cost before prefill)
- **tool calls: GO** — malformed 0 of 67 calls (0.0 %: unknown tool 0, invalid arguments 0, markup in text 0); calls inside the reasoning 0
- **cache: GO** — turns ≥ 2: median cache hit 97.6 %, min 42.8 %; cache paths exact 59, older checkpoint 0
- **memory: NO-GO** — lowest MemAvailable 229 MB; swap used 52 → 52 MB
- server log server-v4r.log: backend sampling off 0, checkpoint restored 11, checkpoint created 0, checkpoint erased 0, full re-processing 0, screened head 0, NaN / inf 0, watchdog kill 0, graph reallocations 0 of 71 requests (0 same size, 0 changed)

**Overall: NO-GO**
