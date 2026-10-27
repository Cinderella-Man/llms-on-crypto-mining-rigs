- **harness: infrastructure: GO** — 12 of 12 attempts finished without an infrastructure failure; 0 tries re-run (none), 0 attempts given up
- **harness: proxy rows: GO** — 73 requests logged; attempts with a row count different from pi's replies: none; rows without timings: 0
- **harness: first token vs prefill: GO** — proxy first token minus server prefill: median 0.18 s, p90 0.32 s (limit 0.5 s + round trip 0.30 s; includes the per-request fixed cost before prefill)
- **tool calls: GO** — malformed 0 of 66 calls (0.0 %: unknown tool 0, invalid arguments 0, markup in text 0); calls inside the reasoning 0
- **cache: GO** — turns ≥ 2: median cache hit 93.3 %, min 25.7 %; cache paths exact 57, older checkpoint 0, reply re-read 4
- **memory: NO-GO** — lowest MemAvailable 581 MB; swap used 355 → 809 MB
- server log server-rel.log: backend sampling off 0, checkpoint restored 13, checkpoint created 0, checkpoint erased 0, full re-processing 0, screened head 0, NaN / inf 0, watchdog kill 0

**Overall: NO-GO**
