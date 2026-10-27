- **harness: infrastructure: GO** — 10 of 10 attempts finished without an infrastructure failure; 0 tries re-run (none), 0 attempts given up
- **harness: proxy rows: GO** — 74 requests logged; attempts with a row count different from pi's replies: none; rows without timings: 0
- **harness: first token vs prefill: GO** — proxy first token minus server prefill: median 0.20 s, p90 0.34 s (limit 0.5 s + round trip 0.30 s; includes the per-request fixed cost before prefill)
- **tool calls: GO** — malformed 0 of 68 calls (0.0 %: unknown tool 0, invalid arguments 0, markup in text 0); calls inside the reasoning 0
- **cache: GO** — turns ≥ 2: median cache hit 95.6 %, min 43.5 %; cache paths exact 62, older checkpoint 0, reply re-read 2
- **memory: NO-GO** — lowest MemAvailable 626 MB; swap used 351 → 622 MB
- server log server-rel.log: backend sampling off 0, checkpoint restored 10, checkpoint created 0, checkpoint erased 0, full re-processing 0, screened head 0, NaN / inf 0, watchdog kill 0

**Overall: NO-GO**
