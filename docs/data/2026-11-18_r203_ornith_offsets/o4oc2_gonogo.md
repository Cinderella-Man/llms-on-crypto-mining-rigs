- **harness: infrastructure: GO** — 36 of 36 attempts finished without an infrastructure failure; 0 tries re-run (none), 0 attempts given up
- **harness: proxy rows: GO** — 192 requests logged; attempts with a row count different from pi's replies: none; rows without timings: 0
- **harness: first token vs prefill: GO** — proxy first token minus server prefill: median 0.08 s, p90 0.12 s (limit 0.5 s + round trip 0.30 s; includes the per-request fixed cost before prefill)
- **tool calls: GO** — malformed 0 of 192 calls (0.0 %: unknown tool 0, invalid arguments 0, markup in text 0); calls inside the reasoning 0
- **cache: GO** — turns ≥ 2: median cache hit 95.0 %, min 44.6 %; cache paths exact 145, older checkpoint 0, reply re-read 11
- **memory: NO-GO** — lowest MemAvailable 1183 MB; swap used 0 → 360 MB
- server log server-o4oc0.log: backend sampling off 0, checkpoint restored 16, checkpoint created 0, checkpoint erased 0, full re-processing 0, screened head 0, NaN / inf 0, watchdog kill 0, graph reallocations 0 of 67 requests (0 same size, 0 changed)
- server log server-o4oc1.log: backend sampling off 0, checkpoint restored 12, checkpoint created 0, checkpoint erased 0, full re-processing 0, screened head 0, NaN / inf 0, watchdog kill 0, graph reallocations 0 of 62 requests (0 same size, 0 changed)
- server log server-o4oc2.log: backend sampling off 0, checkpoint restored 16, checkpoint created 0, checkpoint erased 0, full re-processing 0, screened head 0, NaN / inf 0, watchdog kill 0, graph reallocations 0 of 63 requests (0 same size, 0 changed)

**Overall: NO-GO**
