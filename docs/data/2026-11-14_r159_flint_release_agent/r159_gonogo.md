- **harness: infrastructure: GO** — 48 of 48 attempts finished without an infrastructure failure; 0 tries re-run (none), 0 attempts given up
- **harness: proxy rows: NO-GO** — 287 requests logged; attempts with a row count different from pi's replies: f3a/bugfix-python-scale-generator/a1/t1 (4 rows / 3 replies); rows without timings: 1
- **harness: first token vs prefill: GO** — proxy first token minus server prefill: median 0.17 s, p90 0.31 s (limit 0.5 s + round trip 0.30 s; includes the per-request fixed cost before prefill)
- **tool calls: GO** — malformed 0 of 260 calls (0.0 %: unknown tool 0, invalid arguments 0, markup in text 0); calls inside the reasoning 0
- **cache: GO** — turns ≥ 2: median cache hit 94.4 %, min 23.9 %; cache paths exact 212, older checkpoint 0, reply re-read 26, unknown 1
- **memory: NO-GO** — lowest MemAvailable 873 MB; swap used 30 → 296 MB
- server log server-f3a.log: backend sampling off 0, checkpoint restored 13, checkpoint created 0, checkpoint erased 0, full re-processing 0, screened head 0, NaN / inf 0, watchdog kill 0, graph reallocations 64 of 65 requests (188 same size, 0 changed)
- server log server-f3b.log: backend sampling off 0, checkpoint restored 14, checkpoint created 0, checkpoint erased 0, full re-processing 0, screened head 0, NaN / inf 0, watchdog kill 0, graph reallocations 66 of 68 requests (237 same size, 0 changed)
- server log server-f4a.log: backend sampling off 0, checkpoint restored 16, checkpoint created 0, checkpoint erased 0, full re-processing 0, screened head 0, NaN / inf 0, watchdog kill 0, graph reallocations 71 of 73 requests (323 same size, 0 changed)
- server log server-f4b.log: backend sampling off 0, checkpoint restored 14, checkpoint created 0, checkpoint erased 0, full re-processing 0, screened head 0, NaN / inf 0, watchdog kill 0, graph reallocations 79 of 80 requests (308 same size, 0 changed)

**Overall: NO-GO**
