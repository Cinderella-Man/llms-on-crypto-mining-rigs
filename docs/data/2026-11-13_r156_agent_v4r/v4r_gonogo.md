- **harness: infrastructure: GO** — 96 of 96 attempts finished without an infrastructure failure; 0 tries re-run (none), 0 attempts given up
- **harness: proxy rows: NO-GO** — 618 requests logged; attempts with a row count different from pi's replies: v4c2/bugfix-javascript-complex-numbers/a1/t1 (35 rows / 34 replies); rows without timings: 1
- **harness: first token vs prefill: GO** — proxy first token minus server prefill: median 0.06 s, p90 0.16 s (limit 0.5 s + round trip 0.30 s; includes the per-request fixed cost before prefill)
- **tool calls: GO** — malformed 0 of 584 calls (0.0 %: unknown tool 0, invalid arguments 0, markup in text 0); calls inside the reasoning 0
- **cache: GO** — turns ≥ 2: median cache hit 96.9 %, min 43.2 %; cache paths exact 489, older checkpoint 0, reply re-read 32, unknown 1
- **memory: NO-GO** — lowest MemAvailable 276 MB; swap used 18 → 713 MB
- server log server-gains.log: backend sampling off 0, checkpoint restored 16, checkpoint created 0, checkpoint erased 0, full re-processing 0, screened head 0, NaN / inf 0, watchdog kill 0, graph reallocations 69 of 69 requests (329 same size, 137 changed)
- server log server-v4c2.log: backend sampling off 0, checkpoint restored 17, checkpoint created 0, checkpoint erased 1, full re-processing 1, screened head 0, NaN / inf 0, watchdog kill 0, graph reallocations 0 of 91 requests (0 same size, 0 changed)
- server log server-v4c22.log: backend sampling off 0, checkpoint restored 13, checkpoint created 0, checkpoint erased 0, full re-processing 0, screened head 0, NaN / inf 0, watchdog kill 0, graph reallocations 0 of 65 requests (0 same size, 0 changed)
- server log server-v4c2g.log: backend sampling off 0, checkpoint restored 15, checkpoint created 0, checkpoint erased 0, full re-processing 0, screened head 0, NaN / inf 0, watchdog kill 0, graph reallocations 0 of 65 requests (0 same size, 0 changed)
- server log server-v4c3.log: backend sampling off 0, checkpoint restored 13, checkpoint created 0, checkpoint erased 0, full re-processing 0, screened head 0, NaN / inf 0, watchdog kill 0, graph reallocations 0 of 66 requests (0 same size, 0 changed)
- server log server-v4c32.log: backend sampling off 0, checkpoint restored 18, checkpoint created 0, checkpoint erased 2, full re-processing 1, screened head 0, NaN / inf 0, watchdog kill 0, graph reallocations 0 of 98 requests (0 same size, 0 changed)
- server log server-v4c3ad.log: backend sampling off 0, checkpoint restored 13, checkpoint created 0, checkpoint erased 0, full re-processing 0, screened head 0, NaN / inf 0, watchdog kill 0, graph reallocations 0 of 68 requests (0 same size, 0 changed)
- server log server-v4r.log: backend sampling off 0, checkpoint restored 12, checkpoint created 0, checkpoint erased 1, full re-processing 1, screened head 0, NaN / inf 0, watchdog kill 0, graph reallocations 0 of 95 requests (0 same size, 0 changed)

**Overall: NO-GO**
