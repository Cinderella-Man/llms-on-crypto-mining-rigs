- **harness: infrastructure: GO** — 23 of 24 attempts finished without an infrastructure failure; 1 tries re-run (gains/bugfix-python-scale-generator/a1-t1-infra), 0 attempts given up
- **harness: proxy rows: NO-GO** — 132 requests logged; attempts with a row count different from pi's replies: v3/bugfix-javascript-variable-length-quantity/a1/t1 (2 rows / 1 replies); rows without timings: 1
- **harness: first token vs prefill: GO** — proxy first token minus server prefill: median 0.06 s, p90 0.10 s (limit 0.5 s + round trip 0.30 s; includes the per-request fixed cost before prefill)
- **tool calls: GO** — malformed 0 of 125 calls (0.0 %: unknown tool 0, invalid arguments 0, markup in text 0); calls inside the reasoning 0
- **cache: GO** — turns ≥ 2: median cache hit 95.6 %, min 43.3 %; cache paths exact 104, older checkpoint 0, reply re-read 3, unknown 1
- **memory: NO-GO** — lowest MemAvailable 66 MB; swap used 24 → 964 MB
- server log server-gains.log: backend sampling off 0, checkpoint restored 14, checkpoint created 0, checkpoint erased 0, full re-processing 0, screened head 0, NaN / inf 0, watchdog kill 0, graph reallocations 70 of 70 requests (352 same size, 139 changed)
- server log server-v3.log: backend sampling off 0, checkpoint restored 13, checkpoint created 0, checkpoint erased 0, full re-processing 0, screened head 0, NaN / inf 0, watchdog kill 0, graph reallocations 0 of 65 requests (0 same size, 0 changed)

**Overall: NO-GO**
