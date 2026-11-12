- **harness: infrastructure: GO** — 24 of 24 attempts finished without an infrastructure failure; 0 tries re-run (none), 0 attempts given up
- **harness: proxy rows: NO-GO** — 161 requests logged; attempts with a row count different from pi's replies: v4c2/bugfix-javascript-complex-numbers/a1/t1 (35 rows / 34 replies); rows without timings: 1
- **harness: first token vs prefill: GO** — proxy first token minus server prefill: median 0.06 s, p90 0.15 s (limit 0.5 s + round trip 0.30 s; includes the per-request fixed cost before prefill)
- **tool calls: GO** — malformed 0 of 150 calls (0.0 %: unknown tool 0, invalid arguments 0, markup in text 0); calls inside the reasoning 0
- **cache: GO** — turns ≥ 2: median cache hit 97.0 %, min 43.3 %; cache paths exact 124, older checkpoint 0, reply re-read 12, unknown 1
- **memory: NO-GO** — lowest MemAvailable 435 MB; swap used 18 → 473 MB
- server log server-gains.log: backend sampling off 0, checkpoint restored 16, checkpoint created 0, checkpoint erased 0, full re-processing 0, screened head 0, NaN / inf 0, watchdog kill 0
- server log server-v4c2.log: backend sampling off 0, checkpoint restored 17, checkpoint created 0, checkpoint erased 1, full re-processing 1, screened head 0, NaN / inf 0, watchdog kill 0

**Overall: NO-GO**
