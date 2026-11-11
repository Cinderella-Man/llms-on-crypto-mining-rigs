- **harness: infrastructure: GO** — 24 of 24 attempts finished without an infrastructure failure; 0 tries re-run (none), 0 attempts given up
- **harness: proxy rows: NO-GO** — 126 requests logged; attempts with a row count different from pi's replies: v3rel/bugfix-javascript-variable-length-quantity/a1/t1 (2 rows / 1 replies), v4night/bugfix-javascript-variable-length-quantity/a1/t1 (6 rows / 5 replies); rows without timings: 2
- **harness: first token vs prefill: GO** — proxy first token minus server prefill: median 0.07 s, p90 0.10 s (limit 0.5 s + round trip 0.30 s; includes the per-request fixed cost before prefill)
- **tool calls: GO** — malformed 0 of 114 calls (0.0 %: unknown tool 0, invalid arguments 0, markup in text 0); calls inside the reasoning 0
- **cache: GO** — turns ≥ 2: median cache hit 94.9 %, min 48.9 %; cache paths exact 95, older checkpoint 0, reply re-read 5, unknown 2
- **memory: NO-GO** — lowest MemAvailable 193 MB; swap used 0 → 1004 MB
- server log server-v3rel.log: backend sampling off 0, checkpoint restored 13, checkpoint created 0, checkpoint erased 0, full re-processing 0, screened head 0, NaN / inf 0, watchdog kill 0
- server log server-v4night.log: backend sampling off 0, checkpoint restored 14, checkpoint created 0, checkpoint erased 0, full re-processing 0, screened head 0, NaN / inf 0, watchdog kill 0

**Overall: NO-GO**
